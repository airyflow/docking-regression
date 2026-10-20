"""Reading embeddings and scores at pool scale.

The embedding arrays run to 1.5 TB, so nothing is ever loaded whole. Two access
patterns are supported:

  gather()      pull an arbitrary index set out of a memmap, reading contiguous
                slabs wherever the sorted indices are dense enough to make that
                cheaper than per-row reads. Lustre punishes scattered reads.

  materialize() do that once for a split and write a compact array, so training
                re-reads a few GB instead of walking the full pool every epoch.
"""

import os
import time

import numpy as np
from pathlib import Path

from .pools import Pool, pool_work


def scores_path(pool: Pool):
    return pool_work(pool.name) / "scores.npy"


def load_scores(pool: Pool, mmap: bool = True) -> np.ndarray:
    p = scores_path(pool)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing -- run: python -m dockreg.prepare --pool {pool.name}"
        )
    return np.load(p, mmap_mode="r" if mmap else None)


def open_embeddings(pool: Pool, backbone: str) -> np.ndarray:
    path = pool.embedding_path(backbone)
    if not path.exists():
        raise FileNotFoundError(f"missing embeddings: {path}")
    arr = np.load(path, mmap_mode="r")
    if arr.shape[0] != pool.n_rows:
        raise ValueError(
            f"{path} has {arr.shape[0]} rows, pool {pool.name} expects "
            f"{pool.n_rows}"
        )
    expected = pool.dim(backbone)
    if arr.shape[1] != expected:
        raise ValueError(
            f"{path} has dim {arr.shape[1]}, registry says {expected}"
        )
    return arr


def gather(
    arr: np.ndarray,
    idx: np.ndarray,
    block: int = 65536,
    density: float = 0.25,
    out: np.ndarray = None,
    progress: bool = False,
) -> np.ndarray:
    """Gather rows `idx` from memmap `arr`, preserving the order of `idx`.

    Indices are processed in sorted order for locality and scattered back into
    the caller's ordering at the end. Within a block, if the rows cover a span
    less than 1/`density` times their count, the whole span is read as one slab.
    """
    idx = np.asarray(idx)
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]

    if out is None:
        out = np.empty((idx.size, arr.shape[1]), dtype=arr.dtype)

    n_blocks = (sorted_idx.size + block - 1) // block
    for b in range(n_blocks):
        sl = slice(b * block, min((b + 1) * block, sorted_idx.size))
        chunk = sorted_idx[sl]
        lo, hi = int(chunk[0]), int(chunk[-1]) + 1
        if (hi - lo) * density <= chunk.size:
            # np.array(), not np.asarray(): the latter returns a memmap view
            # and the fancy-index below would page-fault scattered rows.
            slab = np.array(arr[lo:hi])
            out[order[sl]] = slab[chunk - lo]
        else:
            out[order[sl]] = arr[chunk]
        if progress and (b % 50 == 0 or b == n_blocks - 1):
            print(f"    gather {b + 1}/{n_blocks} blocks", flush=True)
    return out


def gather_stream(
    arr: np.ndarray,
    idx: np.ndarray,
    out: np.ndarray = None,
    chunk_rows: int = 262144,
    progress: bool = False,
) -> np.ndarray:
    """Gather rows `idx` by streaming `arr` front to back in large slabs.

    Reads the whole array, including rows nobody asked for, but reads it
    sequentially. That is the right trade whenever the wanted rows are spread
    thinly enough that most slabs contain at least one of them -- which, for a
    random subset of a 10^8-row pool, is essentially always.

    Measured on d4 (116M x 512, Lustre): the per-row path in gather() did not
    finish 100k scattered rows in 10 minutes. Streaming the same file end to
    end is a few minutes.
    """
    idx = np.asarray(idx)
    order = np.argsort(idx, kind="stable")
    sorted_idx = idx[order]
    n_total = arr.shape[0]

    if out is None:
        out = np.empty((idx.size, arr.shape[1]), dtype=arr.dtype)

    n_chunks = (n_total + chunk_rows - 1) // chunk_rows
    t0 = time.time()
    done = 0
    for c in range(n_chunks):
        lo = c * chunk_rows
        hi = min(lo + chunk_rows, n_total)
        j0 = int(np.searchsorted(sorted_idx, lo, side="left"))
        j1 = int(np.searchsorted(sorted_idx, hi, side="left"))
        if j1 <= j0:
            continue
        # np.array() forces an actual sequential read of the window.
        # np.asarray() would return a memmap *view*, and the fancy-index on
        # the next line would then page-fault scattered rows -- which is the
        # very access pattern this function exists to avoid. Measured: the
        # view form ran at ~23 MB/s on d4.
        slab = np.array(arr[lo:hi])
        out[order[j0:j1]] = slab[sorted_idx[j0:j1] - lo]
        done += j1 - j0
        if progress and (c % 200 == 0 or c == n_chunks - 1):
            el = time.time() - t0
            frac = (c + 1) / n_chunks
            eta = el / frac - el if frac > 0 else 0
            print(f"    stream {c + 1}/{n_chunks} slabs, {done:,}/{idx.size:,} "
                  f"rows, {el / 60:.1f} min elapsed, ~{eta / 60:.1f} min left",
                  flush=True)
    return out


# Below this density the wanted rows are too sparse for streaming to pay, and
# the scattered per-row path in gather() wins. Above it, stream.
STREAM_DENSITY = 1e-3


def gather_auto(arr, idx, out=None, progress=False, **kw):
    density = idx.size / arr.shape[0]
    if density >= STREAM_DENSITY:
        return gather_stream(arr, idx, out=out, progress=progress)
    return gather(arr, idx, out=out, progress=progress, **kw)


def cache_path(pool: Pool, backbone: str, split_tag: str, part: str):
    d = pool_work(pool.name) / "cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{backbone}__{split_tag}__{part}.npy"


def materialize(
    pool: Pool,
    backbone: str,
    idx: np.ndarray,
    out_path,
    block: int = 65536,
    progress: bool = True,
    ram_budget_gb: float = 64.0,
) -> np.ndarray:
    """Write the rows `idx` of one backbone to a compact .npy, in `idx` order."""
    arr = open_embeddings(pool, backbone)
    n, dim = idx.size, arr.shape[1]
    gb = n * dim * 4 / 1e9
    if progress:
        print(f"  materializing {n:,} x {dim} ({gb:.2f} GB) -> {out_path}",
              flush=True)

    # Build in RAM, then write once sequentially. gather() scatters its writes
    # to restore `idx` order; doing that straight into a disk-backed memmap
    # turns one streaming write into millions of scattered page writes, which
    # on Lustre costs hours rather than minutes.
    if gb <= ram_budget_gb:
        out = np.empty((n, dim), dtype=np.float32)
        gather_auto(arr, idx, out=out, progress=progress, block=block)
        # Write to a pid-unique temp file, then rename. os.replace is atomic
        # within a filesystem, so a concurrent or interrupted build can never
        # leave a half-written .npy that later reads as corrupt. Two workers
        # racing now produce one valid file instead of one broken one --
        # measured the hard way: 16 parallel workers destroyed a 25.8 GB
        # GROVER cache on 2026-10-18.
        out_path = Path(out_path)
        tmp = out_path.with_name(out_path.stem + f".tmp{os.getpid()}.npy")
        np.save(tmp, out)
        os.replace(tmp, out_path)
        return out

    if progress:
        print(f"  ({gb:.1f} GB exceeds the {ram_budget_gb:.0f} GB RAM budget; "
              f"falling back to a memmapped build)", flush=True)
    out = np.lib.format.open_memmap(
        out_path, mode="w+", dtype=np.float32, shape=(n, dim)
    )
    gather_auto(arr, idx, out=out, progress=progress, block=block)
    out.flush()
    return out


def ensure_cached(
    pool: Pool, backbone: str, idx: np.ndarray, split_tag: str, part: str,
    rebuild: bool = False, progress: bool = True, ram_budget_gb: float = 64.0,
):
    path = cache_path(pool, backbone, split_tag, part)
    if path.exists() and not rebuild:
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] == idx.size:
            if progress:
                print(f"  cache hit: {path.name} ({arr.shape[0]:,} rows)",
                      flush=True)
            return arr
        print(f"  cache stale ({arr.shape[0]:,} != {idx.size:,}), rebuilding",
              flush=True)
    materialize(pool, backbone, idx, path, progress=progress,
                ram_budget_gb=ram_budget_gb)
    return np.load(path, mmap_mode="r")


class Standardizer:
    """Per-feature z-scoring, fitted on a sample of the training rows."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)

    @classmethod
    def fit(cls, X, max_rows: int = 200_000, seed: int = 0):
        n = X.shape[0]
        if n > max_rows:
            rs = np.random.default_rng(seed)
            sel = np.sort(rs.choice(n, size=max_rows, replace=False))
            sample = np.asarray(X[sel], dtype=np.float32)
        else:
            sample = np.asarray(X, dtype=np.float32)
        mean = sample.mean(axis=0)
        std = sample.std(axis=0)
        std[std < 1e-6] = 1.0
        return cls(mean, std)

    def transform(self, X):
        return (np.asarray(X, dtype=np.float32) - self.mean) / self.std

    def to_dict(self):
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])
