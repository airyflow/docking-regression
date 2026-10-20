"""Deterministic train/val/test index splits, shared across backbones.

One split file per (pool, seed) is used by every backbone, so a backbone
comparison never confounds itself with a different set of rows.

Failed docks are dropped here rather than at training time, so that train, val
and test all see the same universe and the filter is recorded alongside the
indices instead of living in someone's command line.

Indices are stored in random order. Take the first N for a smaller training
budget; sort before reading from a memmap.
"""

import argparse
import sys

import numpy as np

from .data import load_scores
from .pools import POOLS, get_pool, pool_work


def split_tag(seed: int, val_frac: float, test_frac: float) -> str:
    return f"seed{seed}_v{val_frac:g}_t{test_frac:g}"


def split_path(pool, tag: str):
    return pool_work(pool.name) / f"split_{tag}.npz"


def make_splits(
    pool,
    seed: int = 0,
    val_frac: float = 0.05,
    test_frac: float = 0.05,
    drop_score_above: float = None,
    drop_nonfinite: bool = True,
):
    scores = np.asarray(load_scores(pool))
    n = scores.shape[0]

    keep = np.ones(n, dtype=bool)
    n_nonfinite = 0
    if drop_nonfinite:
        finite = np.isfinite(scores)
        n_nonfinite = int((~finite).sum())
        keep &= finite
    n_failed = 0
    if drop_score_above is not None:
        # Strictly below: a docking score of exactly 0 is a failure too.
        ok = scores < drop_score_above
        n_failed = int((~ok & keep).sum())
        keep &= ok

    universe = np.flatnonzero(keep).astype(np.int64)
    kept = universe.size
    print(f"  rows {n:,} -> kept {kept:,} "
          f"(dropped {n_nonfinite:,} non-finite, {n_failed:,} above "
          f"{drop_score_above})")

    rs = np.random.default_rng(seed)
    perm = rs.permutation(kept)
    n_test = int(round(kept * test_frac))
    n_val = int(round(kept * val_frac))
    if n_test + n_val >= kept:
        raise ValueError("val_frac + test_frac leaves no training rows")

    test = universe[perm[:n_test]]
    val = universe[perm[n_test:n_test + n_val]]
    train = universe[perm[n_test + n_val:]]

    dtype = np.int32 if n < 2**31 else np.int64
    return {
        "train": train.astype(dtype),
        "val": val.astype(dtype),
        "test": test.astype(dtype),
        "meta": np.array(
            [seed, val_frac, test_frac,
             np.nan if drop_score_above is None else drop_score_above,
             n, kept, n_nonfinite, n_failed],
            dtype=np.float64,
        ),
    }


def load_split(pool, tag: str):
    p = split_path(pool, tag)
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing -- run: python -m dockreg.splits --pool {pool.name}"
        )
    z = np.load(p)
    return {k: z[k] for k in ("train", "val", "test", "meta")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument(
        "--drop-score-above", type=float, default=0.0,
        help="drop rows whose score is at or above this (failed/clashing "
             "docks); pass 'inf' to keep everything. Default 0.0.",
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    thr = None if np.isinf(args.drop_score_above) else args.drop_score_above
    tag = split_tag(args.seed, args.val_frac, args.test_frac)
    out = split_path(pool, tag)
    if out.exists() and not args.force:
        print(f"{out} already exists. Use --force.")
        return 0

    print(f"pool {pool.name}  split {tag}")
    sp = make_splits(pool, seed=args.seed, val_frac=args.val_frac,
                     test_frac=args.test_frac, drop_score_above=thr)
    np.savez(out, **sp)
    print(f"  train={sp['train'].size:,}  val={sp['val'].size:,}  "
          f"test={sp['test'].size:,}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
