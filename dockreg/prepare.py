"""Turn a pool's gzipped score CSV into a float32 .npy in embedding row order.

Run once per pool. Reading 100M rows out of gzip on every job would be absurd;
a float32 array is 4 bytes a row and indexes instantly.

By default the SMILES column is checked against the pool's reference SMILES
file row by row. That check is the whole reason this step exists as its own
command: a silent off-by-one between scores and embeddings produces a model
that trains happily and means nothing.
"""

import argparse
import gzip
import sys

import numpy as np

from .data import scores_path
from .pools import POOLS, get_pool


def read_scores(pool, verify: bool = True, progress_every: int = 5_000_000):
    """Stream the score CSV; return scores in *CSV* row order."""
    scores = np.empty(pool.n_rows, dtype=np.float32)

    ref_path = pool.scores_smiles_path or pool.smiles_path
    ref = open(ref_path) if verify else None
    if verify:
        print(f"  verifying SMILES against {ref_path}", flush=True)

    n = 0
    with gzip.open(pool.scores_csv, "rt") as f:
        header = f.readline().rstrip("\n")
        if not header.lower().startswith("smiles"):
            raise ValueError(f"unexpected header in {pool.scores_csv}: {header!r}")
        for line in f:
            smi, _, val = line.rstrip("\n").rpartition(",")
            if n >= pool.n_rows:
                raise ValueError(
                    f"{pool.scores_csv} has more than {pool.n_rows} rows"
                )
            scores[n] = float(val)
            if ref is not None:
                want = ref.readline().rstrip("\n")
                if want != smi:
                    raise ValueError(
                        f"SMILES mismatch at row {n}:\n"
                        f"  csv : {smi}\n  file: {want}"
                    )
            n += 1
            if progress_every and n % progress_every == 0:
                print(f"    {n:,} rows", flush=True)
    if ref is not None:
        ref.close()

    if n != pool.n_rows:
        raise ValueError(
            f"{pool.scores_csv} yielded {n} rows, registry expects {pool.n_rows}"
        )
    return scores


def to_embedding_order(pool, scores: np.ndarray) -> np.ndarray:
    """Reorder CSV-order scores to match the embedding rows."""
    if pool.perm_path is None:
        return scores
    perm = np.load(pool.perm_path)
    if perm.shape[0] != pool.n_rows:
        raise ValueError(f"{pool.perm_path} length {perm.shape[0]} != {pool.n_rows}")
    # embedding row i corresponds to csv row perm[i]
    print(f"  applying permutation {pool.perm_path.name}", flush=True)
    return scores[perm]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the row-by-row SMILES check (not advised)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing scores.npy")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    out = scores_path(pool)
    if out.exists() and not args.force:
        arr = np.load(out, mmap_mode="r")
        print(f"{out} already exists ({arr.shape[0]:,} rows). Use --force.")
        return 0

    print(f"pool {pool.name}: {pool.target} / {pool.oracle}, "
          f"{pool.n_rows:,} rows")
    print(f"  scores: {pool.scores_csv}")
    scores = read_scores(pool, verify=not args.no_verify)
    scores = to_embedding_order(pool, scores)

    np.save(out, scores)
    finite = np.isfinite(scores)
    print(f"\nwrote {out}")
    print(f"  min={scores[finite].min():.2f}  max={scores[finite].max():.2f}  "
          f"mean={scores[finite].mean():.2f}")
    print(f"  non-finite={int((~finite).sum()):,}  "
          f"zero={int((scores == 0).sum()):,}  "
          f"positive={int((scores > 0).sum()):,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
