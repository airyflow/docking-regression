"""Screen the matched 1536-d Uni-Mol2 against the inherited 768-d one.

The ENHITS 768-d Uni-Mol2 is the best single model in every result in this
repo and appears in every winning fusion combination, yet `al-molecular`'s
provenance work shows it was extracted by a third party with different
settings and has effective rank 2.2 of 768. A matched 1536-d extraction of the
same pool exists at

  al-molecular/results/embed/EnamineHTS/unimol2_embeddings.npy   (2,141,500 x 1536)

whose canonical SMILES file joins to the 2,104,318-row scored pool with every
scored molecule matching exactly one embedding row, no duplicate SMILES on
either side, and 37,182 unused extra rows.

This builds the split caches for that file under a distinct backbone name and
leaves training to `dockreg.train`, so the comparison runs through the same
code path, split and head as every other backbone.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dockreg.data import cache_path, gather_stream  # noqa: E402
from dockreg.pools import get_pool  # noqa: E402
from dockreg.splits import load_split, split_tag  # noqa: E402

SRC = Path("/N/slate/mengjing/repos/al-molecular/results/embed/EnamineHTS")
EMB = SRC / "unimol2_embeddings.npy"
SMI = SRC / "enamine_smiles_canonical.txt"
BACKBONE = "unimol2_1536"          # distinct name; never collides with the 768-d


def build_map(pool):
    """For each scored-pool row, the row index in the 1536-d file."""
    src = [l.rstrip("\n") for l in open(SMI)]
    pool_smiles = [l.rstrip("\n") for l in open(pool.smiles_path)]
    if len(set(src)) != len(src):
        raise ValueError("duplicate SMILES in the 1536-d set; join is ambiguous")
    idx = {s: i for i, s in enumerate(src)}
    try:
        m = np.array([idx[s] for s in pool_smiles], dtype=np.int64)
    except KeyError as e:
        raise ValueError(f"scored molecule absent from the 1536-d set: {e}")
    print(f"  join: {m.size:,} scored rows -> unique rows of "
          f"{len(src):,} ({len(src) - m.size:,} unused)")
    return m


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--max-train", type=int, default=0, help="0 = full split")
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pool = get_pool("enhits")
    tag = split_tag(0, 0.05, 0.05)
    sp = load_split(pool, tag)
    tr, va, te = sp["train"], sp["val"], sp["test"]
    n_tr = min(args.max_train, tr.size) if args.max_train else tr.size
    n_va = min(args.max_eval, va.size) if args.max_eval else va.size
    n_te = min(args.max_eval, te.size) if args.max_eval else te.size
    tr, va, te = tr[:n_tr], va[:n_va], te[:n_te]

    arr = np.load(EMB, mmap_mode="r")
    print(f"source {EMB.name}: {arr.shape} {arr.dtype}")
    m = build_map(pool)

    cache_tag = tag if not args.max_train else f"{tag}__c{n_tr}"
    paths = {p: cache_path(pool, BACKBONE, cache_tag, p)
             for p in ("train", "val", "test")}
    if not args.force and all(
        p.exists() and np.load(p, mmap_mode="r").shape[0] == n
        for p, n in zip(paths.values(), (n_tr, n_va, n_te))
    ):
        print("caches already present; use --force to rebuild")
        return 0

    # split indices address the scored pool; map them into the 1536-d file
    src_idx = np.concatenate([m[tr], m[va], m[te]])
    print(f"gathering {src_idx.size:,} rows ({src_idx.size * arr.shape[1] * 4 / 1e9:.1f} GB)")
    out = gather_stream(arr, src_idx, progress=True)

    for part, lo, hi in (("train", 0, n_tr), ("val", n_tr, n_tr + n_va),
                         ("test", n_tr + n_va, n_tr + n_va + n_te)):
        np.save(paths[part], out[lo:hi])
        print(f"  wrote {paths[part].name} ({hi - lo:,} rows)")

    print(f"\nnow run:\n  python -m dockreg.train --pool enhits "
          f"--backbone {BACKBONE} --epochs 40 --patience 6")
    return 0


if __name__ == "__main__":
    sys.exit(main())
