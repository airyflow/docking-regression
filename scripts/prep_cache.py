"""Build train/val/test embedding caches for a pool in a single streaming pass.

`dockreg.train` materialises each split part on demand. On a 10^8-row pool that
means three separate sequential scans of a file that can be 1.5 TB, because the
wanted rows are too sparse for the scattered-read path (measured: 100k random
rows from d4 did not finish in 10 minutes).

This script gathers the union of the three index sets in one pass and writes
the three cache files under the names `dockreg.data.cache_path` expects, so a
subsequent training run is a cache hit.

  python scripts/prep_cache.py --pool d4 --backbone unimol \
      --cache-budget 500000 --max-eval 200000
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dockreg.data import cache_path, gather_stream, open_embeddings  # noqa: E402
from dockreg.pools import POOLS, get_pool  # noqa: E402
from dockreg.splits import load_split, split_tag  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--cache-budget", type=int, default=0,
                    help="training rows to cache (0 = whole split)")
    ap.add_argument("--max-eval", type=int, default=0,
                    help="val/test rows to cache (0 = whole split)")
    ap.add_argument("--chunk-rows", type=int, default=262144)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    tag = split_tag(0, 0.05, 0.05)
    sp = load_split(pool, tag)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    n_tr = min(args.cache_budget, tr.size) if args.cache_budget else tr.size
    n_va = min(args.max_eval, va.size) if args.max_eval else va.size
    n_te = min(args.max_eval, te.size) if args.max_eval else te.size
    tr, va, te = tr[:n_tr], va[:n_va], te[:n_te]

    cache_tag = tag if not args.cache_budget else f"{tag}__c{n_tr}"
    paths = {p: cache_path(pool, args.backbone, cache_tag, p)
             for p in ("train", "val", "test")}

    if not args.force and all(
        p.exists() and np.load(p, mmap_mode="r").shape[0] == n
        for p, n in zip(paths.values(), (n_tr, n_va, n_te))
    ):
        print("all three caches already present and correctly sized; nothing "
              "to do (use --force to rebuild)")
        return 0

    arr = open_embeddings(pool, args.backbone)
    dim = arr.shape[1]
    total_gb = arr.shape[0] * dim * 4 / 1e9
    out_gb = (n_tr + n_va + n_te) * dim * 4 / 1e9
    print(f"pool={pool.name} backbone={args.backbone} dim={dim}")
    print(f"  source {arr.shape[0]:,} rows, {total_gb:.0f} GB")
    print(f"  caching train={n_tr:,} val={n_va:,} test={n_te:,} "
          f"({out_gb:.1f} GB) in one pass")

    idx = np.concatenate([tr, va, te])
    t0 = time.time()
    out = gather_stream(arr, idx, chunk_rows=args.chunk_rows, progress=True)
    print(f"  gathered in {(time.time() - t0) / 60:.1f} min "
          f"({total_gb / max(time.time() - t0, 1):.2f} GB/s source read)")

    for part, lo, hi in (("train", 0, n_tr),
                         ("val", n_tr, n_tr + n_va),
                         ("test", n_tr + n_va, n_tr + n_va + n_te)):
        np.save(paths[part], out[lo:hi])
        print(f"  wrote {paths[part].name} ({hi - lo:,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
