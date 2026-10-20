"""Verify every cached embedding block is readable and the expected size.

Written after 16 parallel workers raced to rebuild the same 25.8 GB GROVER
cache and left a file that could not be opened at all. A corrupt or silently
truncated cache does not raise until something downstream tries to use it, and
by then the failure looks like a modelling problem rather than a storage one.

  python scripts/check_caches.py --pool enhits          # report
  python scripts/check_caches.py --pool enhits --delete # remove the bad ones
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dockreg.pools import POOLS, get_pool, pool_work  # noqa: E402
from dockreg.splits import load_split, split_tag  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="enhits", choices=sorted(POOLS))
    ap.add_argument("--delete", action="store_true",
                    help="remove caches that are corrupt or wrongly sized")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    tag = split_tag(0, 0.05, 0.05)
    sp = load_split(pool, tag)
    expect = {"train": sp["train"].size, "val": sp["val"].size,
              "test": sp["test"].size}

    cache_dir = pool_work(pool.name) / "cache"
    files = sorted(cache_dir.glob("*.npy"))
    if not files:
        print(f"no caches under {cache_dir}")
        return 0

    bad = []
    print(f"{'file':<62}{'size':>9}  status")
    print("-" * 92)
    for f in files:
        name = f.name
        part = name.rsplit("__", 1)[-1].replace(".npy", "")
        # a "__c<N>" tag means the cache was deliberately capped
        capped = "__c" in name
        gb = os.path.getsize(f) / 1e9
        try:
            a = np.load(f, mmap_mode="r")
        except Exception as e:                      # noqa: BLE001
            print(f"{name:<62}{gb:>8.1f}G  CORRUPT ({type(e).__name__})")
            bad.append(f)
            continue
        want = expect.get(part)
        if want is None:
            status = "ok (unrecognised part)"
        elif capped or part != "train":
            status = f"ok ({a.shape[0]:,} rows)"
        elif a.shape[0] != want:
            status = f"TRUNCATED {a.shape[0]:,} != {want:,}"
            bad.append(f)
        else:
            status = f"ok ({a.shape[0]:,} rows)"
        print(f"{name:<62}{gb:>8.1f}G  {status}")

    print(f"\n{len(files)} caches, {len(bad)} bad")
    if bad and args.delete:
        for f in bad:
            f.unlink()
            print(f"  deleted {f.name}")
        print("Rebuild with scripts/prep_cache.py or let the next run do it "
              "serially.")
    elif bad:
        print("Re-run with --delete to remove them.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
