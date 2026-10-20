"""Late fusion: average the test predictions of several trained backbones.

This is the cheap half of the multimodality question. Each backbone is trained
once, independently, and any subset can then be combined without retraining --
which is exactly why late fusion beat intermediate fusion in the Nature Comms
benchmark it is borrowed from.

Runs must share a split: the test indices are compared element-wise and a
mismatch is an error, not a warning.
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

from .data import load_scores
from .metrics import all_metrics, format_metrics
from .pools import get_pool


def load_run(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    meta = json.loads((run_dir / "metrics.json").read_text())
    return {
        "dir": run_dir,
        "name": run_dir.name,
        "backbone": meta["backbone"],
        "pool": meta["pool"],
        "split_tag": meta["split_tag"],
        "meta": meta,
        "test_pred": np.load(run_dir / "test_pred.npy"),
        "test_idx": np.load(run_dir / "test_idx.npy"),
    }


def check_aligned(runs):
    pools = {r["pool"] for r in runs}
    if len(pools) != 1:
        raise ValueError(f"runs span multiple pools: {sorted(pools)}")
    tags = {r["split_tag"] for r in runs}
    if len(tags) != 1:
        raise ValueError(f"runs span multiple splits: {sorted(tags)}")
    ref = runs[0]["test_idx"]
    for r in runs[1:]:
        if r["test_idx"].shape != ref.shape or not np.array_equal(
            r["test_idx"], ref
        ):
            raise ValueError(
                f"test indices of {r['name']} differ from {runs[0]['name']}"
            )
    return runs[0]["pool"], ref


def fuse(preds, mode: str = "mean") -> np.ndarray:
    P = np.stack(preds).astype(np.float64)
    if mode == "mean":
        return P.mean(axis=0)
    if mode == "zmean":
        mu = P.mean(axis=1, keepdims=True)
        sd = P.std(axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return ((P - mu) / sd).mean(axis=0)
    if mode == "rank":
        R = np.stack([np.argsort(np.argsort(p)) for p in P]).astype(np.float64)
        return R.mean(axis=0)
    raise ValueError(f"unknown fusion mode {mode!r}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+", help="run directories")
    ap.add_argument("--mode", default="mean", choices=("mean", "zmean", "rank"),
                    help="mean: average scores (keeps units, comparable RMSE). "
                         "zmean/rank: ranking only, RMSE meaningless.")
    ap.add_argument("--all-subsets", action="store_true",
                    help="evaluate every non-empty subset of the runs")
    ap.add_argument("--out", default=None, help="write results JSON here")
    args = ap.parse_args(argv)

    runs = [load_run(r) for r in args.runs]
    pool_name, test_idx = check_aligned(runs)
    pool = get_pool(pool_name)
    y = np.asarray(load_scores(pool)[test_idx], dtype=np.float64)

    print(f"pool={pool_name}  split={runs[0]['split_tag']}  "
          f"test={test_idx.size:,}  mode={args.mode}")
    print("\nindividual backbones:")
    singles = {}
    for r in runs:
        m = all_metrics(y, r["test_pred"])
        singles[r["backbone"]] = m
        print(f"  {r['backbone']:12s} dim={r['meta']['dim']:5d}  "
              f"{format_metrics(m)}")

    results = {"pool": pool_name, "split_tag": runs[0]["split_tag"],
               "mode": args.mode, "n_test": int(test_idx.size),
               "individual": singles, "fused": {}}

    if args.all_subsets:
        print("\nsubsets:")
        combos = []
        for k in range(2, len(runs) + 1):
            combos.extend(itertools.combinations(range(len(runs)), k))
    else:
        combos = [tuple(range(len(runs)))]
        print("\nfused:")

    rows = []
    for combo in combos:
        preds = [runs[i]["test_pred"] for i in combo]
        names = [runs[i]["backbone"] for i in combo]
        m = all_metrics(y, fuse(preds, args.mode))
        key = "+".join(names)
        results["fused"][key] = m
        rows.append((key, m))

    if args.all_subsets:
        rows.sort(key=lambda kv: -kv[1]["spearman"])
        for key, m in rows[:25]:
            print(f"  {key:60s} {format_metrics(m)}")
        if len(rows) > 25:
            print(f"  ... {len(rows) - 25} more (see --out JSON)")
    else:
        for key, m in rows:
            print(f"  {key}\n    {format_metrics(m)}")

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
