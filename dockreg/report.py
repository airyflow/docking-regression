"""Collect finished runs into a table.

Reads every metrics.json under a runs directory and prints one row per run,
with the descriptor baseline pulled to the top of each budget group so the
comparison that matters is always in view.
"""

import argparse
import json
import sys
from pathlib import Path

from .pools import pool_work

COLUMNS = [
    ("pearson", "pearson", "{:>10.4f}"),
    ("spearman", "spearman", "{:>10.4f}"),
    ("rmse", "rmse", "{:>10.4f}"),
    ("r2", "r2", "{:>10.4f}"),
    ("recall@0.001", "rec@.1%", "{:>10.4f}"),
    ("recall@0.01", "rec@1%", "{:>10.4f}"),
    ("ef@0.01", "EF@1%", "{:>10.1f}"),
]

BASELINE = "descriptors"


def load_runs(runs_dir: Path):
    out = []
    for p in sorted(runs_dir.glob("*/metrics.json")):
        try:
            out.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  skipping {p}: {e}", file=sys.stderr)
    return out


def table(records, split: str = "test_metrics"):
    if not records:
        return "no runs found"
    by_budget = {}
    for r in records:
        by_budget.setdefault(r["n_train"], []).append(r)

    head = (f"{'backbone':<14}{'dim':>6}{'epochs':>7}"
            + "".join(f"{lbl:>10}" for _, lbl, _ in COLUMNS))
    lines = []
    for budget in sorted(by_budget):
        rows = by_budget[budget]
        # baseline first, then by descending Pearson
        rows.sort(key=lambda r: (r["backbone"] != BASELINE,
                                 -r[split].get("pearson", 0)))
        base = next((r for r in rows if r["backbone"] == BASELINE), None)
        lines.append(f"\ntrain n={budget:,}   (test n={rows[0]['n_test']:,})")
        lines.append(head)
        lines.append("-" * len(head))
        for r in rows:
            m = r[split]
            cells = "".join(
                fmt.format(m[key]) if key in m else " " * 10
                for key, _, fmt in COLUMNS
            )
            mark = ""
            if base is not None and r["backbone"] != BASELINE:
                delta = m.get("pearson", 0) - base[split].get("pearson", 0)
                mark = "  +" if delta > 0 else "  -"
                mark += f"{abs(delta):.3f} vs baseline"
            lines.append(
                f"{r['backbone']:<14}{r['dim']:>6}{r.get('epochs_run', 0):>7}"
                f"{cells}{mark}"
            )
    return "\n".join(lines)


def to_csv(records, path, split: str = "test_metrics"):
    import csv

    keys = ["backbone", "dim", "n_train", "n_test", "epochs_run", "best_epoch"]
    metric_keys = sorted({k for r in records for k in r[split]})
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys + metric_keys)
        for r in sorted(records, key=lambda r: (r["n_train"], r["backbone"])):
            w.writerow([r.get(k, "") for k in keys]
                       + [r[split].get(k, "") for k in metric_keys])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="enhits")
    ap.add_argument("--runs-dir", default=None)
    ap.add_argument("--split", default="test_metrics",
                    choices=("test_metrics", "val_metrics"))
    ap.add_argument("--csv", default=None)
    args = ap.parse_args(argv)

    runs_dir = Path(args.runs_dir or (pool_work(args.pool) / "runs"))
    records = load_runs(runs_dir)
    print(f"{len(records)} run(s) in {runs_dir}")
    print(table(records, args.split))
    if args.csv:
        to_csv(records, args.csv, args.split)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
