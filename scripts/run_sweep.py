"""Learning-curve sweep driver.

Resumable, but a run only counts as done when its recorded `epochs` and
`patience` match what is being asked for. Skipping on "metrics.json exists"
alone silently promotes a short debug run into the results tables -- which is
exactly what happened to the 50k descriptor baseline on 2026-10-16.

Usage:
  python scripts/run_sweep.py --pool enhits
  python scripts/run_sweep.py --pool d4 --budgets 50000,200000,1000000,5000000 \
      --cache-budget 5000000
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dockreg.pools import get_pool, pool_work  # noqa: E402
from dockreg.splits import load_split, split_tag  # noqa: E402

# Small-dimension backbones first, so the cheap curves land before the
# expensive gathers start.
DEFAULT_BACKBONES = ["descriptors", "unimol", "molformer", "smited",
                     "unimol2", "mhgged", "grover3400"]


def run_is_current(run_dir: Path, epochs: int, patience: int,
                   cache_budget: int) -> bool:
    f = run_dir / "metrics.json"
    if not f.exists():
        return False
    try:
        m = json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("epochs") != epochs or m.get("patience") != patience:
        return False
    if cache_budget and m.get("cache_budget") != cache_budget:
        return False
    # A run that stopped early is complete; one that used every epoch without
    # early stopping may simply have been cut short by a smaller --epochs.
    return True


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--backbones", default=None,
                    help="comma-separated; default is every backbone the pool "
                         "has, plus the descriptor baseline")
    ap.add_argument("--budgets", default="50000,200000,1000000,0",
                    help="comma-separated training sizes; 0 means 'all cached'")
    ap.add_argument("--cache-budget", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--materialize-ram-gb", type=float, default=64.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    tag = split_tag(0, 0.05, 0.05)
    train_n = load_split(pool, tag)["train"].size
    cache_rows = min(args.cache_budget, train_n) if args.cache_budget else train_n

    if args.backbones:
        backbones = [b for b in args.backbones.split(",") if b]
    else:
        backbones = [b for b in DEFAULT_BACKBONES
                     if b == "descriptors" or b in pool.backbones]

    budgets = []
    for b in (int(x) for x in args.budgets.split(",")):
        n = cache_rows if b == 0 else min(b, cache_rows)
        if n not in budgets:
            budgets.append(n)

    logdir = pool_work(pool.name) / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    runs_dir = pool_work(pool.name) / "runs"

    print(f"pool={pool.name}  train split={train_n:,}  cache={cache_rows:,}")
    print(f"backbones: {', '.join(backbones)}")
    print(f"budgets:   {', '.join(f'{b:,}' for b in budgets)}\n")

    for bb in backbones:
        for n in budgets:
            name = f"{bb}__{tag}__n{n}__s{args.seed}"
            if run_is_current(runs_dir / name, args.epochs, args.patience,
                              args.cache_budget):
                print(f"[skip] {name}")
                continue
            if args.dry_run:
                print(f"[dry ] {name}")
                continue
            print(f"[run ] {bb} n={n:,}  {time.strftime('%H:%M:%S')}",
                  flush=True)
            cmd = [
                sys.executable, "-m", "dockreg.train",
                "--pool", pool.name, "--backbone", bb,
                "--max-train", str(n), "--epochs", str(args.epochs),
                "--patience", str(args.patience), "--seed", str(args.seed),
                "--materialize-ram-gb", str(args.materialize_ram_gb),
            ]
            if args.cache_budget:
                cmd += ["--cache-budget", str(args.cache_budget)]
            log = logdir / f"{name}.log"
            with open(log, "w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
            if rc != 0:
                print(f"[FAIL] {name} (exit {rc}) -- see {log}")
                print("".join(open(log).readlines()[-5:]))
                continue
            for line in open(log):
                if line.startswith("  test:"):
                    print(f"       {line.rstrip()}")

    print(f"\nsweep done {time.strftime('%H:%M:%S')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
