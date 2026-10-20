"""Size-controlled evaluation.

DOCK and Glide scores scale with molecular size, so a raw Pearson of 0.89 can
be mostly an elaborate way of counting heavy atoms. This module asks what is
left once size is removed, three ways:

  partial correlation   corr(y, pred) with heavy-atom count partialled out of
                        both. One number, linear in size.

  residual correlation  fit y ~ poly(size) and pred ~ poly(size), correlate the
                        residuals. Same idea, but size is allowed to act
                        non-linearly.

  within-bin metrics    split the test set into equal-count heavy-atom bins and
                        score inside each. Nothing linear is assumed, and
                        within-bin enrichment answers the question a screener
                        actually has: among molecules of the size I want, can
                        this model pick the binders?

Runs no models. It reads the predictions each training run already saved, so
it applies to finished work retroactively.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from .data import load_scores
from .descriptors import descriptor_path, ensure_descriptors
from .metrics import topk_recall
from .pools import POOLS, get_pool, pool_work
from .splits import load_split

HEAVY_ATOM_COL = 0  # first feature in both the RDKit and fallback sets


def partial_corr(y, p, s) -> float:
    """corr(y, p) with s partialled out of both."""
    r_yp = np.corrcoef(y, p)[0, 1]
    r_ys = np.corrcoef(y, s)[0, 1]
    r_ps = np.corrcoef(p, s)[0, 1]
    denom = np.sqrt((1 - r_ys**2) * (1 - r_ps**2))
    return float(r_yp - r_ys * r_ps) / float(denom) if denom > 1e-12 else np.nan


def _residualize(v, s, degree: int):
    A = np.vander(s, degree + 1)
    coef, *_ = np.linalg.lstsq(A, v, rcond=None)
    return v - A @ coef


def residual_corr(y, p, s, degree: int = 3) -> float:
    ry = _residualize(y, s, degree)
    rp = _residualize(p, s, degree)
    if ry.std() < 1e-12 or rp.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(ry, rp)[0, 1])


def binned_metrics(y, p, s, n_bins: int = 10, frac: float = 0.01,
                   min_bin: int = 500):
    """Pearson / Spearman / EF inside equal-count size bins, size-weighted."""
    order = np.argsort(s, kind="stable")
    bins = np.array_split(order, n_bins)
    rows, weights = [], []
    for b in bins:
        if b.size < min_bin:
            continue
        yb, pb = y[b], p[b]
        if yb.std() < 1e-12 or pb.std() < 1e-12:
            continue
        rows.append({
            "n": int(b.size),
            "size_lo": float(s[b].min()),
            "size_hi": float(s[b].max()),
            "pearson": float(np.corrcoef(yb, pb)[0, 1]),
            "spearman": float(stats.spearmanr(yb, pb)[0]),
            "ef": topk_recall(yb, pb, frac) / frac,
        })
        weights.append(b.size)
    if not rows:
        return {}, []
    w = np.array(weights, dtype=np.float64)
    w /= w.sum()
    agg = {
        f"bin_{k}": float(np.sum(w * np.array([r[k] for r in rows])))
        for k in ("pearson", "spearman", "ef")
    }
    return agg, rows


def size_for_run(pool, tag: str, run_test_idx: np.ndarray) -> np.ndarray:
    """Heavy-atom count for a run's test rows, reusing the descriptor cache."""
    full_test = load_split(pool, tag)["test"]
    path = descriptor_path(pool, tag, "test")
    if path.exists():
        desc = np.load(path, mmap_mode="r")
        if desc.shape[0] == full_test.size:
            pos = {int(v): i for i, v in enumerate(full_test)}
            try:
                rows = np.array([pos[int(v)] for v in run_test_idx])
            except KeyError:
                rows = None
            if rows is not None:
                return np.asarray(desc[rows, HEAVY_ATOM_COL], dtype=np.float64)
    # no usable cache: compute for exactly these rows
    desc = ensure_descriptors(pool, run_test_idx, tag + "__adhoc", "test")
    return np.asarray(desc[:, HEAVY_ATOM_COL], dtype=np.float64)


def analyse_run(run_dir: Path, pool, scores, n_bins: int, frac: float,
                degree: int):
    meta = json.loads((run_dir / "metrics.json").read_text())
    pred = np.load(run_dir / "test_pred.npy").astype(np.float64)
    idx = np.load(run_dir / "test_idx.npy")
    y = np.asarray(scores[idx], dtype=np.float64)
    s = size_for_run(pool, meta["split_tag"], idx)

    agg, rows = binned_metrics(y, pred, s, n_bins=n_bins, frac=frac)
    out = {
        "backbone": meta["backbone"],
        "dim": meta["dim"],
        "n_train": meta["n_train"],
        "n_test": int(idx.size),
        "raw_pearson": float(np.corrcoef(y, pred)[0, 1]),
        "partial_pearson": partial_corr(y, pred, s),
        "residual_pearson": residual_corr(y, pred, s, degree=degree),
        **agg,
        "bins": rows,
    }
    return out


def size_reference(pool, scores, tag: str, n_bins: int, frac: float):
    """What size alone achieves on the same test rows -- the floor."""
    test = load_split(pool, tag)["test"]
    y = np.asarray(scores[test], dtype=np.float64)
    s = size_for_run(pool, tag, test)
    pred = -s  # bigger molecule -> more negative score
    agg, _ = binned_metrics(y, pred, s, n_bins=n_bins, frac=frac)
    return {
        "backbone": "SIZE ONLY (-heavy_atoms)",
        "dim": 1, "n_train": 0, "n_test": int(test.size),
        "raw_pearson": float(np.corrcoef(y, pred)[0, 1]),
        "partial_pearson": 0.0,
        "residual_pearson": 0.0,
        **agg, "bins": [],
    }


def table(rows):
    head = (f"{'backbone':<16}{'dim':>6}{'n_train':>10}"
            f"{'raw r':>9}{'partial':>9}{'resid':>9}"
            f"{'bin r':>9}{'bin rho':>9}{'bin EF@1%':>11}")
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(
            f"{r['backbone']:<16}{r['dim']:>6}{r['n_train']:>10,}"
            f"{r['raw_pearson']:>9.4f}{r['partial_pearson']:>9.4f}"
            f"{r['residual_pearson']:>9.4f}"
            f"{r.get('bin_pearson', float('nan')):>9.4f}"
            f"{r.get('bin_spearman', float('nan')):>9.4f}"
            f"{r.get('bin_ef', float('nan')):>11.1f}"
        )
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", default="enhits", choices=sorted(POOLS))
    ap.add_argument("runs", nargs="*", help="run dirs (default: all in work)")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--degree", type=int, default=3,
                    help="polynomial degree for the residual fit")
    ap.add_argument("--budget", type=int, default=0,
                    help="only runs with this n_train (0 = all)")
    ap.add_argument("--out", default=None, help="write JSON here")
    ap.add_argument("--show-bins", action="store_true",
                    help="print the per-bin breakdown for each run")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    scores = load_scores(pool)

    run_dirs = [Path(r) for r in args.runs]
    if not run_dirs:
        run_dirs = sorted(
            p.parent for p in (pool_work(pool.name) / "runs").glob(
                "*/metrics.json")
        )
    if not run_dirs:
        print("no runs found")
        return 1

    rows = []
    for d in run_dirs:
        try:
            r = analyse_run(d, pool, scores, args.bins, args.frac, args.degree)
        except (OSError, KeyError, ValueError) as e:
            print(f"  skipping {d.name}: {e}", file=sys.stderr)
            continue
        if args.budget and r["n_train"] != args.budget:
            continue
        rows.append(r)

    if not rows:
        print("no runs matched")
        return 1

    tag = json.loads((run_dirs[0] / "metrics.json").read_text())["split_tag"]
    ref = size_reference(pool, scores, tag, args.bins, args.frac)

    rows.sort(key=lambda r: (r["n_train"], -r["raw_pearson"]))
    print(f"pool={pool.name}  bins={args.bins}  within-bin EF at "
          f"{args.frac:g}\n")
    print(table([ref] + rows))
    print("\nraw r      : Pearson against the true score")
    print("partial    : same, heavy-atom count partialled out of both")
    print(f"resid      : residual correlation after a degree-{args.degree} "
          "size fit")
    print("bin *      : computed inside equal-count size bins, size-weighted")

    if args.show_bins:
        for r in rows:
            print(f"\n{r['backbone']} (n_train={r['n_train']:,})")
            for b in r["bins"]:
                print(f"  heavy {b['size_lo']:>5.0f}-{b['size_hi']:<5.0f} "
                      f"n={b['n']:>7,}  r={b['pearson']:.4f}  "
                      f"rho={b['spearman']:.4f}  EF={b['ef']:.1f}")

    if args.out:
        Path(args.out).write_text(json.dumps([ref] + rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
