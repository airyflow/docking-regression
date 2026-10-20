"""Prediction-level fusion: a suite of methods, all fitted on validation only.

Six methods plus two reference points:

  mean        unweighted average of predictions (the standard late fusion)
  zmean       average after per-model z-scoring, with mean and scale taken
              from *validation* predictions, never test
  rank        average of within-set ranks; discards calibration, keeps order
  nnls        non-negative least-squares weights fitted on validation
  ridge       ridge stacking over validation predictions, with intercept
  best        the single best member chosen on validation (not on test)

  oracle      the retrospectively best subset under `mean`, scored on test.
              An upper bound, not a deployable method, and labelled as such.

Every transform, weight and subset choice is estimated from validation
predictions. The test split is read once, to score.

It also runs the **prospective** version of the error-variance criterion. For
an equal-weight average of two models to beat the better one,

    sigma_2 / sigma_1  <  -rho + sqrt(rho^2 + 3)

Computing sigma and rho on test and checking the same test outcome is
circular. Here they are computed on validation and used to *predict* which
pairs will help on test, which is the form a practitioner could use.
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

# Lower score = better binder, so smaller predictions rank first.


def load_runs(dirs):
    runs = []
    for d in dirs:
        d = Path(d)
        meta = json.loads((d / "metrics.json").read_text())
        runs.append({
            "name": meta.get("backbone", d.name),
            "pool": meta["pool"], "split_tag": meta["split_tag"],
            "n_train": meta["n_train"],
            "val_pred": np.load(d / "val_pred.npy").astype(np.float64),
            "test_pred": np.load(d / "test_pred.npy").astype(np.float64),
            "val_idx": np.load(d / "val_idx.npy"),
            "test_idx": np.load(d / "test_idx.npy"),
        })
    return runs


def check_aligned(runs):
    for key in ("pool", "split_tag", "n_train"):
        vals = {r[key] for r in runs}
        if len(vals) != 1:
            raise ValueError(f"runs differ in {key}: {sorted(vals)}")
    for key in ("val_idx", "test_idx"):
        ref = runs[0][key]
        for r in runs[1:]:
            if r[key].shape != ref.shape or not np.array_equal(r[key], ref):
                raise ValueError(f"{r['name']} has different {key}")
    return runs[0]["pool"], runs[0]["val_idx"], runs[0]["test_idx"]


def _rank(a):
    return np.argsort(np.argsort(a)).astype(np.float64)


def fuse(method, V, T, yv):
    """Fit on validation (V, yv), apply to test (T). Rows are models."""
    if method == "mean":
        return T.mean(axis=0)
    if method == "zmean":
        mu = V.mean(axis=1, keepdims=True)
        sd = V.std(axis=1, keepdims=True)
        sd[sd < 1e-12] = 1.0
        return ((T - mu) / sd).mean(axis=0)
    if method == "rank":
        return np.stack([_rank(t) for t in T]).mean(axis=0)
    if method == "nnls":
        from scipy.optimize import nnls
        w, _ = nnls(V.T, yv)
        if w.sum() <= 1e-12:
            return T.mean(axis=0)
        return w @ T
    if method == "ridge":
        A = np.column_stack([V.T, np.ones(V.shape[1])])
        lam = 1e-3 * A.shape[0]
        coef = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yv)
        return np.column_stack([T.T, np.ones(T.shape[1])]) @ coef
    raise ValueError(f"unknown method {method!r}")


def pair_rule(v1, v2, yv):
    """Predict, from validation only, whether averaging model 2 into model 1
    improves on model 1. Returns (predicted_helps, sigma_ratio, threshold)."""
    e1, e2 = v1 - yv, v2 - yv
    s1, s2 = e1.std(), e2.std()
    if s1 > s2:                      # order so model 1 is the better one
        e1, e2, s1, s2 = e2, e1, s2, s1
    rho = float(np.corrcoef(e1, e2)[0, 1])
    thr = -rho + np.sqrt(rho ** 2 + 3)
    ratio = float(s2 / s1) if s1 > 0 else np.inf
    return ratio < thr, ratio, float(thr)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--methods",
                    default="mean,zmean,rank,nnls,ridge",
                    help="comma-separated subset of the fusion methods")
    ap.add_argument("--max-subset", type=int, default=0,
                    help="largest subset size to evaluate (0 = all members)")
    ap.add_argument("--rule-check", action="store_true",
                    help="run the prospective pair-rule evaluation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    runs = load_runs(args.runs)
    pool_name, val_idx, test_idx = check_aligned(runs)
    pool = get_pool(pool_name)
    scores = load_scores(pool)
    yv = np.asarray(scores[val_idx], dtype=np.float64)
    yt = np.asarray(scores[test_idx], dtype=np.float64)
    names = [r["name"] for r in runs]

    print(f"pool={pool_name}  n_train={runs[0]['n_train']:,}  "
          f"val={val_idx.size:,}  test={test_idx.size:,}")
    print(f"members: {', '.join(names)}\n")

    single = {}
    for r in runs:
        single[r["name"]] = {"val": all_metrics(yv, r["val_pred"]),
                             "test": all_metrics(yt, r["test_pred"])}
        print(f"  {r['name']:<14}{format_metrics(single[r['name']]['test'])}")

    # reference: best single member chosen on VALIDATION
    best_name = min(names, key=lambda n: single[n]["val"]["rmse"])
    best_test = single[best_name]["test"]
    print(f"\nbest single by validation RMSE: {best_name} "
          f"(test rmse {best_test['rmse']:.4f}, pearson "
          f"{best_test['pearson']:.4f})")

    V = np.stack([r["val_pred"] for r in runs])
    T = np.stack([r["test_pred"] for r in runs])
    k_max = args.max_subset or len(runs)

    results = {"pool": pool_name, "n_train": runs[0]["n_train"],
               "members": names, "single": single,
               "best_single_by_val": best_name, "fused": {}}

    methods = [m for m in args.methods.split(",") if m]
    print(f"\n{'method':<8}{'subset':<42}{'test rmse':>11}{'pearson':>10}"
          f"{'EF@1%':>9}{'vs best':>9}")
    print("-" * 89)
    for method in methods:
        rows = []
        for k in range(2, k_max + 1):
            for combo in itertools.combinations(range(len(runs)), k):
                pred = fuse(method, V[list(combo)], T[list(combo)], yv)
                m = all_metrics(yt, pred)
                key = "+".join(names[i] for i in combo)
                results["fused"].setdefault(method, {})[key] = m
                rows.append((key, m))
        # rank: for rank-fusion RMSE is meaningless, sort on spearman
        keyf = (lambda kv: -kv[1]["spearman"]) if method == "rank" \
            else (lambda kv: kv[1]["rmse"])
        rows.sort(key=keyf)
        for key, m in rows[:3]:
            delta = best_test["pearson"] - m["pearson"]
            flag = f"{-delta:+.4f}" if method != "rank" else "    —"
            print(f"{method:<8}{key[:40]:<42}{m['rmse']:>11.4f}"
                  f"{m['pearson']:>10.4f}{m['ef@0.01']:>9.1f}{flag:>9}")

    # oracle upper bound: best subset under mean, chosen on TEST
    oracle = None
    for k in range(2, k_max + 1):
        for combo in itertools.combinations(range(len(runs)), k):
            m = all_metrics(yt, T[list(combo)].mean(axis=0))
            if oracle is None or m["rmse"] < oracle[1]["rmse"]:
                oracle = ("+".join(names[i] for i in combo), m)
    results["oracle_mean_subset"] = {"subset": oracle[0], "metrics": oracle[1]}
    print(f"\noracle (best mean-subset chosen on TEST — upper bound, not a "
          f"method):\n  {oracle[0]}  rmse={oracle[1]['rmse']:.4f} "
          f"pearson={oracle[1]['pearson']:.4f}")

    if args.rule_check:
        print("\nprospective pair rule (sigma and rho from VALIDATION, "
              "outcome on TEST):")
        print(f"  {'pair':<30}{'ratio':>8}{'thresh':>8}{'predict':>9}"
              f"{'actual':>9}{'  ok'}")
        tab, correct = [], 0
        for i, j in itertools.combinations(range(len(runs)), 2):
            helps_pred, ratio, thr = pair_rule(V[i], V[j], yv)
            # actual: does mean-fusion beat the better member ON TEST?
            bi = i if single[names[i]]["val"]["rmse"] <= \
                single[names[j]]["val"]["rmse"] else j
            fused = all_metrics(yt, T[[i, j]].mean(axis=0))
            helps_act = fused["rmse"] < single[names[bi]]["test"]["rmse"]
            ok = helps_pred == helps_act
            correct += ok
            tab.append({"pair": f"{names[i]}+{names[j]}", "ratio": ratio,
                        "threshold": thr, "predicted": bool(helps_pred),
                        "actual": bool(helps_act), "correct": bool(ok)})
            print(f"  {names[i] + '+' + names[j]:<30}{ratio:>8.2f}{thr:>8.2f}"
                  f"{str(helps_pred):>9}{str(helps_act):>9}"
                  f"{'  ✓' if ok else '  ✗'}")
        n = len(tab)
        print(f"\n  rule accuracy: {correct}/{n} ({correct / n * 100:.0f}%)")
        results["pair_rule"] = {"table": tab, "correct": correct, "n": n}

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
