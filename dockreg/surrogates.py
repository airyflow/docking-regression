"""Alternative surrogate families on the same embeddings and splits.

The main results hold the prediction head fixed -- an MLP -- so that a
comparison between backbones is a comparison between representations. The
obvious objection is that one architecture may simply suit some
representations better: that a 1024-512 MLP flatters 768-dimensional geometry
embeddings and handicaps 12 descriptors, or the reverse.

This module answers that by refitting two families with very different
inductive biases on identical cached data:

  ridge    closed-form L2 linear regression, solved on GPU from the Gram
           matrix. Measures how much of the signal is linearly accessible.

  gbdt     LightGBM gradient-boosted trees. Axis-aligned splits, no notion of
           distance in embedding space.

If the ordering of backbones survives all three, it is a property of the
representation rather than of the head.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import Standardizer, ensure_cached, load_scores
from .descriptors import ensure_descriptors
from .metrics import all_metrics, format_metrics
from .pools import POOLS, get_pool, pool_work
from .splits import load_split, split_tag
from .train import DESCRIPTOR_BACKBONE


def fit_ridge(Xtr, ytr, alphas, Xva, yva, device, chunk=200_000):
    """Closed-form ridge from Gram matrices, sweeping alpha on validation.

    Accumulating X^T X (d x d) in chunks keeps memory flat and lets one pass
    over the data serve every alpha.
    """
    d = Xtr.shape[1]
    XtX = torch.zeros((d, d), dtype=torch.float64, device=device)
    Xty = torch.zeros(d, dtype=torch.float64, device=device)
    for i in range(0, Xtr.shape[0], chunk):
        xb = torch.from_numpy(np.ascontiguousarray(Xtr[i:i + chunk])).to(
            device, dtype=torch.float64)
        yb = torch.from_numpy(ytr[i:i + chunk]).to(device, dtype=torch.float64)
        XtX += xb.T @ xb
        Xty += xb.T @ yb
        del xb, yb
    eye = torch.eye(d, dtype=torch.float64, device=device)

    best = (None, None, None)
    for a in alphas:
        try:
            w = torch.linalg.solve(XtX + a * eye, Xty)
        except RuntimeError:
            continue
        pv = _ridge_predict(Xva, w, device)
        rmse = float(np.sqrt(np.mean((pv - yva) ** 2)))
        if best[0] is None or rmse < best[0]:
            best = (rmse, a, w)
    if best[2] is None:
        raise RuntimeError("ridge failed for every alpha")
    return best[2], best[1], best[0]


def _ridge_predict(X, w, device, chunk=200_000):
    out = np.empty(X.shape[0], dtype=np.float64)
    for i in range(0, X.shape[0], chunk):
        xb = torch.from_numpy(np.ascontiguousarray(X[i:i + chunk])).to(
            device, dtype=torch.float64)
        out[i:i + chunk] = (xb @ w).cpu().numpy()
    return out


def run_ridge(Xtr, ytr, Xva, yva, Xte, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alphas = [float(a) for a in args.alphas.split(",")]
    t0 = time.time()
    w, alpha, val_rmse = fit_ridge(Xtr, ytr, alphas, Xva, yva, device)
    info = {"alpha": alpha, "val_rmse_standardised": val_rmse,
            "fit_seconds": time.time() - t0}
    return (_ridge_predict(Xva, w, device),
            _ridge_predict(Xte, w, device), info)


def run_gbdt(Xtr, ytr, Xva, yva, Xte, args):
    import lightgbm as lgb

    t0 = time.time()
    ds = lgb.Dataset(np.ascontiguousarray(Xtr), label=ytr)
    dv = lgb.Dataset(np.ascontiguousarray(Xva), label=yva, reference=ds)
    params = {
        "objective": "regression", "metric": "rmse",
        "learning_rate": args.lr, "num_leaves": args.num_leaves,
        "min_data_in_leaf": 100, "feature_fraction": 0.8,
        "bagging_fraction": 0.8, "bagging_freq": 1,
        "verbosity": -1, "num_threads": args.threads, "seed": args.seed,
    }
    booster = lgb.train(
        params, ds, num_boost_round=args.rounds, valid_sets=[dv],
        callbacks=[lgb.early_stopping(args.patience, verbose=False),
                   lgb.log_evaluation(period=args.log_every)],
    )
    info = {"best_iteration": booster.best_iteration,
            "num_leaves": args.num_leaves, "fit_seconds": time.time() - t0}
    return (booster.predict(Xva, num_iteration=booster.best_iteration),
            booster.predict(Xte, num_iteration=booster.best_iteration), info)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--family", required=True, choices=("ridge", "gbdt"))
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--cache-budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", default="1,10,100,1000,10000",
                    help="ridge: validation sweep")
    ap.add_argument("--rounds", type=int, default=2000, help="gbdt")
    ap.add_argument("--num-leaves", type=int, default=255, help="gbdt")
    ap.add_argument("--lr", type=float, default=0.05, help="gbdt")
    ap.add_argument("--patience", type=int, default=50, help="gbdt")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--materialize-ram-gb", type=float, default=64.0)
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    tag = split_tag(0, 0.05, 0.05)
    sp = load_split(pool, tag)
    train_idx, val_idx, test_idx = sp["train"], sp["val"], sp["test"]

    is_baseline = args.backbone == DESCRIPTOR_BACKBONE
    cache_rows = (min(args.cache_budget, train_idx.size) if args.cache_budget
                  else train_idx.size)
    cache_tag = tag if not args.cache_budget else f"{tag}__c{cache_rows}"
    n_train = min(args.max_train, cache_rows) if args.max_train else cache_rows
    n_v = min(args.max_eval, val_idx.size) if args.max_eval else val_idx.size
    n_t = min(args.max_eval, test_idx.size) if args.max_eval else test_idx.size

    fetch = (
        (lambda idx, part: ensure_descriptors(pool, idx, cache_tag, part))
        if is_baseline else
        (lambda idx, part: ensure_cached(
            pool, args.backbone, idx, cache_tag, part,
            ram_budget_gb=args.materialize_ram_gb))
    )

    print(f"pool={pool.name} backbone={args.backbone} family={args.family}")
    print(f"  train={n_train:,} val={n_v:,} test={n_t:,}")

    Xtr_full = fetch(train_idx[:cache_rows], "train")
    val_idx, test_idx = val_idx[:n_v], test_idx[:n_t]
    Xva, Xte = fetch(val_idx, "val"), fetch(test_idx, "test")
    train_idx = train_idx[:n_train]
    Xtr = np.array(Xtr_full[:n_train], dtype=np.float32)
    Xva = np.array(Xva, dtype=np.float32)
    Xte = np.array(Xte, dtype=np.float32)

    scores = load_scores(pool)
    ytr = np.asarray(scores[train_idx], dtype=np.float32)
    yva = np.asarray(scores[val_idx], dtype=np.float32)
    yte = np.asarray(scores[test_idx], dtype=np.float32)

    # Same standardisation as the MLP path, so only the head differs.
    xstd = Standardizer.fit(Xtr, seed=args.seed)
    Xtr, Xva, Xte = (xstd.transform(Xtr), xstd.transform(Xva),
                     xstd.transform(Xte))
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    ytr_s = (ytr - y_mean) / y_std
    yva_s = (yva - y_mean) / y_std

    fn = run_ridge if args.family == "ridge" else run_gbdt
    val_pred_s, test_pred_s, info = fn(Xtr, ytr_s, Xva, yva_s, Xte, args)
    val_pred = val_pred_s * y_std + y_mean
    test_pred = test_pred_s * y_std + y_mean

    vm, tm = all_metrics(yva, val_pred), all_metrics(yte, test_pred)
    print(f"  fit in {info['fit_seconds']:.0f}s  {info}")
    print(f"  val : {format_metrics(vm)}")
    print(f"  test: {format_metrics(tm)}")

    run_name = f"{args.backbone}__{args.family}__{tag}__n{n_train}__s{args.seed}"
    out_dir = pool_work(pool.name) / "runs_alt" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "test_idx.npy", test_idx)
    (out_dir / "metrics.json").write_text(json.dumps({
        "pool": pool.name, "backbone": args.backbone, "family": args.family,
        "dim": int(Xtr.shape[1]), "split_tag": tag, "n_train": int(n_train),
        "n_val": int(n_v), "n_test": int(n_t), "seed": args.seed,
        "epochs": -1, "patience": -1, "info": info,
        "val_metrics": vm, "test_metrics": tm,
    }, indent=2, default=float))
    print(f"wrote {out_dir}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
