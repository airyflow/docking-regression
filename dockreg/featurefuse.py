"""Feature-level fusion: combine modalities before the prediction head.

Named "feature-level", not "intermediate", deliberately. The multimodality
literature's intermediate fusion trains modality-specific projection layers
jointly with the head; this concatenates frozen embeddings and trains one head
on the result. They are different architectures and a reader should not have to
guess which is meant.

Two variants, because naive concatenation has three confounds:

  concat   Per-modality standardisation (fitted on training rows only), then
           concatenation at native width. Measures what a practitioner would
           actually get, but a 3400-d backbone contributes 283x more features
           than the 12-d descriptor baseline before any learning happens, and
           the head sees many more parameters than any unimodal arm.

  pca      Each modality is projected to a common width (default 128) by PCA
           fitted on training rows, then concatenated. Capacity is far closer
           to matched, so this is the fairer test of complementarity.

           Caveat worth stating in any write-up: the descriptor baseline has
           only 12 features and cannot be projected *up*. It is carried at its
           native width, so capacity is approximately, not exactly, matched.

Both variants fit every transform on training rows only.
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
from .model import build_model, count_params
from .pools import POOLS, get_pool, pool_work
from .splits import load_split, split_tag
from .train import DESCRIPTOR_BACKBONE


def fit_pca(X, k, seed=0, max_rows=200_000):
    """PCA basis from a training subsample. Returns (mean, components)."""
    from sklearn.decomposition import PCA

    n = X.shape[0]
    if n > max_rows:
        rs = np.random.default_rng(seed)
        sel = np.sort(rs.choice(n, size=max_rows, replace=False))
        sample = np.asarray(X[sel], dtype=np.float32)
    else:
        sample = np.asarray(X, dtype=np.float32)
    k = min(k, sample.shape[1], sample.shape[0])
    p = PCA(n_components=k, svd_solver="randomized", random_state=seed)
    p.fit(sample)
    return p.mean_.astype(np.float32), p.components_.astype(np.float32)


def apply_pca(X, mean, comps, chunk=200_000):
    out = np.empty((X.shape[0], comps.shape[0]), dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        out[i:i + chunk] = (np.asarray(X[i:i + chunk], dtype=np.float32)
                            - mean) @ comps.T
    return out


def build_blocks(pool, backbones, idx_sets, cache_tag, args):
    """Return {part: matrix} with every modality standardised and concatenated."""
    parts = {p: [] for p in idx_sets}
    widths = {}
    for bb in backbones:
        fetch = (ensure_descriptors if bb == DESCRIPTOR_BACKBONE else None)
        blocks = {}
        for part, idx in idx_sets.items():
            if bb == DESCRIPTOR_BACKBONE:
                blocks[part] = np.array(
                    ensure_descriptors(pool, idx, cache_tag, part),
                    dtype=np.float32)
            else:
                blocks[part] = np.array(
                    ensure_cached(pool, bb, idx, cache_tag, part,
                                  ram_budget_gb=args.materialize_ram_gb),
                    dtype=np.float32)

        # standardise on training rows only
        std = Standardizer.fit(blocks["train"], seed=args.seed)
        for part in blocks:
            blocks[part] = std.transform(blocks[part])

        if args.mode == "pca" and blocks["train"].shape[1] > args.pca_dim:
            mean, comps = fit_pca(blocks["train"], args.pca_dim, seed=args.seed)
            for part in blocks:
                blocks[part] = apply_pca(blocks[part], mean, comps)

        widths[bb] = blocks["train"].shape[1]
        for part in blocks:
            parts[part].append(blocks[part])

    return {p: np.concatenate(v, axis=1) for p, v in parts.items()}, widths


def train_head(X, y, args, device):
    Xtr, Xva, Xte = X["train"], X["val"], X["test"]
    ytr, yva, yte = y["train"], y["val"], y["test"]
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = build_model(Xtr.shape[1], hidden=(1024, 512), dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = min(args.batch_size, max(16, Xtr.shape[0] // 4))
    steps = max(1, Xtr.shape[0] // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(args.seed)

    def predict(Xa):
        model.eval()
        out = np.empty(Xa.shape[0], dtype=np.float32)
        with torch.no_grad():
            for i in range(0, Xa.shape[0], 8192):
                xb = torch.from_numpy(Xa[i:i + 8192]).to(device)
                out[i:i + 8192] = model(xb).float().cpu().numpy()
        return out * y_std + y_mean

    best, best_state, stale = None, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(Xtr.shape[0])
        for s in range(steps):
            sel = np.sort(order[s * bs:(s + 1) * bs])
            if sel.size == 0:
                continue
            xb = torch.from_numpy(Xtr[sel]).to(device)
            yb = torch.from_numpy((ytr[sel] - y_mean) / y_std).to(device)
            opt.zero_grad(set_to_none=True)
            loss_fn(model(xb), yb).backward()
            opt.step()
            sched.step()
        rmse = float(np.sqrt(np.mean((predict(Xva) - yva) ** 2)))
        if best is None or rmse < best:
            best, stale = rmse, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return predict(Xva), predict(Xte), count_params(model)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--backbones", required=True,
                    help="comma-separated; 'descriptors' allowed")
    ap.add_argument("--mode", default="concat", choices=("concat", "pca"))
    ap.add_argument("--pca-dim", type=int, default=128)
    ap.add_argument("--max-train", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--cache-budget", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--materialize-ram-gb", type=float, default=64.0)
    ap.add_argument("--tag", default=None, help="short name for the run dir")
    args = ap.parse_args(argv)

    pool = get_pool(args.pool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = split_tag(0, 0.05, 0.05)
    sp = load_split(pool, tag)
    tr, va, te = sp["train"], sp["val"], sp["test"]

    cache_rows = (min(args.cache_budget, tr.size) if args.cache_budget
                  else tr.size)
    cache_tag = tag if not args.cache_budget else f"{tag}__c{cache_rows}"
    n_tr = min(args.max_train, cache_rows) if args.max_train else cache_rows
    n_va = min(args.max_eval, va.size) if args.max_eval else va.size
    n_te = min(args.max_eval, te.size) if args.max_eval else te.size

    backbones = [b for b in args.backbones.split(",") if b]
    # Cache the SAME rows dockreg.train would, then slice. Caching only
    # `n_tr` rows under the shared tag silently truncates the full-split
    # cache for every other experiment -- which is exactly what happened on
    # 2026-10-18, leaving MolFormer and SMI-TED at 200k rows and breaking 13
    # downstream runs.
    cache_sets = {"train": tr[:cache_rows], "val": va[:n_va], "test": te[:n_te]}
    idx_sets = {"train": tr[:n_tr], "val": va[:n_va], "test": te[:n_te]}

    print(f"pool={pool.name} mode={args.mode} backbones={'+'.join(backbones)}")
    print(f"  train={n_tr:,} val={n_va:,} test={n_te:,}  device={device}")

    t0 = time.time()
    X, widths = build_blocks(pool, backbones, cache_sets, cache_tag, args)
    # slice the cached training block down to this run's budget
    if n_tr != cache_rows:
        X["train"] = X["train"][:n_tr]
    print("  block widths: " + ", ".join(f"{k}={v}" for k, v in widths.items())
          + f"  -> fused {X['train'].shape[1]}")

    scores = load_scores(pool)
    y = {p: np.asarray(scores[i], dtype=np.float32) for p, i in idx_sets.items()}

    val_pred, test_pred, params = train_head(X, y, args, device)
    vm, tm = all_metrics(y["val"], val_pred), all_metrics(y["test"], test_pred)
    print(f"  params={params:,}  [{time.time() - t0:.0f}s]")
    print(f"  val : {format_metrics(vm)}")
    print(f"  test: {format_metrics(tm)}")

    name = args.tag or f"{'+'.join(backbones)}__{args.mode}"
    out = pool_work(pool.name) / "runs_featurefuse" / \
        f"{name}__{tag}__n{n_tr}__s{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "test_pred.npy", test_pred)
    np.save(out / "val_pred.npy", val_pred)
    np.save(out / "test_idx.npy", idx_sets["test"])
    np.save(out / "val_idx.npy", idx_sets["val"])
    (out / "metrics.json").write_text(json.dumps({
        "pool": pool.name, "backbone": name, "fusion": f"feature-{args.mode}",
        "backbones": backbones, "block_widths": widths,
        "dim": int(X["train"].shape[1]), "params": params,
        "split_tag": tag, "n_train": int(n_tr), "n_val": int(n_va),
        "n_test": int(n_te), "seed": args.seed, "pca_dim": args.pca_dim,
        "epochs": args.epochs, "patience": args.patience,
        "val_metrics": vm, "test_metrics": tm,
    }, indent=2, default=float))
    print(f"wrote {out}/metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
