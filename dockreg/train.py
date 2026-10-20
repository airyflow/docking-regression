"""Train one surrogate: (pool, backbone) -> docking score.

Model selection uses the validation split only. The test split is touched once,
after the best epoch has been chosen, and its predictions are written to disk so
that fuse.py can combine backbones without retraining anything.

Embeddings are cached once per (backbone, split) at full size. Because split
indices are stored in random order, a training budget of N is exactly the first
N rows of that cache -- so a learning-curve sweep re-reads a prefix instead of
re-gathering from the pool for every budget.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .data import Standardizer, ensure_cached, load_scores
from .descriptors import ensure_descriptors, feature_names
from .metrics import all_metrics, format_metrics
from .model import build_model, count_params
from .pools import POOLS, get_pool, pool_work
from .splits import load_split, split_tag

# Validation metrics where larger is better; the rest are minimised.
_MAXIMISE = {"r2", "pearson", "spearman"} | {"recall", "ef"}

# Pseudo-backbone: cheap descriptors instead of a learned embedding. This is
# the floor every real backbone has to clear -- see descriptors.py.
DESCRIPTOR_BACKBONE = "descriptors"


def _is_maximised(metric: str) -> bool:
    return metric in _MAXIMISE or metric.split("@")[0] in _MAXIMISE


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _maybe_load(arr, in_memory: bool, label: str):
    if not in_memory:
        return arr
    gb = arr.nbytes / 1e9
    t0 = time.time()
    # np.asarray() on a memmap returns a *view* sharing the mapped buffer, so
    # every epoch would keep page-faulting against Lustre. np.array() copies.
    out = np.array(arr, dtype=np.float32)
    print(f"  loaded {label} into RAM ({gb:.2f} GB, {time.time() - t0:.0f}s)",
          flush=True)
    return out


@torch.no_grad()
def predict(model, X, std: Standardizer, device, batch_size: int = 8192):
    model.eval()
    out = np.empty(X.shape[0], dtype=np.float32)
    for i in range(0, X.shape[0], batch_size):
        xb = std.transform(X[i:i + batch_size])
        xb = torch.from_numpy(xb).to(device, non_blocking=True)
        out[i:i + batch_size] = model(xb).float().cpu().numpy()
    return out


def run(args):
    pool = get_pool(args.pool)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu
                          else "cpu")
    set_seed(args.seed)

    tag = split_tag(args.split_seed, args.val_frac, args.test_frac)
    sp = load_split(pool, tag)
    train_idx, val_idx, test_idx = sp["train"], sp["val"], sp["test"]

    is_baseline = args.backbone == DESCRIPTOR_BACKBONE
    dim = len(feature_names()) if is_baseline else pool.dim(args.backbone)

    # Cache once, then read budgets as prefixes of that cache. On pools of
    # 10^8 rows the full split is far too large to materialise (GROVER over
    # all of d4 would be 1.5 TB), so --cache-budget caps what is cached and
    # every budget must fit inside it.
    cache_rows = (min(args.cache_budget, train_idx.size) if args.cache_budget
                  else train_idx.size)
    cache_tag = tag if not args.cache_budget else f"{tag}__c{cache_rows}"

    fetch = (
        (lambda idx, part: ensure_descriptors(
            pool, idx, cache_tag, part, rebuild=args.rebuild_cache))
        if is_baseline else
        (lambda idx, part: ensure_cached(
            pool, args.backbone, idx, cache_tag, part,
            rebuild=args.rebuild_cache, ram_budget_gb=args.materialize_ram_gb))
    )

    n_train = min(args.max_train, cache_rows) if args.max_train else cache_rows
    if args.max_train and args.max_train > cache_rows:
        raise ValueError(
            f"--max-train {args.max_train:,} exceeds --cache-budget "
            f"{cache_rows:,}; raise the cache budget or lower the budget"
        )
    n_eval_v = min(args.max_eval, val_idx.size) if args.max_eval else val_idx.size
    n_eval_t = min(args.max_eval, test_idx.size) if args.max_eval else test_idx.size

    budget = f"n{n_train}"
    sub = f"__u{args.subset_seed}" if args.subset_seed else ""
    run_name = (args.run_name
                or f"{args.backbone}__{tag}__{budget}__s{args.seed}{sub}")
    out_dir = Path(args.out_dir or (pool_work(pool.name) / "runs")) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"pool={pool.name} backbone={args.backbone} dim={dim}")
    print(f"  train={n_train:,} val={n_eval_v:,} test={n_eval_t:,}")
    print(f"  device={device}  out={out_dir}")

    # Cache `cache_rows` training rows once; this budget is a prefix of it.
    # --max-eval caps what is cached for val/test too: on d4 the full
    # validation split is 5.7M rows, which for GROVER would be 77 GB per set.
    Xtr_full = fetch(train_idx[:cache_rows], "train")

    val_idx = val_idx[:n_eval_v]
    test_idx = test_idx[:n_eval_t]
    Xva = fetch(val_idx, "val")
    Xte = fetch(test_idx, "test")

    # --subset-seed redraws WHICH training rows are used, holding val/test and
    # the cache fixed. Initialisation seeds (--seed) do not measure this, and
    # at 250 rows the subset draw is the dominant source of variability.
    if args.subset_seed:
        rs = np.random.default_rng(args.subset_seed)
        sel = np.sort(rs.permutation(cache_rows)[:n_train])
    else:
        sel = np.arange(n_train)
    train_idx = train_idx[:cache_rows][sel]
    Xtr = Xtr_full[sel]

    in_mem = args.in_memory and (Xtr.nbytes / 1e9) <= args.in_memory_gb
    Xtr = _maybe_load(Xtr, in_mem, "train")
    Xva = _maybe_load(Xva, True, "val")
    Xte = _maybe_load(Xte, True, "test")

    scores = load_scores(pool)
    ytr = np.asarray(scores[train_idx], dtype=np.float32)
    yva = np.asarray(scores[val_idx], dtype=np.float32)
    yte = np.asarray(scores[test_idx], dtype=np.float32)

    xstd = Standardizer.fit(Xtr, seed=args.seed)
    y_mean, y_std = float(ytr.mean()), float(ytr.std())
    if y_std < 1e-6:
        raise ValueError("training scores have ~zero variance")
    print(f"  y: mean={y_mean:.3f} std={y_std:.3f}")

    hidden = tuple(int(h) for h in args.hidden.split(",") if h)
    model = build_model(Xtr.shape[1], hidden=hidden, dropout=args.dropout)
    model.to(device)
    print(f"  model {hidden} dropout={args.dropout} "
          f"params={count_params(model):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    # Shrink the batch for tiny budgets so a 250-row run still takes several
    # gradient steps per epoch. Identical to scripts/dockstring_probe.py, so
    # low-budget sweeps on the two datasets are comparable. No effect above
    # n_train = 4096, where this returns args.batch_size unchanged.
    bs = min(args.batch_size, max(16, n_train // 4))
    steps_per_epoch = max(1, n_train // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * steps_per_epoch
    )
    loss_fn = torch.nn.MSELoss()

    best_val, best_epoch, best_state = None, -1, None
    maximise = _is_maximised(args.select_on)
    history = []
    rng = np.random.default_rng(args.seed)
    stale = 0

    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(n_train)
        running, nb = 0.0, 0
        t0 = time.time()
        for s in range(steps_per_epoch):
            sel = np.sort(order[s * bs:(s + 1) * bs])
            if sel.size == 0:
                continue
            xb = xstd.transform(Xtr[sel])
            yb = (ytr[sel] - y_mean) / y_std
            xb = torch.from_numpy(xb).to(device, non_blocking=True)
            yb = torch.from_numpy(yb).to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()
            nb += 1

        val_pred = predict(model, Xva, xstd, device, args.eval_batch_size)
        val_pred = val_pred * y_std + y_mean
        vm = all_metrics(yva, val_pred)
        history.append({"epoch": epoch, "train_loss": running / max(nb, 1), **vm})

        score = vm[args.select_on]
        better = (best_val is None
                  or (score > best_val if maximise else score < best_val))
        if better:
            best_val, best_epoch = score, epoch
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        flag = " *" if better else ""
        print(f"  epoch {epoch:3d}  loss={running / max(nb, 1):.4f}  "
              f"{format_metrics(vm)}  [{time.time() - t0:.0f}s]{flag}",
              flush=True)
        if args.patience and stale >= args.patience:
            print(f"  early stop: no val {args.select_on} gain in "
                  f"{args.patience} epochs", flush=True)
            break

    if best_state is None:
        raise RuntimeError("no epoch completed")
    model.load_state_dict(best_state)
    print(f"\nbest epoch {best_epoch} by val {args.select_on}={best_val:.4f}")

    val_pred = predict(model, Xva, xstd, device, args.eval_batch_size)
    val_pred = val_pred * y_std + y_mean
    test_pred = predict(model, Xte, xstd, device, args.eval_batch_size)
    test_pred = test_pred * y_std + y_mean

    vm = all_metrics(yva, val_pred)
    tm = all_metrics(yte, test_pred)
    print(f"  val : {format_metrics(vm)}")
    print(f"  test: {format_metrics(tm)}")

    np.save(out_dir / "test_pred.npy", test_pred)
    np.save(out_dir / "val_pred.npy", val_pred)
    np.save(out_dir / "test_idx.npy", test_idx)
    np.save(out_dir / "val_idx.npy", val_idx)
    torch.save(
        {"state_dict": best_state, "hidden": hidden, "dropout": args.dropout,
         "in_dim": int(Xtr.shape[1]), "x_mean": xstd.mean, "x_std": xstd.std,
         "y_mean": y_mean, "y_std": y_std},
        out_dir / "model.pt",
    )
    record = {
        "pool": pool.name, "target": pool.target, "oracle": pool.oracle,
        "backbone": args.backbone, "dim": int(Xtr.shape[1]),
        "split_tag": tag, "n_train": int(n_train),
        "n_val": int(n_eval_v), "n_test": int(n_eval_t),
        "seed": args.seed, "subset_seed": args.subset_seed,
        "select_on": args.select_on,
        "best_epoch": best_epoch, "best_val": best_val,
        "epochs_run": len(history), "params": count_params(model),
        "hidden": list(hidden), "dropout": args.dropout, "lr": args.lr,
        "batch_size": int(bs), "epochs": args.epochs,
        "patience": args.patience, "cache_budget": int(cache_rows),
        "val_metrics": vm, "test_metrics": tm, "history": history,
    }
    (out_dir / "metrics.json").write_text(json.dumps(record, indent=2))
    print(f"wrote {out_dir}/metrics.json")
    return record


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pool", required=True, choices=sorted(POOLS))
    ap.add_argument("--backbone", required=True,
                    help=f"a pool backbone, or {DESCRIPTOR_BACKBONE!r} for the "
                         "physicochemical baseline")
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0, help="model/init seed")
    ap.add_argument("--subset-seed", type=int, default=0,
                    help="redraw which training rows are used (0 = the first "
                         "n_train of the stored order). Holds val/test and the "
                         "embedding cache fixed, so this measures "
                         "training-subset variance, which --seed does not.")
    ap.add_argument("--max-train", type=int, default=0,
                    help="cap training rows (0 = use all cached)")
    ap.add_argument("--cache-budget", type=int, default=0,
                    help="cap the rows materialised into the embedding cache "
                         "(0 = whole training split). Required on 10^8-row "
                         "pools, where caching the full split is infeasible.")
    ap.add_argument("--materialize-ram-gb", type=float, default=64.0,
                    help="build caches in RAM up to this size; above it, fall "
                         "back to the slower memmapped build")
    ap.add_argument("--max-eval", type=int, default=0,
                    help="cap val/test rows (0 = use all)")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6,
                    help="early stop after this many epochs without a val "
                         "gain (0 = never)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--eval-batch-size", type=int, default=8192)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--hidden", default="1024,512")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--select-on", default="rmse",
                    help="validation metric for model selection "
                         "(rmse, spearman, 'recall@0.01', ...)")
    ap.add_argument("--in-memory", action="store_true", default=True)
    ap.add_argument("--no-in-memory", dest="in_memory", action="store_false")
    ap.add_argument("--in-memory-gb", type=float, default=48.0)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
