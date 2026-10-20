"""Run the docking-regression protocol on a DOCKSTRING target.

Purpose: resolve a contradiction between two projects in this workspace.

  al-eval-framework (DOCKSTRING, AutoDock Vina, 50k molecules) ranks MolFormer
  near-best and Uni-Mol poor.

  docking-regression (EnamineHTS, DOCK3.7, 1.9M molecules) ranks Uni-Mol2 best
  and MolFormer worst.

Those orderings are opposite. The candidate explanations are the oracle (Vina
vs DOCK3.7), the backbone version (Uni-Mol v1 512d vs Uni-Mol2 768d), the data
scale, the surrogate protocol, or a bug in one pipeline.

This script holds the protocol fixed -- same MLP, split scheme, standardisation
and metrics as dockreg.train -- and swaps in the DOCKSTRING data, so protocol
is eliminated as a variable and the remaining candidates can be separated.

Embeddings are read from al-eval-framework's extraction output; nothing is
recomputed, so the backbone versions are exactly the ones that produced the
published decoupling result.
"""

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dockreg.data import Standardizer  # noqa: E402
from dockreg.descriptors import compute as compute_descriptors  # noqa: E402
from dockreg.metrics import all_metrics, format_metrics  # noqa: E402
from dockreg.model import build_model, count_params  # noqa: E402

AL_EVAL = Path("/N/slate/mengjing/repos/al-eval-framework")


def load_target(target: str):
    """SMILES and Vina scores, in the CSV's own row order."""
    path = AL_EVAL / "data" / "dockstring" / f"{target}.csv.gz"
    smiles, scores = [], []
    with gzip.open(path, "rt") as f:
        header = f.readline().rstrip("\n")
        if header != "smiles,score":
            raise ValueError(f"unexpected header in {path}: {header!r}")
        for line in f:
            smi, _, val = line.rstrip("\n").rpartition(",")
            smiles.append(smi)
            scores.append(float(val))
    return smiles, np.array(scores, dtype=np.float32)


def load_embeddings(target: str, backbone: str, smiles):
    """Embeddings aligned to `smiles`, verified element-wise."""
    z = np.load(AL_EVAL / "results" / "embed" / target /
                f"{backbone}_embeddings.npz", allow_pickle=True)
    emb, emb_smiles = z["embeddings"], z["smiles"]
    if len(emb_smiles) != len(smiles):
        raise ValueError(f"{backbone}: {len(emb_smiles)} rows vs "
                         f"{len(smiles)} scores")
    mismatch = int(np.sum(emb_smiles != np.array(smiles, dtype=emb_smiles.dtype)))
    if mismatch:
        raise ValueError(f"{backbone}: {mismatch} SMILES mismatches vs scores")
    return np.asarray(emb, dtype=np.float32)


def morgan(smiles, n_bits=2048, radius=2):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles), n_bits), dtype=np.float32)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            out[i] = np.asarray(gen.GetFingerprintAsNumPy(mol), dtype=np.float32)
    return out


def train_eval(X, y, idx_tr, idx_va, idx_te, args, device):
    Xtr, Xva, Xte = X[idx_tr], X[idx_va], X[idx_te]
    ytr, yva, yte = y[idx_tr], y[idx_va], y[idx_te]

    xstd = Standardizer.fit(Xtr, seed=args.seed)
    Xtr, Xva, Xte = (xstd.transform(Xtr), xstd.transform(Xva),
                     xstd.transform(Xte))
    y_mean, y_std = float(ytr.mean()), float(ytr.std())

    torch.manual_seed(args.seed)
    model = build_model(Xtr.shape[1], hidden=(1024, 512), dropout=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = min(args.batch_size, max(16, len(idx_tr) // 4))
    steps = max(1, len(idx_tr) // bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)
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
        order = rng.permutation(len(idx_tr))
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
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return all_metrics(yte, predict(Xte)), count_params(model)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--target", default="PARP1")
    ap.add_argument("--backbones", default="molformer,unimol,grover,morgan,descriptors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--test-frac", type=float, default=0.05)
    ap.add_argument("--train-n", type=int, default=0,
                    help="cap training rows (0 = all). al-eval-framework's "
                         "static_eval uses train_frac=0.005, i.e. 250 of "
                         "50,000 -- pass --train-n 250 to match it.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smiles, y = load_target(args.target)
    n = len(smiles)
    print(f"target={args.target}  n={n:,}  oracle=AutoDock Vina")
    print(f"  score: min={y.min():.2f} max={y.max():.2f} mean={y.mean():.2f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    n_te = int(round(n * args.test_frac))
    n_va = int(round(n * args.val_frac))
    idx_te, idx_va, idx_tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    if args.train_n:
        idx_tr = idx_tr[:args.train_n]
    print(f"  train={len(idx_tr):,} val={len(idx_va):,} test={len(idx_te):,}\n")

    results = {}
    for bb in (b for b in args.backbones.split(",") if b):
        t0 = time.time()
        if bb == "descriptors":
            X = compute_descriptors(smiles)
        elif bb == "morgan":
            X = morgan(smiles)
        else:
            X = load_embeddings(args.target, bb, smiles)
        m, params = train_eval(X, y, idx_tr, idx_va, idx_te, args, device)
        results[bb] = {"dim": int(X.shape[1]), "params": params, **m}
        print(f"{bb:<13} dim={X.shape[1]:>5}  {format_metrics(m)}  "
              f"[{time.time() - t0:.0f}s]", flush=True)

    print(f"\nranking by Pearson (this protocol, {args.target}/Vina):")
    for i, (k, v) in enumerate(
            sorted(results.items(), key=lambda kv: -kv[1]["pearson"]), 1):
        print(f"  {i}. {k:<13} r={v['pearson']:.4f}  rmse={v['rmse']:.4f}")

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"target": args.target, "oracle": "vina", "n": n,
             "seed": args.seed, "results": results}, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
