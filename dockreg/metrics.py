"""Metrics for docking-score regression.

Convention: lower score = better binder (true for DOCK3.7 and Glide HTVS), so
"top-k" always means the k most negative values.

Regression quality and screening quality are different questions. Pearson /
RMSE describe the whole distribution, which is dominated by the millions of
non-binders. Top-k recall and enrichment describe the only part you would ever
synthesise. Report both; they routinely disagree.
"""

import numpy as np
from scipy import stats


def _as1d(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64).ravel()


def topk_recall(y_true, y_pred, frac: float) -> float:
    """Fraction of the true top-`frac` that lands in the predicted top-`frac`."""
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    n = y_true.size
    k = max(1, int(round(n * frac)))
    true_top = np.argpartition(y_true, k - 1)[:k]
    pred_top = np.argpartition(y_pred, k - 1)[:k]
    return np.intersect1d(true_top, pred_top, assume_unique=False).size / k


def enrichment_factor(y_true, y_pred, frac: float) -> float:
    """Recall at `frac` over the recall a random ranker would get (= frac)."""
    return topk_recall(y_true, y_pred, frac) / frac


def regression_metrics(y_true, y_pred) -> dict:
    y_true, y_pred = _as1d(y_true), _as1d(y_pred)
    err = y_pred - y_true
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
        "pearson": float(stats.pearsonr(y_true, y_pred)[0]),
        "spearman": float(stats.spearmanr(y_true, y_pred)[0]),
    }


DEFAULT_FRACS = (0.001, 0.01, 0.05)


def screening_metrics(y_true, y_pred, fracs=DEFAULT_FRACS) -> dict:
    out = {}
    for f in fracs:
        tag = f"{f:g}".replace("0.", "")
        out[f"recall@{f:g}"] = topk_recall(y_true, y_pred, f)
        out[f"ef@{f:g}"] = enrichment_factor(y_true, y_pred, f)
    return out


def all_metrics(y_true, y_pred, fracs=DEFAULT_FRACS) -> dict:
    m = regression_metrics(y_true, y_pred)
    m.update(screening_metrics(y_true, y_pred, fracs))
    return m


def format_metrics(m: dict) -> str:
    order = ["rmse", "mae", "r2", "pearson", "spearman"]
    head = "  ".join(f"{k}={m[k]:.4f}" for k in order if k in m)
    rest = "  ".join(
        f"{k}={v:.4f}" for k, v in m.items() if k not in order
    )
    return f"{head}   {rest}" if rest else head
