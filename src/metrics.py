"""Forecast-verification metrics used in solar flare prediction.

Accuracy is meaningless here (predicting 'no flare' always is ~95% accurate), so we report
TSS (True Skill Statistic) and HSS (Heidke Skill Score), which are the community standard,
plus recall / precision / F1 and threshold-free scores.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, make_scorer, roc_auc_score, roc_curve


def confusion(y, pred):
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(int)
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tn, fp, fn, tp


def _div(a, b):
    return float(a / b) if b else 0.0


def tss(y, pred) -> float:
    """True Skill Statistic = recall (TPR) - false alarm rate (FPR). Range -1..1, 0 = no skill.
    Unlike precision/F1 it does not depend on the class ratio, which is why flare studies use it."""
    tn, fp, fn, tp = confusion(y, pred)
    return _div(tp, tp + fn) - _div(fp, fp + tn)


def hss2(y, pred) -> float:
    """Heidke Skill Score (HSS2 variant). Skill relative to random chance; 0 = no skill, 1 = perfect."""
    tn, fp, fn, tp = confusion(y, pred)
    denom = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    return _div(2 * (tp * tn - fn * fp), denom)


def evaluate(y, proba, threshold: float) -> dict:
    y = np.asarray(y).astype(int)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion(y, pred)
    recall = _div(tp, tp + fn)
    precision = _div(tp, tp + fp)
    out = {
        "threshold": float(threshold),
        "TSS": tss(y, pred),
        "HSS": hss2(y, pred),
        "recall": recall,
        "precision": precision,
        "F1": _div(2 * precision * recall, precision + recall),
        "FPR": _div(fp, fp + tn),
        "TN": tn, "FP": fp, "FN": fn, "TP": tp,
        "n": int(len(y)), "n_pos": int(y.sum()),
    }
    if 0 < y.sum() < len(y):
        out["ROC_AUC"] = float(roc_auc_score(y, proba))
        out["PR_AUC"] = float(average_precision_score(y, proba))
        out["Brier"] = float(brier_score_loss(y, proba))
        clim = y.mean()
        out["BSS"] = float(1 - out["Brier"] / (clim * (1 - clim))) if 0 < clim < 1 else float("nan")
    return out


def best_threshold(y, proba) -> float:
    """Probability threshold that maximises TSS (= Youden's J on the ROC curve)."""
    fpr, tpr, thr = roc_curve(y, proba)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -1
    return float(thr[int(np.argmax(j))])


tss_scorer = make_scorer(tss)


def bootstrap_ci(y, proba, threshold: float, n_boot: int = 1000, block: int = 24, alpha: float = 0.05,
                 seed: int = 42) -> dict:
    """
Calculate bootstrap confidence intervals for evaluation metrics.

IMPORTANT: These intervals use block bootstrap resampling with
24-row blocks. However, the original file order is not chronological,
and observation windows overlap heavily. Therefore, block resampling
cannot reliably capture temporal dependence. Treat these intervals
as optimistic uncertainty estimates, not definitive confidence intervals.
"""
    
    y = np.asarray(y).astype(int)
    pred = (np.asarray(proba) >= threshold).astype(int)
    n = len(y)
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    out = {"TSS": [], "HSS": [], "recall": []}
    for _ in range(n_boot):
        starts = rng.integers(0, n - block + 1, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n]
        yy, pp = y[idx], pred[idx]
        tn, fp, fn, tp = confusion(yy, pp)
        out["TSS"].append(tss(yy, pp))
        out["HSS"].append(hss2(yy, pp))
        out["recall"].append(_div(tp, tp + fn))
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {k: [float(np.percentile(v, lo)), float(np.percentile(v, hi))] for k, v in out.items()}
