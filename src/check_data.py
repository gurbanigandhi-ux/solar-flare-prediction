"""Sanity checks to run BEFORE training on the Cleaned SWAN-SF pickles.

    python -m src.check_data --path data/raw/cleaned_swansf

Prints, per partition: array shape, NaN/inf counts, value range, label counts and flare rate, then a quick
leakage sniff test (single-feature ROC-AUC on the last value and the window mean of every parameter). Read the output and
compare with the checklist in README.md. Loads one partition at a time.
"""
from __future__ import annotations

import argparse
import pickle

import numpy as np
from sklearn.metrics import roc_auc_score

from . import config
from .data_loading import _find_cleaned_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default=str(config.DATA_RAW / "cleaned_swansf"))
    ap.add_argument("--max-rows", type=int, default=30000, help="rows used for the AUC sniff test (all positives kept)")
    args = ap.parse_args()

    files = _find_cleaned_files(args.path)
    problems, rates = [], {}
    if not files:
        raise SystemExit(f"No Partition*.pkl files found under {args.path} (train/ resampled files are ignored).")
    for part in sorted(files):
        d = files[part]
        if "X" not in d or "y" not in d:
            problems.append(f"partition {part}: missing {'labels' if 'X' in d else 'data'} file")
            continue
        X = np.asarray(pickle.load(open(d["X"], "rb")))
        y = np.asarray(pickle.load(open(d["y"], "rb"))).ravel()
        print(f"\n=== partition {part}: {d['X'].name}")
        print(f"shape {X.shape}  dtype {X.dtype}  labels {y.shape}")
        if X.ndim != 3 or len(X) != len(y):
            problems.append(f"partition {part}: bad shape {X.shape} / {y.shape}")
            continue
        n_nan, n_inf = int(np.isnan(X).sum()), int(np.isinf(X).sum())
        print(f"NaN cells {n_nan} ({n_nan / X.size:.3%}), inf cells {n_inf}, value range [{np.nanmin(X):.3g}, {np.nanmax(X):.3g}]")
        vals, cnt = np.unique(y, return_counts=True)
        print("label counts:", dict(zip(vals.tolist(), cnt.tolist())), f"| positive rate {y.mean():.2%}")
        rates[part] = float(y.mean())
        if not set(vals.tolist()) <= {0, 1}:
            problems.append(f"partition {part}: labels not 0/1")
        if y.sum() == 0:
            problems.append(f"partition {part}: no positive samples")
        if X.shape[2] != len(config.CLEANED_ATTRS):
            problems.append(f"partition {part}: {X.shape[2]} attributes, README lists {len(config.CLEANED_ATTRS)}")
        const = [config.CLEANED_ATTRS[j] for j in range(min(X.shape[2], len(config.CLEANED_ATTRS)))
                 if np.nanstd(X[:, :, j]) < 1e-12]
        if const:
            problems.append(f"partition {part}: constant attributes {const}")

        # leakage sniff test: no single physical parameter should separate the classes almost perfectly
        rng = np.random.default_rng(0)
        pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
        if len(pos) and len(neg):
            neg = rng.choice(neg, min(len(neg), max(args.max_rows - len(pos), 1000)), replace=False)
            idx = np.sort(np.concatenate([pos, neg]))
            last = np.nan_to_num(X[idx, -1, :].astype(float))
            avg = np.nan_to_num(np.nanmean(X[idx].astype(float), axis=1))
            aucs = []
            for j in range(last.shape[1]):
                a = max(max(a_, 1 - a_) for a_ in (roc_auc_score(y[idx], last[:, j]), roc_auc_score(y[idx], avg[:, j])))
                aucs.append((a, config.CLEANED_ATTRS[j] if j < len(config.CLEANED_ATTRS) else f"attr{j}"))
            aucs.sort(reverse=True)
            print("strongest single parameters (AUC, best of last value / window mean):", ", ".join(f"{n} {a:.3f}" for a, n in aucs[:4]))
            if aucs[0][0] > 0.95:
                problems.append(f"partition {part}: {aucs[0][1]} alone has AUC {aucs[0][0]:.3f} - suspiciously high, possible leakage")
        del X

    print("\n=== summary")
    print("positive rate per partition:", {k: f"{v:.2%}" for k, v in rates.items()})
    if sorted(files) != [1, 2, 3, 4, 5]:
        problems.append(f"found partitions {sorted(files)}; the default split needs 1-5 (train 1-3, val 4, test 5)")
    print("PROBLEMS:" if problems else "No problems detected.")
    for p in problems:
        print("  -", p)


if __name__ == "__main__":
    main()
