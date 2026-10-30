"""Read-only validation of a finished run (does NOT modify models/ or the existing reports/ files).

    python -m src.audit                      # uses data/processed/cleaned_features.csv + reports/*.csv
Outputs go to reports/audit/ (audit.json + csv files) and a readable summary is printed.

Sections: 1 duplicates | 2 label-shuffle control | 3 simple baselines | 4 leave-one-partition-out |
          5 bootstrap-block sensitivity | 6 threshold sensitivity | 7 prevalence correction | 8 event-level recall
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config
from .data_loading import load_flat_csv
from .metrics import bootstrap_ci, evaluate, tss

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
META = ["sample_id", "partition", "flare_class", "label"]


def youden(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    j[~np.isfinite(thr)] = -1
    return float(thr[int(np.argmax(j))])


def lr(C=0.01):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=C, class_weight="balanced", max_iter=2000))


def fit_eval(Xtr, ytr, Xva, yva, Xte, yte, cols, C=0.01):
    m = lr(C).fit(Xtr[cols], ytr)
    thr = youden(yva, m.predict_proba(Xva[cols])[:, 1])
    pte = m.predict_proba(Xte[cols])[:, 1]
    e = evaluate(yte, pte, thr)
    return {"test_TSS": e["TSS"], "test_AUC": e["ROC_AUC"], "test_recall": e["recall"], "n_features": len(cols)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(config.DATA_PROCESSED / "cleaned_features.csv"))
    ap.add_argument("--reports", default=str(config.REPORTS_DIR))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    R = Path(a.reports)
    out_dir = R / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict = {}

    df = load_flat_csv(a.features)
    feats = [c for c in df.columns if c not in META]
    df = df.assign(_idx=df["sample_id"].str.extract(r"_(\d+)$")[0].astype(int))
    tr = df[df.partition.isin(config.TRAIN_PARTITIONS)]
    va = df[df.partition.isin(config.VAL_PARTITIONS)]
    te = df[df.partition.isin(config.TEST_PARTITIONS)]
    ytr, yva, yte = tr.label.to_numpy(), va.label.to_numpy(), te.label.to_numpy()
    print(f"rows {len(df)}  features {len(feats)}  train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    # ---------------------------------------------------------------- 1. duplicates
    print("\n[1] DUPLICATES (identical across all feature columns)")
    dup = df.duplicated(subset=feats, keep=False)
    g = df[dup].copy()
    g["gid"] = g.groupby(feats, sort=False).ngroup()
    summ = g.groupby("gid").agg(n=("label", "size"), n_partitions=("partition", "nunique"),
                                n_labels=("label", "nunique"), pos=("label", "sum"),
                                idx_span=("_idx", lambda s: int(s.max() - s.min())))
    std_cols = [c for c in feats if c.endswith("__std")]
    flat = (df[std_cols].abs().max(axis=1) < 1e-9) if std_cols else pd.Series(False, index=df.index)
    d1 = {"rows_in_duplicate_groups": int(dup.sum()), "groups": int(len(summ)),
          "groups_spanning_partitions": int((summ.n_partitions > 1).sum()),
          "groups_with_conflicting_labels": int((summ.n_labels > 1).sum()),
          "groups_containing_positives": int((summ.pos > 0).sum()),
          "groups_adjacent_in_file_order(span<=100)": int((summ.idx_span <= 100).sum()),
          "completely_flat_windows(all std=0)": int(flat.sum()),
          "flat_windows_among_duplicates": int((flat & dup).sum())}
    audit["duplicates"] = d1
    for k, v in d1.items():
        print(f"   {k:48s} {v}")
    if len(summ):
        g.sort_values(["gid", "partition", "_idx"])[["gid", "sample_id", "partition", "label"]].to_csv(out_dir / "duplicates.csv", index=False)
        print("   -> reports/audit/duplicates.csv lists every duplicated row")
    print("   READ: spanning_partitions>0 or conflicting_labels>0 needs attention; small numbers of in-partition duplicates are harmless.")

    # ---------------------------------------------------------------- 2. shuffled-label control
    print("\n[2] LABEL-SHUFFLE CONTROL (train on shuffled labels; test AUC should be ~0.5)")
    rng = np.random.default_rng(a.seed)
    real = fit_eval(tr, ytr, va, yva, te, yte, feats)
    shuf = fit_eval(tr, rng.permutation(ytr), va, yva, te, yte, feats)
    audit["label_shuffle"] = {"real_labels": real, "shuffled_labels": shuf}
    print(f"   real labels     : test AUC {real['test_AUC']:.3f}  TSS {real['test_TSS']:.3f}")
    print(f"   shuffled labels : test AUC {shuf['test_AUC']:.3f}  TSS {shuf['test_TSS']:.3f}")
    print("   READ: shuffled AUC far from 0.5 (outside ~0.45-0.55) would mean the pipeline leaks label info through features/order.")

    # ---------------------------------------------------------------- 3. simple baselines
    print("\n[3] SIMPLE BASELINES (logistic regression, C=0.01, threshold from validation)")
    single_auc = {c: roc_auc_score(ytr, tr[c].fillna(tr[c].median())) for c in feats}
    ranked = sorted(single_auc, key=lambda c: -abs(single_auc[c] - 0.5))
    rows = []
    for name, cols in [("best single feature: " + ranked[0], ranked[:1]),
                       ("top 3 features", ranked[:3]),
                       ("top 10 features", ranked[:10]),
                       ("all features", feats)]:
        rows.append({"model": name, **fit_eval(tr, ytr, va, yva, te, yte, cols)})
    base = pd.DataFrame(rows)
    base.to_csv(out_dir / "simple_baselines.csv", index=False)
    audit["simple_baselines"] = rows
    print(base.round(3).to_string(index=False))
    print("   top single features (train AUC):", ", ".join(f"{c} {single_auc[c]:.3f}" for c in ranked[:5]))
    res = json.load(open(R / "results.json"))
    print(f"   project's final model: test AUC {res['final']['test']['ROC_AUC']:.3f}  TSS {res['final']['test']['TSS']:.3f}")
    print("   READ: if one feature already gets close to the final TSS, say so - the skill is mostly 'active-region size'.")

    # ---------------------------------------------------------------- 4. leave-one-partition-out
    print("\n[4] LEAVE-ONE-PARTITION-OUT (all features, logistic C=0.01, threshold = Youden on the training partitions)")
    rows = []
    for p in sorted(df.partition.unique()):
        trn, tst = df[df.partition != p], df[df.partition == p]
        m = lr().fit(trn[feats], trn.label)
        thr = youden(trn.label, m.predict_proba(trn[feats])[:, 1])
        e = evaluate(tst.label, m.predict_proba(tst[feats])[:, 1], thr)
        rows.append({"held_out_partition": p, "n_pos": e["n_pos"], "AUC": e["ROC_AUC"], "TSS": e["TSS"], "recall": e["recall"], "FPR": e["FPR"]})
    lopo = pd.DataFrame(rows)
    lopo.to_csv(out_dir / "leave_one_partition_out.csv", index=False)
    audit["leave_one_partition_out"] = rows
    print(lopo.round(3).to_string(index=False))
    print("   NOTE: this trains on later partitions to predict earlier ones - a stability check, not an operational estimate.")

    # ---------------------------------------------------------------- predictions-based checks
    tp = pd.read_csv(R / "test_predictions.csv")
    thr = res["final"]["threshold"]
    y, p = tp["label"].to_numpy(), tp["proba"].to_numpy()

    print("\n[5] BOOTSTRAP BLOCK-LENGTH SENSITIVITY (test TSS 95% interval)")
    rows = []
    for b in (1, 24, 100, 300, 1000):
        ci = bootstrap_ci(y, p, thr, n_boot=500, block=b)["TSS"]
        rows.append({"block_rows": b, "TSS_low": ci[0], "TSS_high": ci[1]})
        print(f"   block={b:5d}  TSS in [{ci[0]:.3f}, {ci[1]:.3f}]")
    audit["bootstrap_sensitivity"] = rows
    print("   READ: quote the widest interval. Windows overlap ~91%, so block=1 is far too optimistic; the project default is block=24.")

    print("\n[6] THRESHOLD SENSITIVITY (why candidate-table TSS differs from the final calibrated TSS)")
    rows = []
    for mult in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0):
        e = evaluate(y, p, thr * mult)
        rows.append({"threshold": thr * mult, "x_chosen": mult, "TSS": e["TSS"], "recall": e["recall"], "FPR": e["FPR"]})
    fpr_, tpr_, th_ = roc_curve(y, p)
    best_te = float(th_[int(np.argmax(np.where(np.isfinite(th_), tpr_ - fpr_, -1)))])
    e = evaluate(y, p, best_te)
    rows.append({"threshold": best_te, "x_chosen": best_te / thr, "TSS": e["TSS"], "recall": e["recall"], "FPR": e["FPR"]})
    th = pd.DataFrame(rows)
    th.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
    audit["threshold_sensitivity"] = rows
    print(th.round(4).to_string(index=False))
    print("   (last row = threshold that would be best on TEST - for diagnosis only, never use it for reporting)")
    cmp_ = pd.read_csv(R / "model_comparison.csv")
    print("   candidate-table rows use UN-calibrated default-parameter models with their own validation threshold;")
    print("   the final model is tuned + Platt-calibrated and its threshold was re-chosen on calibrated validation scores.")
    print(cmp_[["candidate", "threshold", "test_TSS", "test_recall"]].round(3).head(4).to_string(index=False))

    print("\n[7] NEGATIVE-SAMPLING (prevalence) CORRECTION")
    try:
        nf = json.load(open(str(a.features).replace(".csv", ".meta.json")))["neg_frac"]
    except Exception:
        nf = None
    if nf is None or nf >= 1:
        print("   neg_frac = 1.0 (or unknown): no correction needed/possible.")
    else:
        e = evaluate(y, p, thr)
        tp_, fn_, fp_, tn_ = e["TP"], e["FN"], e["FP"] / nf, e["TN"] / nf
        prec = tp_ / (tp_ + fp_)
        hss = 2 * (tp_ * tn_ - fn_ * fp_) / ((tp_ + fn_) * (fn_ + tn_) + (tp_ + fp_) * (fp_ + tn_))
        odds = p / (1 - np.clip(p, 1e-9, 1 - 1e-9)) * nf               # prior-shift correction of the probabilities
        p_true = odds / (1 + odds)
        w = np.where(y == 0, 1 / nf, 1.0)
        bs = float(np.average((p_true - y) ** 2, weights=w))
        prev = float(np.average(y, weights=w))
        bss = 1 - bs / (prev * (1 - prev))
        audit["prevalence_correction"] = {"neg_frac": nf, "true_pos_rate_test": prev, "precision_sampled": e["precision"],
                                          "precision_corrected": prec, "HSS_sampled": e["HSS"], "HSS_corrected": hss,
                                          "FP_per_TP_corrected": fp_ / tp_, "Brier_corrected": bs, "BSS_corrected": bss,
                                          "TSS_unchanged": e["TSS"]}
        print(f"   test flare rate: sampled {y.mean():.2%} -> corrected {prev:.2%}")
        print(f"   precision: {e['precision']:.3f} -> {prec:.3f}   HSS: {e['HSS']:.3f} -> {hss:.3f}   false alarms per hit: {fp_ / tp_:.1f}")
        print(f"   Brier (prior-corrected probabilities): {bs:.4f}   BSS vs climatology: {bss:.3f}   (TSS/recall unchanged: {e['TSS']:.3f})")
        print("   READ: report the corrected precision/HSS/BSS as the operational numbers; keep sampled ones only as a by-product.")

    print("\n[8] EVENT-LEVEL VIEW (runs of consecutive positive windows in file order ~ one flare episode)")
    te2 = tp.copy()
    te2["_idx"] = te2["sample_id"].str.extract(r"_(\d+)$")[0].astype(int)
    pos = te2[te2.label == 1].sort_values("_idx")
    if len(pos):
        run = (pos["_idx"].diff() != 1).cumsum()
        runs = pos.groupby(run).agg(len_=("pred", "size"), flagged=("pred", "sum"))
        audit["event_level"] = {"positive_windows": int(len(pos)), "episodes": int(len(runs)),
                                "median_windows_per_episode": float(runs.len_.median()),
                                "episodes_detected_at_least_once": int((runs.flagged > 0).sum()),
                                "episode_recall": float((runs.flagged > 0).mean()),
                                "window_recall": float(pos.pred.mean())}
        for k, v in audit["event_level"].items():
            print(f"   {k:34s} {v:.3f}" if isinstance(v, float) else f"   {k:34s} {v}")
        print("   READ: only meaningful if the pickle order is time-ordered within active regions (median episode length of"
              " several to ~24 windows suggests it is). Each flare yields many near-identical positive windows, so"
              " 'window recall' overstates how many distinct events were caught; the number of episodes is the real sample size.")
    json.dump(audit, open(out_dir / "audit.json", "w"), indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nsaved: reports/audit/audit.json (+ csv files)")


if __name__ == "__main__":
    main()
