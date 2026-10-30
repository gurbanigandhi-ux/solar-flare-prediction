"""Builds static figures (reports/figures/*.png) and a markdown report (reports/REPORT.md)
from the artefacts written by src.train. Can be re-run on its own:  python -m src.report"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, roc_curve

from . import config
from .metrics import tss

R = config.REPORTS_DIR
F = config.FIG_DIR


def _save(name):
    plt.tight_layout()
    plt.savefig(F / name, dpi=130)
    plt.close()


def make_figures():
    F.mkdir(parents=True, exist_ok=True)
    res = json.load(open(R / "results.json"))
    cmp_df = pd.read_csv(R / "model_comparison.csv")
    tp = pd.read_csv(R / "test_predictions.csv")
    vp = pd.read_csv(R / "val_predictions.csv")
    cand = pd.read_csv(R / "candidate_predictions.csv")
    thr = res["final"]["threshold"]

    # class balance
    bal = pd.DataFrame(res["eda"]["partition_balance"])
    plt.figure(figsize=(6, 3.5))
    plt.bar(bal["partition"].astype(str), bal["pos_rate"] * 100, color="#d9822b")
    plt.xlabel("Partition (chronological)"); plt.ylabel("% major-flare windows (M/X)")
    plt.title("Class imbalance by partition"); _save("class_balance.png")

    # model comparison
    plt.figure(figsize=(8, 4.5))
    d = cmp_df.sort_values("val_TSS")
    y = np.arange(len(d))
    plt.barh(y - 0.2, d["val_TSS"], 0.4, label="validation", color="#4c78a8")
    plt.barh(y + 0.2, d["test_TSS"], 0.4, label="test", color="#f58518")
    plt.yticks(y, d["candidate"]); plt.xlabel("TSS"); plt.legend(); plt.title("Models x imbalance strategies")
    _save("model_comparison.png")

    # ROC / PR
    base = cand[cand["split"] == "test"]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    fpr, tpr, _ = roc_curve(tp["label"], tp["proba"])
    ax[0].plot(fpr, tpr, label="final model", lw=2)
    if "logreg|class_weight" in base:
        f2, t2, _ = roc_curve(base["label"], base["logreg|class_weight"])
        ax[0].plot(f2, t2, label="logistic baseline", lw=1.5, ls="--")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set_xlabel("False positive rate"); ax[0].set_ylabel("True positive rate")
    ax[0].set_title("ROC (test)"); ax[0].legend()
    pr, rc, _ = precision_recall_curve(tp["label"], tp["proba"])
    ax[1].plot(rc, pr, lw=2); ax[1].axhline(tp["label"].mean(), color="k", ls=":", lw=1)
    ax[1].set_xlabel("Recall"); ax[1].set_ylabel("Precision"); ax[1].set_title("Precision-Recall (test)")
    _save("roc_pr.png")

    # confusion matrix
    f = res["final"]["test"]
    cm = np.array([[f["TN"], f["FP"]], [f["FN"], f["TP"]]])
    plt.figure(figsize=(4, 3.6)); plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["No flare", "M/X flare"]); plt.yticks([0, 1], ["No flare", "M/X flare"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion matrix (test)"); _save("confusion_matrix.png")

    # reliability
    frac, mean_p = calibration_curve(tp["label"], tp["proba"], n_bins=8, strategy="quantile")
    plt.figure(figsize=(4.5, 4)); plt.plot(mean_p, frac, "o-"); plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency"); plt.title("Reliability (test)")
    _save("reliability.png")

    # TSS vs threshold
    grid = np.linspace(0.005, 0.9, 120)
    plt.figure(figsize=(6, 3.8))
    plt.plot(grid, [tss(vp["label"], vp["proba"] >= g) for g in grid], label="validation")
    plt.plot(grid, [tss(tp["label"], tp["proba"] >= g) for g in grid], label="test")
    plt.axvline(thr, color="k", ls=":", label=f"chosen = {thr:.3f}")
    plt.xlabel("Probability threshold"); plt.ylabel("TSS"); plt.legend(); plt.title("TSS vs decision threshold")
    _save("tss_threshold.png")

    # importance
    perm = pd.read_csv(R / "permutation_importance.csv").head(15).iloc[::-1]
    plt.figure(figsize=(7, 5)); plt.barh(perm["feature"], perm["importance"], xerr=perm["std"], color="#54a24b")
    plt.xlabel("Drop in ROC-AUC when shuffled"); plt.title("Permutation importance (test)"); _save("permutation_importance.png")
    if res.get("explainability", {}).get("shap") and (R / "shap_values.npz").exists():
        try:
            import shap
            z = np.load(R / "shap_values.npz", allow_pickle=True)
            shap.summary_plot(z["values"], pd.DataFrame(z["data"], columns=z["features"]), show=False, max_display=15)
            _save("shap_summary.png")
        except Exception:
            plt.close("all")


def _fmt(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def make_report():
    make_figures()
    res = json.load(open(R / "results.json"))
    cmp_df = pd.read_csv(R / "model_comparison.csv")
    fs = pd.read_csv(R / "feature_selection.csv")
    perm = pd.read_csv(R / "permutation_importance.csv")
    fin, tuned, eda = res["final"], res["tuning"], res["eda"]
    t, v, b = fin["test"], fin["validation"], fin["baseline_logreg_test"]
    syn = res["eda"]["source"] == "synthetic"
    top_par = perm.groupby("parameter")["importance"].sum().sort_values(ascending=False).head(5)

    md = []
    md.append("# Solar Flare Prediction - Results Report\n")
    if syn:
        md.append("> **These results come from SYNTHETIC data** generated to mimic SWAN-SF. They validate the "
                  "pipeline, not the physics. Re-run with `--source swansf` for real results.\n")
    if eda["source"] == "cleaned":
        md.append("> **Data:** Cleaned SWAN-SF (EskandariNasab et al., 2024), `test/` files: FPCKNN imputation and LSBZM "
                  "normalisation were applied upstream; no resampled or synthetic (TimeGAN) windows are used. Labels are "
                  "binary (M/X vs everything else), so no per-flare-class breakdown is available.\n")
    md.append("## 1. Problem\nPredict whether a solar active region will produce an **M- or X-class flare within the "
              "next 24 hours**, from a 12-hour window of photospheric magnetic-field parameters. Binary classification, "
              "heavily imbalanced.\n")
    md.append("## 2. Data\n"
              f"- Source: `{eda['source']}`; {res['split']['train']['n'] + res['split']['val']['n'] + res['split']['test']['n']} windows, "
              f"{eda['data_quality']['n_features_raw']} raw summary features\n"
              f"- Major-flare rate: {eda['overall_pos_rate']:.2%}"
              + (f" (all M/X windows kept, only {eda['neg_frac']:.0%} of B/C/quiet windows sampled, so this is higher than the "
                 "true rate; TSS, recall and false-alarm rate are unaffected, precision and Brier score are not)" if eda.get("neg_frac") and eda["neg_frac"] < 1 else "")
              + "\n"
              f"- Missing cells: {eda['data_quality']['pct_cells_missing']:.2f}%; duplicate rows: {eda['data_quality']['n_duplicate_rows']}\n"
              f"- Chronological split: train partitions {config.TRAIN_PARTITIONS}, validation {config.VAL_PARTITIONS}, "
              f"test {config.TEST_PARTITIONS}. Test data is never used to choose anything.\n\n"
              "![balance](figures/class_balance.png)\n")
    if res.get("preprocessing", {}).get("mode") == "light":
        md.append("## 3. Preprocessing\nThe cleaned dataset is already imputed and normalised, so the pipeline only keeps safety "
                  "nets: infinite values to NaN, drop columns >40% missing, median imputation of any residual NaN, then "
                  "standardisation. Winsorising and log transforms are skipped because the values are already normalised. "
                  "Each 12-hour window is summarised into mean/std/min/max/last/delta/slope per parameter. All steps are fitted "
                  "on training data only (inside the pipeline).\n")
    else:
        md.append("## 3. Preprocessing\nInfinite values to NaN, drop columns >40% missing, median imputation, winsorise at the "
                  "0.5/99.5 percentiles, signed-log on skewed columns, standardise. All fitted on training data only (inside the pipeline).\n")
    md.append("## 4. Feature selection\n" + fs.round(4).to_markdown(index=False) + "\n")
    md.append("## 5. Model comparison (3 models x 3 imbalance strategies)\n"
              + cmp_df[["candidate", "cv_ROC_AUC_train", "val_TSS", "val_ROC_AUC", "test_TSS", "test_HSS", "test_ROC_AUC"]]
              .round(3).to_markdown(index=False) + "\n\n![models](figures/model_comparison.png)\n")
    md.append(f"## 6. Tuning\nRandomized search ({tuned['n_iter']} iterations, leave-one-partition-out CV) on "
              f"`{tuned['model']}` / `{tuned['strategy']}`. Validation AUC: tuned {tuned['val_AUC_tuned']:.3f} vs default "
              f"{tuned['val_AUC_untuned']:.3f} -> used **{'tuned' if tuned['used_tuned'] else 'default'}**.\n\n"
              f"Best parameters: `{tuned['best_params']}`\n")
    md.append(f"## 7. Final test results ({fin['model']})\n"
              f"Probabilities are Platt-calibrated on the validation partition; the decision threshold ({fin['threshold']:.3f}) "
              f"maximises TSS on validation.\n\n"
              "| Metric | Final model | Logistic baseline |\n|---|---|---|\n"
              f"| TSS | **{_fmt(t['TSS'])}** | {_fmt(b['TSS'])} |\n| HSS | {_fmt(t['HSS'])} | {_fmt(b['HSS'])} |\n"
              f"| Recall | {_fmt(t['recall'])} | {_fmt(b['recall'])} |\n| Precision | {_fmt(t['precision'])} | {_fmt(b['precision'])} |\n"
              f"| ROC-AUC | {_fmt(t['ROC_AUC'])} | {_fmt(b['ROC_AUC'])} |\n| PR-AUC | {_fmt(t['PR_AUC'])} | {_fmt(b['PR_AUC'])} |\n"
              f"| Brier skill score | {_fmt(t.get('BSS'))} | - |\n\n"
              + (f"95% block-bootstrap intervals on the test partition: TSS {fin['test_ci']['TSS'][0]:.3f} to {fin['test_ci']['TSS'][1]:.3f}, "
                 f"recall {fin['test_ci']['recall'][0]:.3f} to {fin['test_ci']['recall'][1]:.3f} "
"(block bootstrap, 24-row blocks; windows overlap heavily, so treat as optimistic). "
f"(logistic baseline TSS "
                 f"{fin['baseline_test_ci']['TSS'][0]:.3f} to {fin['baseline_test_ci']['TSS'][1]:.3f}). If the intervals overlap, "
                 "the improvement over the baseline is not statistically convincing.\n\n" if "test_ci" in fin else "")
              + f"Confusion matrix (test): TP={t['TP']}, FN={t['FN']}, FP={t['FP']}, TN={t['TN']}\n\n"
              "![roc](figures/roc_pr.png)\n![cm](figures/confusion_matrix.png) ![rel](figures/reliability.png)\n"
              "![thr](figures/tss_threshold.png)\n")
    md.append("## 8. Explainability\nMost influential physical parameters (summed permutation importance): "
              + ", ".join(f"`{k}`" for k in top_par.index) + ".\n\n![perm](figures/permutation_importance.png)\n")
    if (config.FIG_DIR / "shap_summary.png").exists():
        md.append("![shap](figures/shap_summary.png)\n")
    md.append("## 9. Limitations and future work\n"
              "- Only photospheric magnetic-field summary statistics are used; adding EUV/AIA texture features, flare history "
              "and sequence models (LSTM/Transformer on the raw time series) are natural extensions.\n"
              "- Flare forecasting skill is modest even for state-of-the-art models; TSS values should be read against published "
              "SWAN-SF benchmarks, not against 1.0.\n"
              "- Partitions are time-ordered but active regions can span partitions boundaries; limb-region data are less reliable.\n"
              "- The test set is one partition with a limited number of M/X events and strongly overlapping windows, so a single TSS "
              "value is uncertain (see the bootstrap interval). Model and threshold selection also rest on one validation partition.\n"
              "- Threshold optimised for TSS; an operational user may prefer a different recall/false-alarm trade-off "
              "(the dashboard has a threshold slider).\n")
    (R / "REPORT.md").write_text("\n".join(md))


if __name__ == "__main__":
    make_report()
