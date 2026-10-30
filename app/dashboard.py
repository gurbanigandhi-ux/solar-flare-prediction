"""Solar Flare Prediction dashboard.   Run from the project root:   streamlit run app/dashboard.py"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import precision_recall_curve, roc_curve

from src import config
from src.metrics import evaluate
from src.model_io import load_model

st.set_page_config(page_title="Solar Flare Prediction", page_icon="☀️", layout="wide")
R = config.REPORTS_DIR


# ------------------------------------------------------------------------------------------
@st.cache_data
def load_artifacts():
    need = [R / "results.json", R / "test_predictions.csv", config.DATA_PROCESSED / "dataset.csv"]
    if not all(p.exists() for p in need):
        return None
    out = {
        "res": json.load(open(R / "results.json")),
        "data": pd.read_csv(config.DATA_PROCESSED / "dataset.csv"),
        "cmp": pd.read_csv(R / "model_comparison.csv"),
        "fs": pd.read_csv(R / "feature_selection.csv"),
        "test": pd.read_csv(R / "test_predictions.csv"),
        "val": pd.read_csv(R / "val_predictions.csv"),
        "cand": pd.read_csv(R / "candidate_predictions.csv"),
        "perm": pd.read_csv(R / "permutation_importance.csv"),
        "shap": pd.read_csv(R / "shap_importance.csv") if (R / "shap_importance.csv").exists() else None,
    }
    return out


@st.cache_resource
def get_model():
    return load_model()


A = load_artifacts()
if A is None:
    st.error("No results found. Run the training pipeline first (see README), then reload.")
    st.stop()

res, data = A["res"], A["data"]
model = get_model()
meta = model.meta
META = ["sample_id", "partition", "flare_class", "label"]
feat_cols = [c for c in data.columns if c not in META]
params = sorted({c.split("__")[0] for c in feat_cols})

# ------------------------------------------------------------------------------------------
st.sidebar.title("☀️ Solar Flare Prediction")
page = st.sidebar.radio("Page", ["Overview & EDA", "Preprocessing & features", "Model comparison",
                                 "Final model performance", "Explainability", "What-if explorer",
                                 "Report & downloads"])
if meta.get("synthetic"):
    st.sidebar.warning("Using SYNTHETIC data (SWAN-SF-like). Results validate the pipeline, not the physics.")
st.sidebar.caption(f"Model: {res['final']['model']}  \nTrained: {res['created']}")

PLOT = dict(template="plotly_white")

# ------------------------------------------------------------------------------------------
if page == "Overview & EDA":
    st.title("Overview & exploratory analysis")
    st.write("**Goal:** predict whether an active region will produce an **M/X-class flare in the next 24 h** "
             "from 12 hours of magnetic-field parameters.")
    nf = res["eda"].get("neg_frac")
    st.caption("Headline numbers use the full training table; the charts below use a random sample of up to 6,000 windows."
               + (f" Only {nf:.0%} of B/C/flare-quiet windows were kept (all M/X kept), so class counts and the flare rate "
                  "are higher than in reality." if nf and nf < 1 else ""))
    dq = res["eda"]["data_quality"]
    c = st.columns(5)
    c[0].metric("Windows", f"{dq['n_rows']:,}")
    c[1].metric("Raw features", dq["n_features_raw"])
    c[2].metric("Major-flare rate", f"{res['eda']['overall_pos_rate']:.2%}")
    c[3].metric("Missing cells", f"{dq['pct_cells_missing']:.2f}%")
    c[4].metric("Duplicates", dq["n_duplicate_rows"])

    l, r = st.columns(2)
    cls_order = ["N", "B", "C", "M", "X"]
    counts = data["flare_class"].value_counts().reindex(cls_order).fillna(0).reset_index()
    counts.columns = ["flare_class", "count"]
    l.plotly_chart(px.bar(counts, x="flare_class", y="count", log_y=True, title="Flare class counts (log scale)",
                          color="flare_class", **PLOT), width="stretch")
    bal = pd.DataFrame(res["eda"]["partition_balance"])
    r.plotly_chart(px.bar(bal, x="partition", y="pos_rate", title="Major-flare rate by partition (chronological)",
                          labels={"pos_rate": "M/X rate"}, **PLOT), width="stretch")

    st.subheader("Distribution of a parameter by outcome")
    p1, p2 = st.columns([1, 3])
    par = p1.selectbox("Parameter", params, index=params.index("R_VALUE") if "R_VALUE" in params else 0)
    stat = p1.selectbox("Statistic", ["mean", "max", "last", "std", "slope", "delta", "min"])
    col = f"{par}__{stat}"
    d = data[[col, "label"]].dropna()
    d["outcome"] = d["label"].map({0: "No M/X flare", 1: "M/X flare"})
    d["value"] = np.sign(d[col]) * np.log1p(d[col].abs()) if p1.checkbox("Signed-log scale", True) else d[col]
    p2.plotly_chart(px.histogram(d, x="value", color="outcome", barmode="overlay", opacity=0.6, nbins=60,
                                 histnorm="probability density", title=f"{col}", **PLOT), width="stretch")

    st.subheader("Correlation between parameter means")
    mean_cols = [f"{p}__mean" for p in params if f"{p}__mean" in data]
    corr = data[mean_cols].rename(columns=lambda c: c.split("__")[0]).corr()
    st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto", **PLOT),
                    width="stretch")
    st.caption("Many magnetic parameters are strongly correlated - one reason feature selection matters.")

# ------------------------------------------------------------------------------------------
elif page == "Preprocessing & features":
    st.title("Preprocessing & feature selection")
    st.markdown("""
**Preprocessing (fit on training partitions only, inside the model pipeline):**
The cleaned SWAN-SF data is **already imputed and normalised upstream**, so only light safety nets run:
1. Replace ±inf with NaN → 2. drop columns with >40 % missing → 3. median imputation of any residual NaN →
4. standardise. (Winsorising and log transforms are skipped for the cleaned data; they are used only for raw SWAN-SF.)

**Feature engineering:** each 12-hour parameter series is summarised with mean, std, min, max, last value,
change (last−first) and linear slope.

**Split:** chronological partitions - train {tr}, validation {va}, test {te}. No random shuffling (it would leak the future).
""".format(tr=config.TRAIN_PARTITIONS, va=config.VAL_PARTITIONS, te=config.TEST_PARTITIONS))
    sp = res["split"]
    st.table(pd.DataFrame(sp).T.rename(columns={"n": "windows", "positives": "M/X flares"}))

    st.subheader("Feature selection comparison")
    fs = A["fs"].copy()
    st.dataframe(fs.style.format({"val_ROC_AUC": "{:.3f}", "val_TSS": "{:.3f}"}), width="stretch")
    st.plotly_chart(px.scatter(fs, x="n_features", y="val_ROC_AUC", text="method", size_max=14,
                               title="Validation ROC-AUC vs number of features", **PLOT).update_traces(textposition="top center"),
                    width="stretch")
    st.info(f"Selected method: **{meta['selection']}**")

    st.subheader("Missing values (raw data, top 15 columns)")
    miss = data[feat_cols].isna().mean().sort_values(ascending=False).head(15).reset_index()
    miss.columns = ["feature", "fraction_missing"]
    st.plotly_chart(px.bar(miss, x="fraction_missing", y="feature", orientation="h", **PLOT), width="stretch")

# ------------------------------------------------------------------------------------------
elif page == "Model comparison":
    st.title("Model comparison")
    st.write("Three models × three ways of handling class imbalance. **Selection uses validation only**; "
             "the held-out test partition is reported once, for the final model, on the next page.")
    cmp_df = A["cmp"]
    st.dataframe(cmp_df[["candidate", "cv_ROC_AUC_train", "val_TSS", "val_HSS", "val_ROC_AUC", "fit_seconds"]]
                 .style.format({c: "{:.3f}" for c in cmp_df.columns if c.startswith(("val_", "cv_"))})
                 .background_gradient(subset=["val_TSS"], cmap="Greens"), width="stretch")
    m = cmp_df.melt(id_vars=["model", "strategy"], value_vars=["val_TSS"], var_name="split", value_name="TSS")
    st.plotly_chart(px.bar(m, x="model", y="TSS", color="strategy", facet_col="split", barmode="group",
                           title="TSS by model and imbalance strategy", **PLOT), width="stretch")
    tune = res["tuning"]
    st.subheader("Hyper-parameter tuning")
    st.write(f"`{tune['model']}` / `{tune['strategy']}` - {tune['n_iter']} random-search iterations, "
             f"leave-one-partition-out CV. Validation AUC tuned **{tune['val_AUC_tuned']:.3f}** vs default "
             f"**{tune['val_AUC_untuned']:.3f}** → used **{'tuned' if tune['used_tuned'] else 'default'}** parameters.")
    st.json(tune["best_params"])

# ------------------------------------------------------------------------------------------
elif page == "Final model performance":
    st.title("Final model performance (held-out test partition)")
    test = A["test"]
    split = st.radio("Evaluate on", ["Test", "Validation"], horizontal=True)
    df = test if split == "Test" else A["val"]
    default_thr = float(res["final"]["threshold"])
    thr = st.slider("Decision threshold", 0.005, 0.95, default_thr, 0.005,
                    help="Lower = catch more flares but more false alarms.")
    m = evaluate(df["label"], df["proba"], thr)
    c = st.columns(6)
    for col, key in zip(c, ["TSS", "HSS", "recall", "precision", "ROC_AUC", "PR_AUC"]):
        col.metric(key.replace("_", "-"), f"{m[key]:.3f}")
    st.caption(f"Default threshold {default_thr:.3f} maximises TSS on validation. "
               f"Positives in {split.lower()}: {m['n_pos']} of {m['n']}.")
    _nf = res["eda"].get("neg_frac")
    if _nf and _nf < 1:
        _ap = R / "audit" / "audit.json"
        _pc = json.load(open(_ap)).get("prevalence_correction") if _ap.exists() else None
        st.caption(f"⚠️ Negatives were sampled at {_nf:.0%}, so precision, PR-AUC and the probabilities on this page refer to a "
                   "higher flare rate than reality. TSS, recall and false-alarm rate are unaffected."
                   + (f" Prevalence-corrected test precision ≈ {_pc['precision_corrected']:.2f}, HSS ≈ {_pc['HSS_corrected']:.2f} "
                      f"(~{_pc['FP_per_TP_corrected']:.0f} false-alarm windows per hit)." if _pc else
                      " Run `python -m src.audit` to get prevalence-corrected precision and HSS."))

    l, r = st.columns(2)
    fpr, tpr, _ = roc_curve(df["label"], df["proba"])
    fig = go.Figure(go.Scatter(x=fpr, y=tpr, name="final model", mode="lines"))
    base = A["cand"][A["cand"]["split"] == split.lower()]
    if "logreg|class_weight" in base:
        f2, t2, _ = roc_curve(base["label"], base["logreg|class_weight"])
        fig.add_trace(go.Scatter(x=f2, y=t2, name="logistic baseline", mode="lines", line=dict(dash="dash")))
    fig.add_trace(go.Scatter(x=[m["FPR"]], y=[m["recall"]], mode="markers", marker=dict(size=12, color="red"), name="threshold"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="grey"), showlegend=False))
    fig.update_layout(title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate", **PLOT)
    l.plotly_chart(fig, width="stretch")

    pr, rc, _ = precision_recall_curve(df["label"], df["proba"])
    fig = go.Figure(go.Scatter(x=rc, y=pr, mode="lines", name="PR"))
    fig.add_hline(y=m["n_pos"] / m["n"], line_dash="dot", annotation_text="no-skill")
    fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers", marker=dict(size=12, color="red"), name="threshold"))
    fig.update_layout(title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision", **PLOT)
    r.plotly_chart(fig, width="stretch")

    l, r = st.columns(2)
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    l.plotly_chart(px.imshow(cm, text_auto=True, x=["No flare", "M/X flare"], y=["No flare", "M/X flare"],
                             labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues",
                             title="Confusion matrix", **PLOT), width="stretch")
    grid = np.linspace(0.005, 0.9, 90)
    tss_curve = pd.DataFrame({"threshold": grid, "TSS": [evaluate(df["label"], df["proba"], g)["TSS"] for g in grid]})
    r.plotly_chart(px.line(tss_curve, x="threshold", y="TSS", title="TSS vs threshold", **PLOT)
                   .add_vline(x=thr, line_dash="dot"), width="stretch")

    st.subheader("Probability distribution & calibration")
    l, r = st.columns(2)
    d2 = df.assign(outcome=df["label"].map({0: "No M/X flare", 1: "M/X flare"}))
    l.plotly_chart(px.histogram(d2, x="proba", color="outcome", nbins=50, barmode="overlay", opacity=0.65, log_y=True,
                                title="Predicted probability by true outcome", **PLOT), width="stretch")
    q = pd.qcut(df["proba"], 8, duplicates="drop")
    cal = df.groupby(q, observed=True).agg(pred=("proba", "mean"), obs=("label", "mean")).reset_index(drop=True)
    fig = go.Figure(go.Scatter(x=cal["pred"], y=cal["obs"], mode="lines+markers", name="model"))
    fig.add_trace(go.Scatter(x=[0, cal["pred"].max()], y=[0, cal["pred"].max()], mode="lines", line=dict(dash="dot"), name="perfect"))
    fig.update_layout(title="Reliability diagram", xaxis_title="Mean predicted probability", yaxis_title="Observed flare frequency", **PLOT)
    r.plotly_chart(fig, width="stretch")

    st.subheader("Error analysis by flare class")
    ea = df.assign(pred=(df["proba"] >= thr).astype(int)).groupby("flare_class").agg(
        windows=("label", "size"), flagged=("pred", "sum"), mean_probability=("proba", "mean")).reindex(["N", "B", "C", "M", "X"]).dropna()
    ea["flagged_%"] = 100 * ea["flagged"] / ea["windows"]
    st.dataframe(ea.style.format({"mean_probability": "{:.3f}", "flagged_%": "{:.1f}"}), width="stretch")
    st.caption("M and X rows are the ones we want flagged (recall); B/C/N flagged = false alarms. "
               "C-class false alarms are the hardest to avoid.")

# ------------------------------------------------------------------------------------------
elif page == "Explainability":
    st.title("What drives the predictions?")
    perm = A["perm"]
    top_n = st.slider("Features to show", 5, min(40, len(perm)), 15)
    l, r = st.columns(2)
    p = perm.head(top_n).iloc[::-1]
    l.plotly_chart(px.bar(p, x="importance", y="feature", error_x="std", orientation="h",
                          title="Permutation importance (test AUC drop)", **PLOT), width="stretch")
    agg = perm.groupby("parameter")["importance"].sum().sort_values(ascending=False).head(12).iloc[::-1].reset_index()
    r.plotly_chart(px.bar(agg, x="importance", y="parameter", orientation="h",
                          title="Aggregated by physical parameter", **PLOT), width="stretch")
    if A["shap"] is not None:
        st.subheader("SHAP")
        sh = A["shap"].head(top_n).iloc[::-1]
        st.plotly_chart(px.bar(sh, x="mean_abs_shap", y="feature", orientation="h", title="Mean |SHAP value|", **PLOT),
                        width="stretch")
        fig_path = config.FIG_DIR / "shap_summary.png"
        if fig_path.exists():
            st.image(str(fig_path), caption="SHAP summary (each dot is a window; colour = feature value)")
    st.subheader("Parameter glossary")
    gl = pd.DataFrame({"parameter": list(config.PARAMS), "meaning": list(config.PARAMS.values())})
    st.dataframe(gl[gl["parameter"].isin(params)], width="stretch", hide_index=True)
    st.caption("Importances are shared among correlated features, so read the aggregated-by-parameter view as the "
               "more stable summary.")

# ------------------------------------------------------------------------------------------
elif page == "What-if explorer":
    st.title("What-if explorer")
    st.caption("Scores historical windows from the test partition and lets you rescale parameters. It is a sensitivity demo, not a real-time forecast.")
    tab1, tab2 = st.tabs(["Interactive what-if", "Batch (CSV upload)"])

    with tab1:
        st.write("Start from a real window, then scale individual magnetic parameters to see how the predicted "
                 "probability responds.")
        test_df = data[data["partition"].isin(config.TEST_PARTITIONS)].reset_index(drop=True)
        c1, c2 = st.columns([1, 2])
        mode = c1.selectbox("Starting window", ["Random quiet window", "Random flaring (M/X) window", "Median window"])
        seed = c1.number_input("Pick #", 0, 999, 0)
        if mode == "Median window":
            base = test_df[feat_cols].median().to_frame().T
            truth = "n/a"
        else:
            want = 1 if "flaring" in mode else 0
            pool = test_df[test_df["label"] == want]
            row = pool.iloc[int(seed) % len(pool)]
            base = row[feat_cols].to_frame().T.astype(float)
            truth = f"{row['flare_class']} (actual)"
        c1.caption(f"Window class: {truth}")

        top_params = (A["perm"].groupby("parameter")["importance"].sum().sort_values(ascending=False).head(6).index.tolist())
        c2.write("**Scale the most influential parameters** (1.0 = unchanged):")
        x = base.copy()
        cols = c2.columns(2)
        for i, p in enumerate(top_params):
            f = cols[i % 2].slider(f"{p}", 0.2, 3.0, 1.0, 0.05, key=f"s_{p}")
            affected = [c for c in feat_cols if c.startswith(p + "__")]
            x[affected] = x[affected] * f
        prob = float(model.predict_proba(x)[0])
        base_prob = float(model.predict_proba(base)[0])
        flag = prob >= model.threshold
        g1, g2 = st.columns([1, 1])
        g1.plotly_chart(go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": " %"},
            title={"text": "P(M/X flare in next 24 h)"},
            gauge={"axis": {"range": [0, max(50, prob * 100 * 1.3)]},
                   "threshold": {"line": {"color": "red", "width": 3}, "value": model.threshold * 100},
                   "bar": {"color": "#d9822b"}})).update_layout(height=300), width="stretch")
        g2.metric("Forecast", "⚠️ FLARE ALERT" if flag else "✅ No alert", delta=f"{(prob - base_prob) * 100:+.1f} pts vs unscaled")
        g2.caption(f"Alert threshold = {model.threshold * 100:.1f} % (maximises TSS). Probabilities are on the negative-sampled "
                   f"flare rate ({res['eda']['overall_pos_rate']:.1%}), so they overstate real-world risk.")
        g2.caption("Scaling multiplies all summary statistics of the parameter together, so the window stays self-consistent.")

    with tab2:
        st.write("Upload a CSV with the same feature columns as the training table "
                 "(see `data/processed/dataset.csv`; extra columns are ignored).")
        up = st.file_uploader("CSV", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            missing = [c for c in model.raw_features if c not in new.columns]
            if missing:
                st.error(f"Missing {len(missing)} required columns, e.g. {missing[:5]}")
            else:
                out = new.copy()
                out["flare_probability"] = model.predict_proba(new)
                out["alert"] = (out["flare_probability"] >= model.threshold).astype(int)
                st.dataframe(out[[c for c in ["sample_id", "flare_probability", "alert"] if c in out]].head(200), width="stretch")
                st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        else:
            sample = data.sample(min(50, len(data)), random_state=0)
            st.download_button("Download a sample input file", sample.to_csv(index=False).encode(), "sample_input.csv", "text/csv")

# ------------------------------------------------------------------------------------------
else:
    st.title("Report & downloads")
    rep = R / "REPORT.md"
    if rep.exists():
        text = rep.read_text()
        st.download_button("Download REPORT.md", text, "REPORT.md")
        st.download_button("Download test predictions (CSV)", A["test"].to_csv(index=False).encode(), "test_predictions.csv")
        st.download_button("Download model comparison (CSV)", A["cmp"].to_csv(index=False).encode(), "model_comparison.csv")
        with st.expander("Preview report", expanded=True):
            st.markdown("\n".join(l for l in text.splitlines() if not l.startswith("![")))
        st.subheader("Static figures")
        figs = sorted(config.FIG_DIR.glob("*.png"))
        cols = st.columns(2)
        for i, f in enumerate(figs):
            cols[i % 2].image(str(f), caption=f.stem.replace("_", " "))
    else:
        st.warning("No report found. Run the training pipeline first.")
