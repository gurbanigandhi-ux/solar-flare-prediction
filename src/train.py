"""End-to-end pipeline:  data -> EDA/quality -> split -> preprocessing -> feature selection ->
model comparison (3 models x 3 imbalance strategies) -> tuning -> calibration + threshold ->
test evaluation -> explainability -> save model, results, figures and report.

Usage:
    python -m src.train --source synthetic            # works out of the box
    python -m src.train --source cleaned --path data/raw/cleaned_swansf   # Cleaned SWAN-SF test/ pickles
    python -m src.train --source swansf --path data/raw/swansf
    python -m src.train --source csv --path my_table.csv
    add --quick for a fast smoke run
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut, RandomizedSearchCV, cross_val_score
from sklearn.pipeline import Pipeline

from . import config
from .data_loading import (load_cleaned_swansf, load_flat_csv, load_swansf_raw, make_synthetic,
                           split_by_partition)
from .features import MODEL_NAMES, SELECTION_METHODS, STRATEGIES, build_model, make_selector, search_space, unwrap
from .metrics import best_threshold, bootstrap_ci, evaluate
from .model_io import FlareModel, save_model
from .preprocess import build_preprocessor, data_quality_report

warnings.filterwarnings("ignore")
META_COLS = ["sample_id", "partition", "flare_class", "label"]


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def make_pipeline(selection, k, model, strategy, pos_weight, quick=False, light=False):
    # light = data already imputed + normalised upstream (Cleaned SWAN-SF): keep safety nets, skip winsor/log
    prep = build_preprocessor(winsor=None, log_skew=False) if light else build_preprocessor()
    return Pipeline([
        ("prep", prep),
        ("select", make_selector(selection, k)),
        ("clf", build_model(model, strategy, pos_weight, quick)),
    ])


def proba(pipe, X):
    return pipe.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------------------------
def load_data(args) -> pd.DataFrame:
    if args.source == "synthetic":
        n = 5000 if args.quick else args.n_samples
        return make_synthetic(n_samples=n)
    if args.source in ("swansf", "cleaned"):
        cache = config.DATA_PROCESSED / f"{args.source}_features.csv"
        side = config.DATA_PROCESSED / f"{args.source}_features.meta.json"
        if cache.exists() and not args.reload:
            log(f"    using cached features {cache.name} (use --reload to re-parse the raw files)")
            cached = json.load(open(side))["neg_frac"] if side.exists() else None
            if cached is not None and abs(cached - args.neg_frac) > 1e-9:
                log(f"    NOTE: cache was built with neg_frac={cached}, you asked for {args.neg_frac}; add --reload to rebuild")
            args.neg_frac_used = cached
            return load_flat_csv(cache)
        if args.source == "swansf":
            df = load_swansf_raw(args.path, max_files=args.max_files, neg_frac=args.neg_frac)
        else:
            df = load_cleaned_swansf(args.path, neg_frac=args.neg_frac, max_samples=args.max_per_partition)
        config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache, index=False, float_format="%.6g")
        json.dump({"neg_frac": args.neg_frac}, open(side, "w"))
        args.neg_frac_used = args.neg_frac
        return df
    if args.source == "csv":
        return load_flat_csv(args.path)
    raise ValueError(args.source)


def stage_eda(df, feats, source):
    dq = data_quality_report(df, feats)
    balance = (df.groupby("partition")["label"].agg(n="size", positives="sum")
               .assign(pos_rate=lambda d: d.positives / d.n).reset_index())
    classes = df["flare_class"].value_counts().to_dict()
    return {"source": source, "data_quality": dq, "class_counts": classes,
            "partition_balance": balance.to_dict(orient="records"),
            "overall_pos_rate": float(df["label"].mean())}


def run(args):
    t0 = time.time()
    if args.neg_frac is None:
        args.neg_frac = 0.25 if args.source == "cleaned" else 0.05
    if args.path is None:
        args.path = str(config.DATA_RAW / ("cleaned_swansf" if args.source == "cleaned" else "swansf"))
    light = args.prep == "light" or (args.prep == "auto" and args.source == "cleaned")
    config.REPORTS_DIR.mkdir(exist_ok=True, parents=True)
    config.FIG_DIR.mkdir(exist_ok=True, parents=True)
    config.DATA_PROCESSED.mkdir(exist_ok=True, parents=True)

    # 1 ------ data + EDA ------------------------------------------------------------------
    log("1/8 loading data")
    df = load_data(args)
    feats = [c for c in df.columns if c not in META_COLS]
    # the dashboard only needs a sample (keeps the repo small); training below uses the full table
    df.sample(min(len(df), 6000), random_state=config.RANDOM_STATE).sort_index().to_csv(
        config.DATA_PROCESSED / "dataset.csv", index=False, float_format="%.5g")
    results = {"created": datetime.now().isoformat(timespec="seconds"), "quick": args.quick,
               "eda": stage_eda(df, feats, args.source),
               "preprocessing": {"mode": "light" if light else "full"}}
    results["eda"]["neg_frac"] = getattr(args, "neg_frac_used", None)
    log(f"    {len(df)} samples, {len(feats)} raw features, {df['label'].mean():.2%} positive")

    # 2 ------ chronological split -----------------------------------------------------------
    train, val, test = split_by_partition(df)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        if part["label"].nunique() < 2:
            raise SystemExit(f"{name} split has only one class - check partitions/labels.")
    Xtr, ytr = train[feats], train["label"].to_numpy()
    Xva, yva = val[feats], val["label"].to_numpy()
    Xte, yte = test[feats], test["label"].to_numpy()
    pos_weight = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    results["split"] = {k: {"n": int(len(v)), "positives": int(v["label"].sum())}
                        for k, v in [("train", train), ("val", val), ("test", test)]}
    log(f"    split sizes {results['split']}")

    # 3 ------ feature selection comparison -------------------------------------------------
    log("2/8 comparing feature selection methods (XGBoost, class-weighted)")
    k = args.k_features
    fs_rows = []
    for method in SELECTION_METHODS:
        t = time.time()
        pipe = make_pipeline(method, k, "xgboost", "class_weight", pos_weight, args.quick, light).fit(Xtr, ytr)
        p_va = proba(pipe, Xva)
        n_feat = pipe[:-1].transform(Xtr.head(5)).shape[1]
        fs_rows.append({"method": method, "n_features": int(n_feat),
                        "val_ROC_AUC": roc_auc_score(yva, p_va),
                        "val_TSS": evaluate(yva, p_va, best_threshold(yva, p_va))["TSS"],
                        "fit_seconds": round(time.time() - t, 1)})
        log(f"    {method:20s} feats={n_feat:4d}  val AUC={fs_rows[-1]['val_ROC_AUC']:.3f}  TSS={fs_rows[-1]['val_TSS']:.3f}")
    fs = pd.DataFrame(fs_rows)
    # prefer the best val AUC but if within 0.003 of the best, take the one with fewer features
    top = fs["val_ROC_AUC"].max()
    best_sel = fs[fs["val_ROC_AUC"] >= top - 0.003].sort_values("n_features").iloc[0]["method"]
    fs["selected"] = fs["method"] == best_sel
    fs.to_csv(config.REPORTS_DIR / "feature_selection.csv", index=False)
    log(f"    -> using '{best_sel}'")

    # 4 ------ model x imbalance-strategy comparison -----------------------------------------
    log("3/8 comparing models and class-imbalance strategies")
    rows, cand_pred = [], {"val": {"sample_id": val["sample_id"], "label": yva},
                           "test": {"sample_id": test["sample_id"], "label": yte}}
    cv = LeaveOneGroupOut()
    fitted = {}
    for model in MODEL_NAMES:
        for strat in STRATEGIES:
            t = time.time()
            pipe = make_pipeline(best_sel, k, model, strat, pos_weight, args.quick, light).fit(Xtr, ytr)
            p_va, p_te = proba(pipe, Xva), proba(pipe, Xte)
            thr = best_threshold(yva, p_va)
            mv, mt = evaluate(yva, p_va, thr), evaluate(yte, p_te, thr)
            cv_auc = np.nan
            if not args.quick:
                cv_auc = cross_val_score(clone(pipe), Xtr, ytr, groups=train["partition"], cv=cv,
                                         scoring="roc_auc").mean()
            cid = f"{model}|{strat}"
            rows.append({"candidate": cid, "model": model, "strategy": strat, "cv_ROC_AUC_train": cv_auc,
                         "val_TSS": mv["TSS"], "val_HSS": mv["HSS"], "val_ROC_AUC": mv["ROC_AUC"],
                         "val_PR_AUC": mv["PR_AUC"], "threshold": thr,
                         "test_TSS": mt["TSS"], "test_HSS": mt["HSS"], "test_recall": mt["recall"],
                         "test_precision": mt["precision"], "test_ROC_AUC": mt["ROC_AUC"],
                         "test_PR_AUC": mt["PR_AUC"], "fit_seconds": round(time.time() - t, 1)})
            cand_pred["val"][cid] = p_va
            cand_pred["test"][cid] = p_te
            fitted[cid] = pipe
            log(f"    {cid:28s} val TSS={mv['TSS']:.3f}  AUC={mv['ROC_AUC']:.3f}")
    cmp_df = pd.DataFrame(rows)
    # model selection uses ONLY the validation partition (test is never used to choose anything)
    cmp_df = cmp_df.sort_values(["val_TSS", "val_ROC_AUC"], ascending=False).reset_index(drop=True)
    best_cand = cmp_df.iloc[0]
    baseline_row = cmp_df[cmp_df["candidate"] == "logreg|class_weight"].iloc[0]
    log(f"    -> best candidate on validation: {best_cand['candidate']}")

    # 5 ------ hyper-parameter tuning ---------------------------------------------------------
    log("4/8 tuning the best model family (randomized search, leave-one-partition-out CV)")
    b_model, b_strat = best_cand["model"], best_cand["strategy"]
    base_pipe = make_pipeline(best_sel, k, b_model, b_strat, pos_weight, args.quick, light)
    search = RandomizedSearchCV(base_pipe, search_space(b_model, b_strat), n_iter=args.n_iter, cv=cv,
                                scoring="roc_auc", random_state=config.RANDOM_STATE, n_jobs=1, refit=True)
    search.fit(Xtr, ytr, groups=train["partition"])
    tuned = search.best_estimator_
    p_va_t, p_va_b = proba(tuned, Xva), proba(fitted[best_cand["candidate"]], Xva)
    auc_t, auc_b = roc_auc_score(yva, p_va_t), roc_auc_score(yva, p_va_b)
    use_tuned = auc_t >= auc_b
    final_pipe = tuned if use_tuned else fitted[best_cand["candidate"]]
    results["tuning"] = {"model": b_model, "strategy": b_strat, "n_iter": args.n_iter,
                         "best_params": {k_.replace("clf__", "").replace("estimator__", ""): v
                                         for k_, v in search.best_params_.items()},
                         "cv_ROC_AUC": float(search.best_score_), "val_AUC_tuned": auc_t,
                         "val_AUC_untuned": auc_b, "used_tuned": bool(use_tuned)}
    log(f"    val AUC tuned={auc_t:.3f} vs default={auc_b:.3f} -> {'tuned' if use_tuned else 'default'}")

    # 6 ------ calibration, threshold, final test evaluation ----------------------------------
    log("5/8 calibrating probabilities and evaluating on the held-out test partition")
    meta = {"source": args.source, "synthetic": args.source == "synthetic", "model": b_model,
            "strategy": b_strat, "selection": best_sel, "k_features": k,
            "trained": results["created"], "positive_classes": config.POSITIVE_CLASSES}
    fm = FlareModel(final_pipe, feats, meta).fit_calibration(Xva, yva)
    p_va_cal = fm.predict_proba(Xva)
    fm.threshold = best_threshold(yva, p_va_cal)
    p_te_cal = fm.predict_proba(Xte)
    final_val = evaluate(yva, p_va_cal, fm.threshold)
    final_test = evaluate(yte, p_te_cal, fm.threshold)
    base_test = evaluate(yte, cand_pred["test"]["logreg|class_weight"], baseline_row["threshold"])
    results["final"] = {"model": f"{b_model} ({b_strat})", "threshold": fm.threshold,
                        "validation": final_val, "test": final_test, "baseline_logreg_test": base_test,
                        "test_ci": bootstrap_ci(yte, p_te_cal, fm.threshold),
                        "baseline_test_ci": bootstrap_ci(yte, cand_pred["test"]["logreg|class_weight"],
                                                         baseline_row["threshold"])}
    log(f"    TEST  TSS={final_test['TSS']:.3f}  HSS={final_test['HSS']:.3f}  recall={final_test['recall']:.3f}  "
        f"precision={final_test['precision']:.3f}  AUC={final_test['ROC_AUC']:.3f}")
    ci = results["final"]["test_ci"]["TSS"]
    log(f"    TEST TSS 95% block-bootstrap CI: [{ci[0]:.3f}, {ci[1]:.3f}]")
    log(f"    baseline logistic regression TSS={base_test['TSS']:.3f}")

    # 7 ------ explainability -----------------------------------------------------------------
    log("6/8 explainability (permutation importance + SHAP)")
    Xt_te = fm.transform(Xte)
    sel_cols = list(Xt_te.columns)
    rng = np.random.default_rng(config.RANDOM_STATE)
    base_auc = roc_auc_score(yte, fm.raw_score(Xte))
    imp_rows = []
    for c in sel_cols:
        drops = []
        for _ in range(3):
            Xp = Xte.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base_auc - roc_auc_score(yte, fm.raw_score(Xp)))
        imp_rows.append({"feature": c, "parameter": c.split("__")[0],
                         "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    perm = pd.DataFrame(imp_rows).sort_values("importance", ascending=False)
    perm.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)

    shap_ok = False
    try:
        import shap
        clf = fm.classifier
        n_s = min(1500, len(Xt_te))
        Xs = Xt_te.sample(n_s, random_state=config.RANDOM_STATE)
        if hasattr(clf, "get_booster") or hasattr(clf, "estimators_"):
            sv = shap.TreeExplainer(clf).shap_values(Xs)
        else:
            sv = shap.LinearExplainer(clf, Xt_te).shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[-1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, -1]
        shap_df = pd.DataFrame({"feature": sel_cols, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        shap_df["parameter"] = shap_df["feature"].str.split("__").str[0]
        shap_df.sort_values("mean_abs_shap", ascending=False).to_csv(config.REPORTS_DIR / "shap_importance.csv", index=False)
        np.savez_compressed(config.REPORTS_DIR / "shap_values.npz", values=sv, data=Xs.to_numpy(), features=np.array(sel_cols))
        shap_ok = True
    except Exception as e:  # SHAP is a bonus; never fail the pipeline because of it
        log(f"    SHAP skipped: {e}")
    results["explainability"] = {"shap": shap_ok}

    # 8 ------ save everything ---------------------------------------------------------------
    log("7/8 saving model, predictions, tables")
    save_model(fm)
    cmp_df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    tp = test[["sample_id", "partition", "flare_class", "label"]].copy()
    tp["proba"] = p_te_cal
    tp["pred"] = (p_te_cal >= fm.threshold).astype(int)
    tp.to_csv(config.REPORTS_DIR / "test_predictions.csv", index=False)
    vp = val[["sample_id", "partition", "flare_class", "label"]].copy()
    vp["proba"] = p_va_cal
    vp.to_csv(config.REPORTS_DIR / "val_predictions.csv", index=False)
    pd.concat([pd.DataFrame({"split": s, **{k_: v for k_, v in d.items()}}) for s, d in cand_pred.items()]
              ).to_csv(config.REPORTS_DIR / "candidate_predictions.csv", index=False)
    results["runtime_seconds"] = round(time.time() - t0, 1)
    with open(config.REPORTS_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    log("8/8 figures + report")
    from .report import make_report
    make_report()
    log(f"done in {results['runtime_seconds']}s  ->  streamlit run app/dashboard.py")
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["synthetic", "cleaned", "swansf", "csv"], default="synthetic")
    ap.add_argument("--path", default=None, help="data folder/file (defaults depend on --source)")
    ap.add_argument("--n-samples", type=int, default=12000, help="synthetic data size")
    ap.add_argument("--max-files", type=int, default=None, help="debug: random sample of this many SWAN-SF files per partition")
    ap.add_argument("--neg-frac", type=float, default=None,
                    help="swansf/cleaned: fraction of negative windows to keep (all M/X are kept). 1.0 = everything. "
                         "Default 0.05 (swansf raw) / 0.25 (cleaned)")
    ap.add_argument("--max-per-partition", type=int, default=None,
                    help="cleaned: cap rows kept per partition (all positives always kept) - for quick tests")
    ap.add_argument("--prep", choices=["auto", "full", "light"], default="auto",
                    help="light = skip winsorise/log (data already normalised upstream). auto: light for --source cleaned")
    ap.add_argument("--reload", action="store_true", help="re-parse raw SWAN-SF instead of using the cached features")
    ap.add_argument("--k-features", type=int, default=30)
    ap.add_argument("--n-iter", type=int, default=12, help="random-search iterations")
    ap.add_argument("--quick", action="store_true", help="fast smoke run")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
