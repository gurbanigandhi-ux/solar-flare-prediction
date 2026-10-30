"""Feature selection and model construction."""
from __future__ import annotations

from functools import partial

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from . import config


# ----------------------------------------------------------------------------------------------
# Feature selection
# ----------------------------------------------------------------------------------------------
class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from every pair with |Pearson r| above `threshold` (and constant columns)."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        const = X.columns[X.std() < 1e-12]
        corr = X.drop(columns=const).corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop = [c for c in upper.columns if (upper[c] > self.threshold).any()]
        self.keep_ = [c for c in X.columns if c not in set(drop) | set(const)]
        return self

    def transform(self, X):
        return pd.DataFrame(X)[self.keep_]


SELECTION_METHODS = ["none", "corr", "corr+mutual_info", "corr+rf_importance", "corr+rfe_logreg"]


def make_selector(method: str, k: int = 30, random_state: int = config.RANDOM_STATE):
    """Return an sklearn step (or 'passthrough') implementing the chosen feature selection method."""
    if method == "none":
        return "passthrough"
    steps = [("corr", CorrelationFilter(0.95))]
    if method == "corr+mutual_info":
        steps.append(("mi", SelectKBest(partial(mutual_info_classif, random_state=random_state), k=k)
                      .set_output(transform="pandas")))
    elif method == "corr+rf_importance":
        rf = RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=5,
                                    class_weight="balanced_subsample", n_jobs=-1, random_state=random_state)
        steps.append(("rf", SelectFromModel(rf, max_features=k, threshold=-np.inf).set_output(transform="pandas")))
    elif method == "corr+rfe_logreg":
        lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
        steps.append(("rfe", RFE(lr, n_features_to_select=k, step=0.2).set_output(transform="pandas")))
    elif method != "corr":
        raise ValueError(f"unknown selection method {method}")
    return Pipeline(steps)


# ----------------------------------------------------------------------------------------------
# Imbalance handling + models
# ----------------------------------------------------------------------------------------------
class UnderSamplingClassifier(ClassifierMixin, BaseEstimator):
    """Fit `estimator` on all positives plus a random subset of negatives (neg:pos = `ratio`).
    Sits at the END of the pipeline so preprocessing/selection still see the full training data."""

    def __init__(self, estimator=None, ratio: float = 5.0, random_state: int = config.RANDOM_STATE):
        self.estimator = estimator
        self.ratio = ratio
        self.random_state = random_state

    def fit(self, X, y):
        y = np.asarray(y)
        rng = np.random.default_rng(self.random_state)
        pos = np.flatnonzero(y == 1)
        neg = np.flatnonzero(y == 0)
        n_neg = min(len(neg), int(self.ratio * len(pos)))
        idx = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
        self.estimator_ = clone(self.estimator).fit(X.iloc[idx] if hasattr(X, "iloc") else X[idx], y[idx])
        self.classes_ = self.estimator_.classes_
        return self

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    def predict(self, X):
        return self.estimator_.predict(X)


MODEL_NAMES = ["logreg", "random_forest", "xgboost"]
STRATEGIES = ["none", "class_weight", "undersample"]


def build_model(name: str, strategy: str = "class_weight", pos_weight: float = 1.0, quick: bool = False,
                random_state: int = config.RANDOM_STATE):
    """name in MODEL_NAMES, strategy in STRATEGIES. `pos_weight` = n_negative / n_positive in train."""
    weighted = strategy == "class_weight"
    n_trees = 120 if quick else 300
    if name == "logreg":
        est = LogisticRegression(max_iter=3000, C=0.1, class_weight="balanced" if weighted else None)
    elif name == "random_forest":
        est = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=5, max_features="sqrt", n_jobs=-1,
                                     class_weight="balanced_subsample" if weighted else None,
                                     random_state=random_state)
    elif name == "xgboost":
        est = XGBClassifier(n_estimators=n_trees, max_depth=4, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, eval_metric="logloss", tree_method="hist", n_jobs=-1,
                            scale_pos_weight=pos_weight if weighted else 1.0, random_state=random_state)
    else:
        raise ValueError(name)
    if strategy == "undersample":
        return UnderSamplingClassifier(est, ratio=5.0, random_state=random_state)
    return est


def unwrap(clf):
    """Return the underlying fitted estimator (strips the undersampling wrapper)."""
    return clf.estimator_ if isinstance(clf, UnderSamplingClassifier) else clf


# Hyper-parameter search spaces (keys are relative to the bare estimator; `prefix` is added for the wrapper).
SEARCH_SPACES = {
    "logreg": {"C": [0.001, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    "random_forest": {"n_estimators": [200, 300, 500], "max_depth": [6, 10, 16, None],
                      "min_samples_leaf": [2, 5, 10, 20], "max_features": ["sqrt", 0.3, 0.5]},
    "xgboost": {"n_estimators": [200, 300, 500], "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.02, 0.05, 0.1], "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.5, 0.8, 1.0], "min_child_weight": [1, 5, 10]},
}


def search_space(name: str, strategy: str) -> dict:
    prefix = "clf__estimator__" if strategy == "undersample" else "clf__"
    return {prefix + k: v for k, v in SEARCH_SPACES[name].items()}
