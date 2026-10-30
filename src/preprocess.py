"""Preprocessing. Everything here is an sklearn transformer that is FIT ON TRAINING DATA ONLY
(it lives inside the model Pipeline), so validation/test data never leaks into the statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CleanInfs(BaseEstimator, TransformerMixin):
    """Coerce to a float DataFrame and turn +/-inf into NaN."""

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X).reindex(columns=self.columns_).astype(float)
        return X.replace([np.inf, -np.inf], np.nan)


class DropSparse(BaseEstimator, TransformerMixin):
    """Drop columns with more than `max_missing` fraction of missing values (measured on train)."""

    def __init__(self, max_missing: float = 0.4):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        frac = X.isna().mean()
        self.keep_ = frac[frac <= self.max_missing].index.tolist()
        return self

    def transform(self, X):
        return X[self.keep_]


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its [lower, upper] training percentiles (tames sensor spikes/outliers)."""

    def __init__(self, lower: float = 0.5, upper: float = 99.5):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lo_ = X.quantile(self.lower / 100)
        self.hi_ = X.quantile(self.upper / 100)
        return self

    def transform(self, X):
        return X.clip(self.lo_, self.hi_, axis=1)


class SkewLog(BaseEstimator, TransformerMixin):
    """Signed log1p on columns that are strongly skewed in the training data (|skew| > threshold).
    Magnetic-field quantities span orders of magnitude, so this helps linear models a lot."""

    def __init__(self, skew_threshold: float = 1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        sk = X.skew()
        self.cols_ = sk[sk.abs() > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.cols_:
            X[c] = np.sign(X[c]) * np.log1p(np.abs(X[c]))
        return X


def build_preprocessor(max_missing: float = 0.4, impute: str = "median",
                       winsor=(0.5, 99.5), log_skew: bool = True, scale: bool = True) -> Pipeline:
    steps = [
        ("clean", CleanInfs()),
        ("drop_sparse", DropSparse(max_missing)),
        ("impute", SimpleImputer(strategy=impute).set_output(transform="pandas")),
    ]
    if winsor:
        steps.append(("winsor", Winsorizer(*winsor)))
    if log_skew:
        steps.append(("skewlog", SkewLog()))
    if scale:
        steps.append(("scale", StandardScaler().set_output(transform="pandas")))
    return Pipeline(steps)


def data_quality_report(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Quick summary of the raw data problems the preprocessing stage deals with."""
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    miss = X.isna().mean().sort_values(ascending=False)
    return {
        "n_rows": int(len(df)),
        "n_features_raw": int(len(feature_cols)),
        "pct_cells_missing": float(X.isna().to_numpy().mean() * 100),
        "n_cols_with_missing": int((miss > 0).sum()),
        "worst_missing_cols": {k: float(v) for k, v in miss.head(5).items()},
        "n_duplicate_rows": int(df.duplicated(subset=feature_cols).sum()),
        "n_constant_cols": int((X.nunique(dropna=True) <= 1).sum()),
    }
