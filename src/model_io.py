"""Final deployable model: preprocessing + selection + classifier + probability calibration + threshold."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import config


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class FlareModel:
    """Wraps a fitted sklearn Pipeline. Platt scaling (fit on the validation partition) turns the raw
    classifier score into a probability that matches the true flare rate; `threshold` is the TSS-optimal
    cut-off on that probability."""

    def __init__(self, pipeline, raw_features: list[str], meta: dict):
        self.pipeline = pipeline
        self.raw_features = raw_features
        self.meta = meta
        self.calibrator_ = None
        self.threshold = 0.5

    def raw_score(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X[self.raw_features])[:, 1]

    def fit_calibration(self, X_val: pd.DataFrame, y_val) -> "FlareModel":
        s = self.raw_score(X_val)
        self.calibrator_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(s), np.asarray(y_val))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        s = self.raw_score(X)
        if self.calibrator_ is None:
            return s
        return self.calibrator_.predict_proba(_logit(s))[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Features after preprocessing + selection (what the classifier actually sees)."""
        return self.pipeline[:-1].transform(X[self.raw_features])

    @property
    def classifier(self):
        from .features import unwrap
        return unwrap(self.pipeline.named_steps["clf"])


def save_model(model: FlareModel, path=None):
    path = path or (config.MODELS_DIR / "final_model.joblib")
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return path


def load_model(path=None) -> FlareModel:
    return joblib.load(path or (config.MODELS_DIR / "final_model.joblib"))
