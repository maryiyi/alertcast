"""Baseline models, LightGBM trainer, and evaluation utilities for AlertCast."""

from typing import Any
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


class BaselineB0Prior:
    """B0: Per-oblast historical prior base rate."""
    def __init__(self):
        self.prior_map: dict[str, float] = {}
        self.global_prior: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        df = pd.DataFrame({"oblast": X_train.index.get_level_values("oblast"), "y": y_train.values})
        self.global_prior = float(df["y"].mean())
        self.prior_map = df.groupby("oblast", observed=False)["y"].mean().to_dict()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        oblasts = X.index.get_level_values("oblast")
        probs = np.array([self.prior_map.get(ob, self.global_prior) for ob in oblasts], dtype=np.float64)
        return probs


class BaselineB1RecentActivity:
    """B1: Recent alert activity baseline (score based on starts_3h)."""
    def __init__(self):
        self.max_starts: float = 1.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.max_starts = max(1.0, float(X_train["starts_3h"].quantile(0.999)))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # Logistic / linear score bounded [0, 1]
        scores = np.clip(X["starts_3h"].values / self.max_starts, 0.0, 1.0)
        return scores


class BaselineB2HourlyClimatology:
    """B2: Hourly climatology (per oblast and hour_local with Laplace smoothing)."""
    def __init__(self):
        self.climatology_map: dict[tuple[str, int], float] = {}
        self.global_prior: float = 0.0

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        df = pd.DataFrame({
            "oblast": X_train.index.get_level_values("oblast"),
            "hour_local": X_train["hour_local"].values,
            "y": y_train.values,
        })
        self.global_prior = (df["y"].sum() + 1.0) / (len(df) + 2.0)
        
        # Laplace smoothed rates
        grouped = df.groupby(["oblast", "hour_local"], observed=False)["y"].agg(["sum", "count"])
        for (ob, h), row in grouped.iterrows():
            prob = (row["sum"] + 1.0) / (row["count"] + 2.0)
            self.climatology_map[(ob, h)] = float(prob)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        oblasts = X.index.get_level_values("oblast")
        hours = X["hour_local"].values
        probs = np.array(
            [self.climatology_map.get((ob, h), self.global_prior) for ob, h in zip(oblasts, hours)],
            dtype=np.float64,
        )
        return probs


class BaselineB3TrailingRate:
    """B3: Trailing 30-day base rate converted to 3-hour window rate."""
    def __init__(self):
        pass

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        # start_rate_30d is starts/day. Expected starts in 3h = (rate / 24) * 3
        # Poisson probability of >= 1 start: 1 - exp(-lambda)
        lam = (X["start_rate_30d"].values / 24.0) * 3.0
        prob = 1.0 - np.exp(-lam)
        return np.clip(prob, 0.0, 1.0)


def compute_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray, b2_brier: float | None = None) -> dict[str, float]:
    """Compute standard metrics: PR-AUC, ROC-AUC, Brier score, and Brier skill score."""
    pr_auc = float(average_precision_score(y_true, y_pred_proba))
    roc_auc = float(roc_auc_score(y_true, y_pred_proba))
    brier = float(brier_score_loss(y_true, y_pred_proba))
    
    bss = None
    if b2_brier is not None and b2_brier > 0:
        bss = float(1.0 - (brier / b2_brier))

    res = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier": brier,
    }
    if bss is not None:
        res["brier_skill_score"] = bss
    return res


def train_lightgbm_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 250,
    learning_rate: float = 0.05,
    num_leaves: int = 31,
    seed: int = 42,
) -> LGBMClassifier:
    """Train LightGBM classifier with fixed seed and categorical features."""
    clf = LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=seed,
        deterministic=True,
        force_row_wise=True,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X_train, y_train, categorical_feature=["oblast_id"])
    return clf
