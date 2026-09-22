from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_true) == 0 or k <= 0:
        return np.nan
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(np.mean(y_true[order]))


def recall_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    positives = y_true.sum()
    if positives == 0 or k <= 0:
        return np.nan
    k = min(k, len(y_true))
    order = np.argsort(-y_score)[:k]
    return float(y_true[order].sum() / positives)


def lift_at_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.01) -> float:
    base = float(np.mean(y_true)) if len(y_true) else np.nan
    if not base or np.isnan(base):
        return np.nan
    k = max(1, int(np.ceil(len(y_true) * fraction)))
    return precision_at_k(y_true, y_score, k) / base


def rare_event_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)
    out = {
        "n": float(len(y_true)),
        "events": float(y_true.sum()),
        "base_event_rate": float(np.mean(y_true)) if len(y_true) else np.nan,
        "model_event_rate": float(np.mean(y_pred)) if len(y_pred) else np.nan,
        "brier": brier_score_loss(y_true, y_score) if len(np.unique(y_true)) > 1 else np.nan,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision_at_10": precision_at_k(y_true, y_score, 10),
        "precision_at_100": precision_at_k(y_true, y_score, 100),
        "recall_at_100": recall_at_k(y_true, y_score, 100),
        "lift_top_1pct": lift_at_fraction(y_true, y_score, 0.01),
        "lift_top_5pct": lift_at_fraction(y_true, y_score, 0.05),
    }
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = roc_auc_score(y_true, y_score)
        out["pr_auc"] = average_precision_score(y_true, y_score)
    else:
        out["roc_auc"] = np.nan
        out["pr_auc"] = np.nan
    return out


def daily_rank_table(predictions: pd.DataFrame, score_col: str) -> pd.DataFrame:
    out = predictions.copy()
    out["model_rank"] = out.groupby("date")[score_col].rank(ascending=False, method="first")
    return out.sort_values(["date", "model_rank"])
