from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import rare_event_metrics
from .leakage_tests import fail_if_label_columns_in_features


def optional_estimators(random_state: int = 42) -> dict[str, object]:
    estimators: dict[str, object] = {}
    try:
        from xgboost import XGBClassifier

        estimators["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
        )
    except Exception:
        pass
    try:
        from lightgbm import LGBMClassifier

        estimators["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, random_state=random_state)
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier

        estimators["catboost"] = CatBoostClassifier(iterations=300, learning_rate=0.05, verbose=False, random_seed=random_state)
    except Exception:
        pass
    return estimators


def numeric_pipeline(estimator: object) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler(with_mean=False)),
            ("model", estimator),
        ]
    )


def model_zoo(random_state: int = 42) -> dict[str, Pipeline]:
    base: dict[str, object] = {
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=500, random_state=random_state),
    }
    base.update(optional_estimators(random_state))
    return {name: numeric_pipeline(model) for name, model in base.items()}


@dataclass(frozen=True)
class ChronologicalSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str


def split_by_dates(frame: pd.DataFrame, split: ChronologicalSplit) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= split.train_start) & (df["date"] <= split.train_end)]
    val = df[(df["date"] >= split.validation_start) & (df["date"] <= split.validation_end)]
    test = df[(df["date"] >= split.test_start) & (df["date"] <= split.test_end)]
    return train, val, test


def train_and_evaluate(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    split: ChronologicalSplit,
    estimators: dict[str, Pipeline] | None = None,
) -> pd.DataFrame:
    fail_if_label_columns_in_features(features)
    train, val, test = split_by_dates(frame, split)
    estimators = estimators or model_zoo()
    rows = []
    for name, pipe in estimators.items():
        pipe.fit(train[features], train[target].astype(int))
        for sample_name, sample in (("validation", val), ("test", test)):
            if sample.empty:
                continue
            if hasattr(pipe, "predict_proba"):
                scores = pipe.predict_proba(sample[features])[:, 1]
            else:
                scores = pipe.decision_function(sample[features])
            metrics = rare_event_metrics(sample[target].astype(int).to_numpy(), scores)
            metrics.update({"model": name, "sample": sample_name})
            rows.append(metrics)
    return pd.DataFrame(rows)


def ablation_feature_sets(feature_names: Iterable[str]) -> dict[str, list[str]]:
    names = list(feature_names)
    groups = {
        "market_only": [c for c in names if c.startswith(("return_", "volatility_", "volume_", "gap", "intraday", "distance_", "abnormal_"))],
        "fundamentals_only": [c for c in names if c in {"cash_runway_months", "current_ratio", "working_capital", "cash_debt_ratio", "revenue", "net_income"}],
        "sec_text_only": [c for c in names if any(x in c for x in ("strategic_", "investment_bank", "change_of_control", "committee", "transaction_"))],
        "corporate_structure_only": [c for c in names if any(x in c for x in ("market_cap", "share", "float", "split", "dilution"))],
        "insider_only": [c for c in names if c.startswith(("insider_", "form4_"))],
        "everything": names,
    }
    groups["sec_plus_fundamentals"] = groups["sec_text_only"] + groups["fundamentals_only"]
    groups["sec_plus_market"] = groups["sec_text_only"] + groups["market_only"]
    return {k: v for k, v in groups.items() if v}
