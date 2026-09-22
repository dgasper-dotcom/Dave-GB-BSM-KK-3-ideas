from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import rare_event_metrics
from .labels import make_event_labels
from .leakage_tests import fail_if_label_columns_in_features
from .point_in_time import assert_no_future_information


EXCLUDE_SUBSTRINGS = (
    "timestamp",
    "_ts",
    "date",
    "event",
    "label",
    "target",
    "announcement",
    "url",
    "source",
    "company",
    "ticker",
    "cik",
)


def load_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        try:
            frame.to_parquet(path, index=False)
            return
        except Exception:
            path = path.with_suffix(".csv")
    frame.to_csv(path, index=False)


def select_numeric_features(frame: pd.DataFrame, target: str) -> list[str]:
    candidates = []
    for col in frame.columns:
        low = col.lower()
        if col == target or any(x in low for x in EXCLUDE_SUBSTRINGS):
            continue
        numeric = pd.to_numeric(frame[col], errors="coerce")
        if numeric.notna().sum() == 0:
            continue
        if numeric.nunique(dropna=True) <= 1:
            continue
        frame[col] = numeric
        candidates.append(col)
    fail_if_label_columns_in_features(candidates)
    return candidates


def build_estimator(name: str) -> Pipeline:
    if name == "logistic":
        model = LogisticRegression(max_iter=2000, class_weight="balanced")
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", model),
            ]
        )
    if name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=42,
            n_jobs=-1,
        )
        return Pipeline([("impute", SimpleImputer(strategy="median")), ("model", model)])
    raise ValueError(f"Unknown model: {name}")


def chronological_split(frame: pd.DataFrame, train_end: str | None, test_start: str | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = frame.copy()
    df["date"] = pd.to_datetime(df["date"])
    if train_end and test_start:
        train = df[df["date"] <= pd.Timestamp(train_end)]
        test = df[df["date"] >= pd.Timestamp(test_start)]
        return train, test
    cutoff = df["date"].quantile(0.8)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def train(
    features_path: Path,
    events_path: Path,
    calendar_path: Path,
    out_dir: Path,
    horizon: int,
    model_name: str,
    train_end: str | None,
    test_start: str | None,
    allow_smoke: bool,
) -> dict[str, object]:
    features = load_table(features_path)
    events = load_table(events_path)
    calendar = load_table(calendar_path)

    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    if "prediction_ts" in features.columns:
        features["prediction_ts"] = pd.to_datetime(features["prediction_ts"], utc=True)
    if "feature_timestamp" in features.columns and "prediction_ts" in features.columns:
        assert_no_future_information(features, timestamp_cols=("feature_timestamp",))

    target = f"event_{horizon}d"
    if target not in features.columns:
        labeled = make_event_labels(features, events, calendar, horizons=(horizon,))
    else:
        labeled = features.copy()
    labeled[target] = labeled[target].fillna(0).astype(int)

    feature_names = select_numeric_features(labeled, target)
    if not feature_names:
        raise ValueError("No usable numeric features found.")

    train_frame, test_frame = chronological_split(labeled, train_end, test_start)
    mode = "chronological"
    if train_frame[target].nunique() < 2:
        if not allow_smoke:
            raise ValueError(
                f"Training split has one class only for {target}. "
                "Add more companies/events or rerun with --allow-smoke for a non-evidentiary in-sample smoke model."
            )
        mode = "single_company_in_sample_smoke"
        train_frame = labeled.copy()
        test_frame = labeled.copy()

    estimator = build_estimator(model_name)
    estimator.fit(train_frame[feature_names], train_frame[target])
    scores = estimator.predict_proba(test_frame[feature_names])[:, 1]
    metrics = rare_event_metrics(test_frame[target].to_numpy(), scores)
    metrics.update(
        {
            "mode": mode,
            "model": model_name,
            "target": target,
            "feature_count": len(feature_names),
            "train_rows": int(len(train_frame)),
            "train_events": int(train_frame[target].sum()),
            "test_rows": int(len(test_frame)),
            "test_events": int(test_frame[target].sum()),
            "unique_companies": int(labeled["cik"].nunique()) if "cik" in labeled.columns else None,
            "date_min": str(labeled["date"].min().date()),
            "date_max": str(labeled["date"].max().date()),
        }
    )
    if len(np.unique(test_frame[target])) > 1:
        metrics["roc_auc"] = roc_auc_score(test_frame[target], scores)
        metrics["pr_auc"] = average_precision_score(test_frame[target], scores)

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{model_name}_{target}.joblib"
    features_path_out = out_dir / f"{model_name}_{target}_features.json"
    metrics_path = out_dir / f"{model_name}_{target}_metrics.json"
    predictions_path = out_dir / f"{model_name}_{target}_predictions.csv"
    labeled_path = out_dir / f"{model_name}_{target}_labeled_panel.csv"

    metrics["model_path"] = str(model_path)
    metrics["feature_list_path"] = str(features_path_out)
    metrics["metrics_path"] = str(metrics_path)
    metrics["predictions_path"] = str(predictions_path)
    metrics["labeled_panel_path"] = str(labeled_path)

    joblib.dump(estimator, model_path)
    features_path_out.write_text(json.dumps(feature_names, indent=2))

    predictions = test_frame[["cik", "ticker", "date", target]].copy()
    predictions[f"predicted_probability_{horizon}d"] = scores
    predictions["rank"] = predictions.groupby("date")[f"predicted_probability_{horizon}d"].rank(
        ascending=False, method="first"
    )
    predictions.to_csv(predictions_path, index=False)
    save_table(labeled, labeled_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=float))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--out-dir", default="models")
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--model", choices=["logistic", "random_forest"], default="logistic")
    parser.add_argument("--train-end")
    parser.add_argument("--test-start")
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()

    metrics = train(
        features_path=Path(args.features),
        events_path=Path(args.events),
        calendar_path=Path(args.calendar),
        out_dir=Path(args.out_dir),
        horizon=args.horizon,
        model_name=args.model,
        train_end=args.train_end,
        test_start=args.test_start,
        allow_smoke=args.allow_smoke,
    )
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    main()
