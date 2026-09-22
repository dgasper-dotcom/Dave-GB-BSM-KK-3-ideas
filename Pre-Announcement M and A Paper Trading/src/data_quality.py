from __future__ import annotations

import pandas as pd

from .point_in_time import assert_no_future_information


def quality_report(frame: pd.DataFrame) -> dict[str, object]:
    report: dict[str, object] = {
        "rows": int(len(frame)),
        "date_min": str(pd.to_datetime(frame["date"]).min().date()) if "date" in frame and len(frame) else None,
        "date_max": str(pd.to_datetime(frame["date"]).max().date()) if "date" in frame and len(frame) else None,
        "duplicate_observations": 0,
        "missing_prediction_ts": None,
    }
    if {"cik", "date"}.issubset(frame.columns):
        report["duplicate_observations"] = int(frame.duplicated(["cik", "date"]).sum())
    if "prediction_ts" in frame.columns:
        report["missing_prediction_ts"] = int(frame["prediction_ts"].isna().sum())
    return report


def run_quality_checks(frame: pd.DataFrame, timestamp_cols: tuple[str, ...] = ("feature_timestamp",)) -> None:
    if {"cik", "date"}.issubset(frame.columns) and frame.duplicated(["cik", "date"]).any():
        raise AssertionError("Duplicate company-date observations detected")
    if "prediction_ts" in frame.columns:
        assert_no_future_information(frame, timestamp_cols=tuple(c for c in timestamp_cols if c in frame.columns))
