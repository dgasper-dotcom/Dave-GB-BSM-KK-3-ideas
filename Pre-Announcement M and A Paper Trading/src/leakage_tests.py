from __future__ import annotations

import pandas as pd

from .point_in_time import assert_no_future_information


def fail_if_future_features(frame: pd.DataFrame) -> None:
    timestamp_cols = tuple(c for c in frame.columns if c.endswith("_timestamp") and c != "prediction_ts")
    if timestamp_cols:
        assert_no_future_information(frame, timestamp_cols=timestamp_cols)


def fail_if_label_columns_in_features(feature_names: list[str]) -> None:
    forbidden = ("event_", "next_event", "announcement", "label", "target")
    bad = [name for name in feature_names if name.startswith(forbidden) or any(x in name for x in ("future", "post_"))]
    if bad:
        raise AssertionError(f"Leak-prone feature names detected: {bad}")
