from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from typing import Iterable

import pandas as pd


MARKET_CLOSE_ET = time(16, 0)


@dataclass(frozen=True)
class TimestampPolicy:
    """Policy for converting source timestamps into tradable feature timestamps."""

    after_close_available_next_session: bool = True
    market_close_time: time = MARKET_CLOSE_ET


def to_utc_timestamp(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def prediction_timestamp(date: object, tz: str = "America/New_York") -> pd.Timestamp:
    ts = pd.Timestamp(date)
    if ts.tzinfo is None:
        local_date = ts.normalize().tz_localize(tz)
    else:
        local_date = ts.tz_convert(tz).normalize()
    local = local_date + pd.Timedelta(hours=16)
    return local.tz_convert("UTC")


def next_trading_session(date: object, calendar: Iterable[object]) -> pd.Timestamp:
    d = pd.Timestamp(date)
    if d.tzinfo is not None:
        d = d.tz_convert("UTC").tz_localize(None)
    d = d.normalize()
    dates = pd.Series(pd.to_datetime(list(calendar))).dt.normalize().drop_duplicates().sort_values()
    future = dates[dates > d]
    if future.empty:
        raise ValueError(f"No trading session after {d.date()}")
    return pd.Timestamp(future.iloc[0])


def filing_available_timestamp(
    acceptance_ts: object,
    calendar: Iterable[object],
    tz: str = "America/New_York",
    policy: TimestampPolicy = TimestampPolicy(),
) -> pd.Timestamp:
    """Return the earliest prediction timestamp that may use a filing."""

    accepted = pd.Timestamp(acceptance_ts)
    if accepted.tzinfo is None:
        accepted = accepted.tz_localize(tz)
    else:
        accepted = accepted.tz_convert(tz)

    local_date = accepted.normalize()
    if (
        policy.after_close_available_next_session
        and accepted.timetz().replace(tzinfo=None) > policy.market_close_time
    ):
        available_date = next_trading_session(local_date, calendar)
    else:
        available_date = local_date
    return prediction_timestamp(available_date, tz=tz)


def assert_no_future_information(
    frame: pd.DataFrame,
    prediction_col: str = "prediction_ts",
    timestamp_cols: tuple[str, ...] = ("feature_timestamp",),
) -> None:
    missing = [c for c in (prediction_col, *timestamp_cols) if c not in frame.columns]
    if missing:
        raise AssertionError(f"Missing timestamp columns: {missing}")

    pred = pd.to_datetime(frame[prediction_col], utc=True)
    violations: dict[str, int] = {}
    examples = []
    for col in timestamp_cols:
        src = pd.to_datetime(frame[col], utc=True)
        mask = src.notna() & pred.notna() & (src > pred)
        if mask.any():
            violations[col] = int(mask.sum())
            examples.append(frame.loc[mask, [prediction_col, col]].head(3))
    if violations:
        raise AssertionError(f"Future information detected: {violations}; examples={examples}")


def merge_asof_point_in_time(
    observations: pd.DataFrame,
    sources: pd.DataFrame,
    by: list[str],
    obs_ts_col: str = "prediction_ts",
    source_ts_col: str = "information_available_timestamp",
    suffix: str = "_src",
) -> pd.DataFrame:
    """Backward as-of merge that only joins source rows available by the prediction timestamp."""

    left = observations.copy()
    right = sources.copy()
    left[obs_ts_col] = pd.to_datetime(left[obs_ts_col], utc=True)
    right[source_ts_col] = pd.to_datetime(right[source_ts_col], utc=True)
    left = left.sort_values(by + [obs_ts_col])
    right = right.sort_values(by + [source_ts_col])
    merged = pd.merge_asof(
        left,
        right,
        left_on=obs_ts_col,
        right_on=source_ts_col,
        by=by,
        direction="backward",
        suffixes=("", suffix),
    )
    if source_ts_col in merged.columns:
        assert_no_future_information(
            merged.rename(columns={obs_ts_col: "prediction_ts", source_ts_col: "feature_timestamp"}),
            timestamp_cols=("feature_timestamp",),
        )
    return merged
