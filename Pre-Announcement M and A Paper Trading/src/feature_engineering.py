from __future__ import annotations

import pandas as pd

from .point_in_time import assert_no_future_information, prediction_timestamp


def make_observation_grid(universe: pd.DataFrame, calendar: pd.DataFrame | pd.Series) -> pd.DataFrame:
    dates = calendar["date"] if isinstance(calendar, pd.DataFrame) else calendar
    dates = pd.to_datetime(dates).dt.normalize().drop_duplicates().sort_values()
    rows = []
    for company in universe.to_dict("records"):
        start = pd.Timestamp(company.get("start_date", dates.min())).normalize()
        end = pd.Timestamp(company.get("end_date", dates.max())).normalize()
        active_dates = dates[(dates >= start) & (dates <= end)]
        for d in active_dates:
            rows.append(
                {
                    "cik": str(company["cik"]).zfill(10),
                    "ticker": company["ticker"],
                    "company": company.get("company"),
                    "exchange": company.get("exchange"),
                    "date": d,
                    "prediction_ts": prediction_timestamp(d),
                }
            )
    return pd.DataFrame(rows)


def combine_features(observations: pd.DataFrame, *feature_frames: pd.DataFrame) -> pd.DataFrame:
    out = observations.copy()
    keys = ["cik", "ticker", "date", "prediction_ts"]
    ts_cols = []
    for i, frame in enumerate(feature_frames):
        if frame is None or frame.empty:
            continue
        f = frame.copy()
        renamed_ts = f"feature_timestamp_{i}"
        if "feature_timestamp" in f.columns:
            f = f.rename(columns={"feature_timestamp": renamed_ts})
            ts_cols.append(renamed_ts)
        join_keys = [k for k in keys if k in f.columns and k in out.columns]
        out = out.merge(f, on=join_keys, how="left", suffixes=("", f"_{i}"))
    if ts_cols:
        check = out[["prediction_ts", *ts_cols]].copy()
        assert_no_future_information(check, timestamp_cols=tuple(ts_cols))
        out["feature_timestamp"] = pd.to_datetime(out[ts_cols].stack(), utc=True).groupby(level=0).max()
    return out
