from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .point_in_time import assert_no_future_information


LOOKBACK_TRADING_DAYS = [90, 60, 40, 20, 10, 5, 1]
SIGNAL_COLUMNS = [
    "strategic_alternatives_hit_90d",
    "investment_bank_hit_90d",
    "change_of_control_hit_90d",
    "committee_hit_90d",
    "financing_hit_90d",
    "cash_runway_months",
    "market_cap",
    "volume_to_20d_avg",
]


def pre_event_dates(event_date: pd.Timestamp, calendar: pd.Series) -> dict[int, pd.Timestamp]:
    cal = pd.Series(pd.to_datetime(calendar)).dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    idx = cal.searchsorted(pd.Timestamp(event_date).normalize())
    out = {}
    for n in LOOKBACK_TRADING_DAYS:
        target_idx = idx - n
        if target_idx >= 0:
            out[n] = pd.Timestamp(cal.iloc[target_idx])
    return out


def build_case_study(features: pd.DataFrame, events: pd.DataFrame, calendar: pd.DataFrame | pd.Series) -> str:
    cal = calendar["date"] if isinstance(calendar, pd.DataFrame) else calendar
    ev = events.copy()
    ev["announcement_ts"] = pd.to_datetime(ev["announcement_ts"], utc=True)
    aemd_events = ev[ev["ticker"].str.upper().eq("AEMD")].sort_values("announcement_ts")
    if aemd_events.empty:
        raise ValueError("No AEMD event found in events table")
    event = aemd_events.iloc[0]
    event_date = event["announcement_ts"].tz_convert("America/New_York").normalize().tz_localize(None)
    dates = pre_event_dates(event_date, pd.Series(cal))

    frame = features.copy()
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["prediction_ts"] = pd.to_datetime(frame["prediction_ts"], utc=True)
    if "feature_timestamp" in frame.columns:
        assert_no_future_information(frame, timestamp_cols=("feature_timestamp",))

    lines = [
        "# AEMD Point-in-Time Case Study",
        "",
        f"Event type: {event.get('event_type', 'UNKNOWN')}",
        f"First public announcement timestamp: {event['announcement_ts']}",
        f"Source: {event.get('announcement_source', '')} {event.get('announcement_url', '')}",
        "",
        "This report shows only features available by each prediction timestamp.",
        "",
    ]
    available_cols = [c for c in SIGNAL_COLUMNS if c in frame.columns]
    for lookback, d in dates.items():
        row = frame[(frame["ticker"].str.upper() == "AEMD") & (frame["date"] == d)]
        lines.append(f"## {lookback} Trading Days Before ({d.date()})")
        if row.empty:
            lines.append("")
            lines.append("No feature row available.")
            lines.append("")
            continue
        r = row.iloc[0]
        lines.append("")
        lines.append(f"Prediction timestamp: {r['prediction_ts']}")
        if "feature_timestamp" in row.columns:
            lines.append(f"Latest feature timestamp: {r.get('feature_timestamp')}")
        lines.append("")
        for col in available_cols:
            lines.append(f"- `{col}`: {r.get(col)}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--calendar", required=True)
    parser.add_argument("--out", default="reports/aemd_case_study.md")
    args = parser.parse_args()
    features = pd.read_parquet(args.features)
    events = pd.read_csv(args.events)
    calendar = pd.read_csv(args.calendar)
    report = build_case_study(features, events, calendar)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(out)


if __name__ == "__main__":
    main()
