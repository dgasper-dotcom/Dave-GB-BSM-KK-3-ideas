from __future__ import annotations

import pandas as pd


def build_trading_calendar(calendar: pd.DataFrame | pd.Series) -> pd.Series:
    if isinstance(calendar, pd.DataFrame):
        values = calendar["date"]
    else:
        values = calendar
    return pd.Series(pd.to_datetime(values)).dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)


def nth_future_session(date: pd.Timestamp, calendar: pd.Series, n: int) -> pd.Timestamp:
    d = pd.Timestamp(date).normalize()
    loc = calendar.searchsorted(d, side="right")
    idx = loc + n - 1
    if idx >= len(calendar):
        return pd.NaT
    return pd.Timestamp(calendar.iloc[idx])


def make_event_labels(
    observations: pd.DataFrame,
    events: pd.DataFrame,
    calendar: pd.DataFrame | pd.Series,
    horizons: tuple[int, ...] = (5, 10, 20, 40, 60, 90),
) -> pd.DataFrame:
    cal = build_trading_calendar(calendar)
    obs = observations.copy()
    obs["date"] = pd.to_datetime(obs["date"]).dt.normalize()
    ev = events.copy()
    ev["announcement_ts"] = pd.to_datetime(ev["announcement_ts"], utc=True)
    ev["announcement_date"] = ev["announcement_ts"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    ev["cik"] = ev["cik"].astype(str).str.zfill(10)
    obs["cik"] = obs["cik"].astype(str).str.zfill(10)

    out = obs.copy()
    out["next_event_date"] = pd.NaT
    out["next_event_type"] = None
    for h in horizons:
        out[f"event_{h}d"] = 0

    for idx, row in out.iterrows():
        company_events = ev[(ev["cik"] == row["cik"]) & (ev["announcement_date"] > row["date"])]
        if company_events.empty:
            continue
        first = company_events.sort_values("announcement_date").iloc[0]
        out.at[idx, "next_event_date"] = first["announcement_date"]
        out.at[idx, "next_event_type"] = first["event_type"]
        sessions_to_event = int(((cal > row["date"]) & (cal <= first["announcement_date"])).sum())
        for h in horizons:
            if 0 < sessions_to_event <= h:
                out.at[idx, f"event_{h}d"] = 1
    out["time_to_event_trading_days"] = out.apply(
        lambda r: int(((cal > r["date"]) & (cal <= r["next_event_date"])).sum())
        if pd.notna(r["next_event_date"])
        else pd.NA,
        axis=1,
    )
    return out
