from __future__ import annotations

import pandas as pd


WINDOWS = {
    "ret_m60_m30": (-60, -30),
    "ret_m30_m20": (-30, -20),
    "ret_m20_m10": (-20, -10),
    "ret_m10_m5": (-10, -5),
    "ret_m5_m1": (-5, -1),
    "ret_announcement_day": (0, 0),
    "ret_p1": (1, 1),
    "ret_p5": (1, 5),
    "ret_p10": (1, 10),
    "ret_p20": (1, 20),
}


def event_study_returns(events: pd.DataFrame, prices: pd.DataFrame, calendar: pd.Series) -> pd.DataFrame:
    cal = pd.Series(pd.to_datetime(calendar)).dt.normalize().drop_duplicates().sort_values().reset_index(drop=True)
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"]).dt.normalize()
    px = px.set_index(["ticker", "date"]).sort_index()
    rows = []
    for event in events.to_dict("records"):
        ticker = event["ticker"]
        event_date = pd.Timestamp(event["announcement_ts"]).tz_convert("America/New_York").normalize().tz_localize(None)
        idx = cal.searchsorted(event_date)
        row = {"ticker": ticker, "event_date": event_date, "event_type": event.get("event_type")}
        for name, (start_offset, end_offset) in WINDOWS.items():
            start_idx = idx + start_offset
            end_idx = idx + end_offset
            if start_idx < 0 or end_idx >= len(cal):
                row[name] = pd.NA
                continue
            start_date = cal.iloc[start_idx]
            end_date = cal.iloc[end_idx]
            try:
                start_close = px.loc[(ticker, start_date), "close"]
                end_close = px.loc[(ticker, end_date), "close"]
                row[name] = end_close / start_close - 1
            except KeyError:
                row[name] = pd.NA
        rows.append(row)
    return pd.DataFrame(rows)
