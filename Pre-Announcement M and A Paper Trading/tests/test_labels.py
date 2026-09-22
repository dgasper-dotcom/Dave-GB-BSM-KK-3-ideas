import pandas as pd

from src.labels import make_event_labels
from src.point_in_time import prediction_timestamp


def test_event_label_next_twenty_sessions():
    calendar = pd.date_range("2026-01-01", periods=40, freq="B")
    observations = pd.DataFrame(
        [
            {"cik": "1", "ticker": "AAA", "date": calendar[0], "prediction_ts": prediction_timestamp(calendar[0])},
            {"cik": "1", "ticker": "AAA", "date": calendar[20], "prediction_ts": prediction_timestamp(calendar[20])},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "cik": "1",
                "ticker": "AAA",
                "event_type": "MERGER",
                "announcement_ts": pd.Timestamp(calendar[10]).tz_localize("America/New_York") + pd.Timedelta(hours=8),
            }
        ]
    )
    labeled = make_event_labels(observations, events, calendar, horizons=(5, 20))
    assert labeled.loc[0, "event_5d"] == 0
    assert labeled.loc[0, "event_20d"] == 1
    assert labeled.loc[1, "event_20d"] == 0


def test_event_label_when_calendar_ends_on_event_date():
    calendar = pd.date_range("2026-08-20", "2026-09-17", freq="B")
    observations = pd.DataFrame(
        [
            {"cik": "1", "ticker": "AAA", "date": calendar[0], "prediction_ts": prediction_timestamp(calendar[0])},
            {"cik": "1", "ticker": "AAA", "date": calendar[-2], "prediction_ts": prediction_timestamp(calendar[-2])},
        ]
    )
    events = pd.DataFrame(
        [
            {
                "cik": "1",
                "ticker": "AAA",
                "event_type": "MERGER",
                "announcement_ts": pd.Timestamp(calendar[-1]).tz_localize("America/New_York") + pd.Timedelta(hours=8),
            }
        ]
    )
    labeled = make_event_labels(observations, events, calendar, horizons=(20,))
    assert labeled.loc[0, "event_20d"] == 1
    assert labeled.loc[1, "event_20d"] == 1
