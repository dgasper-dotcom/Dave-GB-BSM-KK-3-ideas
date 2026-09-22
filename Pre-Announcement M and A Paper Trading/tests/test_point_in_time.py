import pandas as pd
import pytest

from src.point_in_time import (
    assert_no_future_information,
    filing_available_timestamp,
    prediction_timestamp,
)


def test_after_close_filing_available_next_session():
    calendar = pd.to_datetime(["2026-03-12", "2026-03-13", "2026-03-16"])
    ts = filing_available_timestamp("2026-03-12 17:01:00-04:00", calendar)
    assert ts == prediction_timestamp("2026-03-13")


def test_before_close_filing_available_same_session():
    calendar = pd.to_datetime(["2026-03-12", "2026-03-13"])
    ts = filing_available_timestamp("2026-03-12 15:59:00-04:00", calendar)
    assert ts == prediction_timestamp("2026-03-12")


def test_future_feature_raises():
    frame = pd.DataFrame(
        {
            "prediction_ts": [pd.Timestamp("2026-01-02 21:00:00Z")],
            "feature_timestamp": [pd.Timestamp("2026-01-03 21:00:00Z")],
        }
    )
    with pytest.raises(AssertionError):
        assert_no_future_information(frame)
