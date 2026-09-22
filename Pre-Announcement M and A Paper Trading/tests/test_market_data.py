import pandas as pd
import pytest

from src.market_data import market_features


def test_market_features_use_past_and_current_only():
    dates = pd.date_range("2026-01-01", periods=25, freq="B")
    prices = pd.DataFrame(
        {
            "ticker": ["AAA"] * len(dates),
            "date": dates,
            "open": range(10, 10 + len(dates)),
            "high": range(11, 11 + len(dates)),
            "low": range(9, 9 + len(dates)),
            "close": range(10, 10 + len(dates)),
            "volume": [1000] * len(dates),
        }
    )
    out = market_features(prices)
    assert pd.isna(out.loc[0, "return_1d"])
    assert out.loc[1, "return_1d"] == pytest.approx(0.1)
    assert "adv_20d_dollars" in out.columns
