import pandas as pd
import pytest

from src.two_sleeve_return_strategy import summarize_portfolio


def trade_frame(rows):
    return pd.DataFrame(
        {
            "date": pd.to_datetime([row[0] for row in rows]),
            "ticker": [row[1] for row in rows],
            "net_return": [row[2] for row in rows],
            "event_20d": [row[3] for row in rows],
            "exit_kind": [row[4] for row in rows],
        }
    )


def test_summarize_portfolio_equal_weights_idle_sleeves_as_cash():
    sleeve_a = trade_frame(
        [
            ("2026-01-02", "AAA", 0.10, 0, "time_stop_20d"),
            ("2026-02-02", "BBB", -0.02, 0, "time_stop_20d"),
        ]
    )
    sleeve_b = trade_frame(
        [
            ("2026-01-02", "CCC", 0.20, 1, "announcement_peak_after_close"),
        ]
    )

    summary, periods = summarize_portfolio({"a": sleeve_a, "b": sleeve_b})

    assert summary["trades"] == 3
    assert summary["tickers"] == 3
    assert summary["periods"] == 2
    assert summary["total_return"] == pytest.approx((1.15 * 0.99) - 1)
    assert summary["max_drawdown"] == pytest.approx(-0.01)
    assert periods["portfolio_return"].tolist() == pytest.approx([0.15, -0.01])
