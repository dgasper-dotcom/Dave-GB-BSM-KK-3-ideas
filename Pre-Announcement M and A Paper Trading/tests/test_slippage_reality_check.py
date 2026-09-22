import pandas as pd

from src.slippage_reality_check import (
    adverse_slippage_bps,
    analyze_quote_observations,
    make_recommendations,
    normalize_fill_observations,
)


def test_adverse_slippage_bps_handles_buy_and_sell_direction():
    actual = pd.Series([10.10, 9.90])
    benchmark = pd.Series([10.00, 10.00])
    side = pd.Series(["buy", "sell"])

    out = adverse_slippage_bps(actual, benchmark, side)

    assert out.round(6).tolist() == [100.0, 100.0]


def test_normalize_fill_observations_joins_panel_benchmarks():
    fills = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "signal_date": ["2026-01-02"],
            "actual_entry_price": [10.10],
            "actual_exit_price": [10.80],
            "planned_position_dollars": [1000.0],
        }
    )
    panel = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "date": pd.to_datetime(["2026-01-02"]),
            "next_open": [10.00],
            "exit_close_20d": [11.00],
        }
    )

    out = normalize_fill_observations(fills, panel)

    entry = out[out["cost_type"].eq("entry")].iloc[0]
    exit_row = out[out["cost_type"].eq("exit")].iloc[0]
    assert round(entry["adverse_slippage_bps"], 6) == 100.0
    assert round(exit_row["adverse_slippage_bps"], 6) == round((1 - 10.80 / 11.00) * 10000, 6)


def test_analyze_quote_observations_computes_spread_and_book_cost():
    quotes = pd.DataFrame(
        {
            "ticker": ["ABC"],
            "bid_px_00": [9.98],
            "bid_sz_00": [1000],
            "ask_px_00": [10.02],
            "ask_sz_00": [100],
            "ask_px_01": [10.04],
            "ask_sz_01": [100],
            "side": ["buy"],
            "order_shares": [150],
        }
    )

    out = analyze_quote_observations(quotes)

    assert round(out["half_spread_bps"].iloc[0], 6) == 20.0
    assert round(out["book_fill_rate"].iloc[0], 6) == 1.0
    assert round(out["book_cost_vs_mid_bps"].iloc[0], 6) == round(((100 * 10.02 + 50 * 10.04) / 150 / 10 - 1) * 10000, 6)


def test_make_recommendations_prefers_actual_fills():
    fills = pd.DataFrame(
        {
            "cost_type": ["entry", "entry", "exit", "exit"],
            "adverse_slippage_bps": [10.0, 30.0, 5.0, 15.0],
        }
    )
    quotes = pd.DataFrame({"half_spread_bps": [100.0, 200.0]})

    rec = make_recommendations(fills, quotes, conservative_quantile=0.75)

    assert rec["method"] == "actual_fill_observations"
    assert rec["entry_slippage_bps"] == 25.0
    assert rec["exit_slippage_bps"] == 12.5
