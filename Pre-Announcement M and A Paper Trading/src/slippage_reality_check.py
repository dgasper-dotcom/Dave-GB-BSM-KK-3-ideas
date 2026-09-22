from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .walk_forward_return_backtest import table


BUY_ALIASES = {"buy", "b", "long", "entry"}
SELL_ALIASES = {"sell", "s", "short", "exit"}


def normalize_side(value: object) -> str:
    text = str(value).strip().lower()
    if text in BUY_ALIASES:
        return "buy"
    if text in SELL_ALIASES:
        return "sell"
    raise ValueError(f"Unknown side: {value!r}")


def adverse_slippage_bps(actual_price: pd.Series, benchmark_price: pd.Series, side: pd.Series) -> pd.Series:
    actual = pd.to_numeric(actual_price, errors="coerce")
    benchmark = pd.to_numeric(benchmark_price, errors="coerce")
    normalized = side.map(normalize_side)
    direction = np.where(normalized.eq("sell"), -1.0, 1.0)
    return pd.Series(direction * (actual / benchmark - 1.0) * 10000.0, index=actual.index)


def _first_existing(frame: pd.DataFrame, columns: Iterable[str]) -> str | None:
    for column in columns:
        if column in frame.columns:
            return column
    return None


def _join_panel_benchmarks(fills: pd.DataFrame, panel: pd.DataFrame | None) -> pd.DataFrame:
    if panel is None:
        return fills
    if "ticker" not in fills.columns:
        return fills

    date_col = _first_existing(fills, ["signal_date", "date"])
    if date_col is None:
        return fills

    panel_cols = [
        "ticker",
        "date",
        "close",
        "next_open",
        "exit_close_20d",
        "adv_20d_dollars",
        "intraday_range",
        "order_pct_adv",
        "price",
    ]
    available = [col for col in panel_cols if col in panel.columns]
    right = panel[available].copy()
    right["date"] = pd.to_datetime(right["date"]).dt.normalize()

    left = fills.copy()
    left[date_col] = pd.to_datetime(left[date_col]).dt.normalize()
    joined = left.merge(
        right,
        left_on=["ticker", date_col],
        right_on=["ticker", "date"],
        how="left",
        suffixes=("", "_panel"),
    )
    if date_col != "date" and "date_panel" in joined.columns:
        joined = joined.drop(columns=["date_panel"])
    return joined


def normalize_fill_observations(
    fills: pd.DataFrame,
    panel: pd.DataFrame | None = None,
    entry_benchmark_column: str = "next_open",
    exit_benchmark_column: str = "exit_close_20d",
) -> pd.DataFrame:
    joined = _join_panel_benchmarks(fills, panel)
    observations: list[pd.DataFrame] = []

    if {"actual_price", "benchmark_price"}.issubset(joined.columns):
        generic = joined.copy()
        if "side" not in generic.columns:
            generic["side"] = "buy"
        generic["cost_type"] = generic.get("cost_type", "fill")
        generic["actual_price_used"] = generic["actual_price"]
        generic["benchmark_price_used"] = generic["benchmark_price"]
        observations.append(generic)

    if "actual_entry_price" in joined.columns:
        entry = joined.copy()
        entry["side"] = "buy"
        entry["cost_type"] = "entry"
        entry["actual_price_used"] = entry["actual_entry_price"]
        benchmark_col = _first_existing(entry, ["entry_benchmark_price", entry_benchmark_column, "planned_entry_price"])
        if benchmark_col is not None:
            entry["benchmark_price_used"] = entry[benchmark_col]
            entry["benchmark_source"] = benchmark_col
            observations.append(entry)

    if "actual_exit_price" in joined.columns:
        exit_frame = joined.copy()
        exit_frame["side"] = "sell"
        exit_frame["cost_type"] = "exit"
        exit_frame["actual_price_used"] = exit_frame["actual_exit_price"]
        benchmark_col = _first_existing(exit_frame, ["exit_benchmark_price", exit_benchmark_column])
        if benchmark_col is not None:
            exit_frame["benchmark_price_used"] = exit_frame[benchmark_col]
            exit_frame["benchmark_source"] = benchmark_col
            observations.append(exit_frame)

    if not observations:
        return pd.DataFrame()

    out = pd.concat(observations, ignore_index=True, sort=False)
    out["side"] = out["side"].map(normalize_side)
    out["actual_price_used"] = pd.to_numeric(out["actual_price_used"], errors="coerce")
    out["benchmark_price_used"] = pd.to_numeric(out["benchmark_price_used"], errors="coerce")
    out = out[out["actual_price_used"].gt(0) & out["benchmark_price_used"].gt(0)].copy()
    if out.empty:
        return out
    out["adverse_slippage_bps"] = adverse_slippage_bps(
        out["actual_price_used"], out["benchmark_price_used"], out["side"]
    )
    if "planned_position_dollars" in out.columns:
        out["notional"] = pd.to_numeric(out["planned_position_dollars"], errors="coerce")
    elif "notional" in out.columns:
        out["notional"] = pd.to_numeric(out["notional"], errors="coerce")
    elif "shares" in out.columns:
        out["notional"] = pd.to_numeric(out["shares"], errors="coerce") * out["actual_price_used"]
    else:
        out["notional"] = np.nan
    return out


def weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna() & weights.gt(0)
    if not valid.any():
        return float(values.mean()) if values.notna().any() else np.nan
    return float(np.average(values[valid], weights=weights[valid]))


def summarize_observations(frame: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    group_cols = group_cols or []
    rows = []
    grouped = [((), frame)] if not group_cols else frame.groupby(group_cols, dropna=False)
    for key, group in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        values = group["adverse_slippage_bps"].dropna()
        row = {col: value for col, value in zip(group_cols, key)}
        row.update(
            {
                "observations": int(len(values)),
                "mean_bps": float(values.mean()) if len(values) else np.nan,
                "median_bps": float(values.median()) if len(values) else np.nan,
                "p75_bps": float(values.quantile(0.75)) if len(values) else np.nan,
                "p90_bps": float(values.quantile(0.90)) if len(values) else np.nan,
                "p95_bps": float(values.quantile(0.95)) if len(values) else np.nan,
                "worst_bps": float(values.max()) if len(values) else np.nan,
                "favorable_rate": float(values.lt(0).mean()) if len(values) else np.nan,
                "notional_weighted_mean_bps": weighted_average(group["adverse_slippage_bps"], group["notional"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _price_levels(row: pd.Series, side: str, levels: int = 10) -> list[tuple[float, float]]:
    prefix = "ask" if side == "buy" else "bid"
    out = []
    for idx in range(levels):
        px_col = f"{prefix}_px_{idx:02d}"
        sz_col = f"{prefix}_sz_{idx:02d}"
        if px_col not in row or sz_col not in row:
            continue
        px = pd.to_numeric(row[px_col], errors="coerce")
        sz = pd.to_numeric(row[sz_col], errors="coerce")
        if pd.notna(px) and pd.notna(sz) and px > 0 and sz > 0:
            out.append((float(px), float(sz)))
    return out


def estimate_book_vwap(row: pd.Series, side: str, shares: float, levels: int = 10) -> tuple[float, float]:
    remaining = float(shares)
    if not np.isfinite(remaining) or remaining <= 0:
        return np.nan, 0.0
    notional = 0.0
    filled = 0.0
    for px, size in _price_levels(row, side, levels=levels):
        take = min(remaining, size)
        notional += take * px
        filled += take
        remaining -= take
        if remaining <= 0:
            break
    if filled <= 0:
        return np.nan, 0.0
    return notional / filled, filled / float(shares)


def analyze_quote_observations(quotes: pd.DataFrame) -> pd.DataFrame:
    if quotes.empty:
        return pd.DataFrame()
    out = quotes.copy()
    bid_col = _first_existing(out, ["bid", "best_bid", "bid_px", "bid_px_00"])
    ask_col = _first_existing(out, ["ask", "best_ask", "ask_px", "ask_px_00"])
    if bid_col is None or ask_col is None:
        raise ValueError("Quote data must include bid/ask columns or bid_px_00/ask_px_00.")
    out["bid_used"] = pd.to_numeric(out[bid_col], errors="coerce")
    out["ask_used"] = pd.to_numeric(out[ask_col], errors="coerce")
    out = out[out["bid_used"].gt(0) & out["ask_used"].gt(out["bid_used"])].copy()
    if out.empty:
        return out
    out["mid"] = (out["bid_used"] + out["ask_used"]) / 2.0
    out["spread_bps"] = (out["ask_used"] - out["bid_used"]) / out["mid"] * 10000.0
    out["half_spread_bps"] = out["spread_bps"] / 2.0

    side_col = _first_existing(out, ["side", "order_side"])
    shares_col = _first_existing(out, ["shares", "order_shares", "order_qty"])
    if side_col is not None and shares_col is not None:
        book_prices = []
        fill_rates = []
        costs = []
        for _, row in out.iterrows():
            side = normalize_side(row[side_col])
            shares = pd.to_numeric(row[shares_col], errors="coerce")
            book_price, fill_rate = estimate_book_vwap(row, side, shares)
            book_prices.append(book_price)
            fill_rates.append(fill_rate)
            if pd.notna(book_price):
                costs.append(
                    adverse_slippage_bps(
                        pd.Series([book_price]), pd.Series([row["mid"]]), pd.Series([side])
                    ).iloc[0]
                )
            else:
                costs.append(np.nan)
        out["book_vwap_price"] = book_prices
        out["book_fill_rate"] = fill_rates
        out["book_cost_vs_mid_bps"] = costs
    return out


def make_recommendations(
    fill_obs: pd.DataFrame,
    quote_obs: pd.DataFrame,
    conservative_quantile: float,
) -> dict[str, object]:
    recommendation: dict[str, object] = {
        "method": "insufficient_data",
        "conservative_quantile": conservative_quantile,
        "entry_slippage_bps": None,
        "exit_slippage_bps": None,
        "fixed_cost_bps": 0.0,
        "notes": [],
    }

    if not fill_obs.empty:
        recommendation["method"] = "actual_fill_observations"
        for cost_type, output_key in [("entry", "entry_slippage_bps"), ("exit", "exit_slippage_bps")]:
            subset = fill_obs[fill_obs["cost_type"].eq(cost_type)]["adverse_slippage_bps"].dropna()
            if not subset.empty:
                recommendation[output_key] = max(0.0, float(subset.quantile(conservative_quantile)))
        all_values = fill_obs["adverse_slippage_bps"].dropna()
        recommendation["all_fill_p75_bps"] = float(all_values.quantile(0.75)) if not all_values.empty else np.nan
        recommendation["all_fill_p90_bps"] = float(all_values.quantile(0.90)) if not all_values.empty else np.nan
        recommendation["notes"].append("Positive bps means adverse slippage versus the selected benchmark.")

    if not quote_obs.empty:
        half_spread = quote_obs["half_spread_bps"].dropna()
        marketable_col = "book_cost_vs_mid_bps" if "book_cost_vs_mid_bps" in quote_obs.columns else "half_spread_bps"
        marketable = quote_obs[marketable_col].dropna()
        if recommendation["method"] == "insufficient_data":
            recommendation["method"] = "quote_or_l2_observations"
            recommendation["entry_slippage_bps"] = max(0.0, float(marketable.quantile(conservative_quantile)))
            recommendation["exit_slippage_bps"] = max(0.0, float(half_spread.quantile(conservative_quantile)))
        recommendation["quote_half_spread_p75_bps"] = (
            float(half_spread.quantile(0.75)) if not half_spread.empty else np.nan
        )
        recommendation["quote_half_spread_p90_bps"] = (
            float(half_spread.quantile(0.90)) if not half_spread.empty else np.nan
        )
        if marketable_col == "book_cost_vs_mid_bps":
            recommendation["book_cost_vs_mid_p75_bps"] = (
                float(marketable.quantile(0.75)) if not marketable.empty else np.nan
            )
            recommendation["book_cost_vs_mid_p90_bps"] = (
                float(marketable.quantile(0.90)) if not marketable.empty else np.nan
            )
        recommendation["notes"].append("Quote-derived costs measure spread/book cost, not broker-specific fill quality.")

    return recommendation


def clean_json_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: clean_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json_value(item) for item in value]
    if isinstance(value, (float, np.floating)) and (np.isnan(value) or np.isinf(value)):
        return None
    return value


def write_report(
    out_dir: Path,
    fill_obs: pd.DataFrame,
    quote_obs: pd.DataFrame,
    summary: pd.DataFrame,
    recommendation: dict[str, object],
) -> None:
    lines = [
        "# Real Slippage Reality Check",
        "",
        "This report converts actual fills and/or quote snapshots into bps assumptions that can replace the fixed slippage constants in the paper model.",
        "",
        "## Recommended Cost Inputs",
        "",
        table(
            pd.DataFrame([clean_json_value(recommendation)]),
            ["method", "entry_slippage_bps", "exit_slippage_bps", "fixed_cost_bps", "conservative_quantile"],
        ),
        "",
        "## Fill Summary",
        "",
        table(summary, ["cost_type", "side", "observations", "median_bps", "p75_bps", "p90_bps", "worst_bps", "notional_weighted_mean_bps"]),
        "",
        "## Quote Summary",
        "",
    ]
    if quote_obs.empty:
        lines.append("No quote observations were provided.")
    else:
        quote_summary = pd.DataFrame(
            [
                {
                    "observations": len(quote_obs),
                    "half_spread_p50_bps": quote_obs["half_spread_bps"].median(),
                    "half_spread_p75_bps": quote_obs["half_spread_bps"].quantile(0.75),
                    "half_spread_p90_bps": quote_obs["half_spread_bps"].quantile(0.90),
                    "book_cost_p75_bps": quote_obs["book_cost_vs_mid_bps"].quantile(0.75)
                    if "book_cost_vs_mid_bps" in quote_obs.columns
                    else np.nan,
                }
            ]
        )
        lines.append(table(quote_summary, list(quote_summary.columns)))
    lines.extend(
        [
            "",
            "## Data Caveats",
            "",
            "- Actual broker fills are the highest-quality slippage data because they include routing, order type, price improvement, partial fills, and rejects.",
            "- NBBO/L2 quote snapshots estimate available liquidity and spread cost, but they do not prove where a broker would have routed or filled an order.",
            "- SEC Rule 605 data is useful for public execution-quality benchmarking by broker/venue, but it is monthly aggregate data rather than our exact order-level slippage.",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def run(
    fills_path: Path | None,
    quotes_path: Path | None,
    panel_path: Path | None,
    out_dir: Path,
    conservative_quantile: float,
    entry_benchmark_column: str,
    exit_benchmark_column: str,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(panel_path, parse_dates=["date"]) if panel_path else None

    if fills_path:
        fills = pd.read_csv(fills_path)
        fill_obs = normalize_fill_observations(
            fills,
            panel=panel,
            entry_benchmark_column=entry_benchmark_column,
            exit_benchmark_column=exit_benchmark_column,
        )
    else:
        fill_obs = pd.DataFrame()

    if quotes_path:
        quotes = pd.read_csv(quotes_path)
        quote_obs = analyze_quote_observations(quotes)
    else:
        quote_obs = pd.DataFrame()

    summary = summarize_observations(fill_obs, ["cost_type", "side"]) if not fill_obs.empty else pd.DataFrame()
    recommendation = make_recommendations(fill_obs, quote_obs, conservative_quantile)

    if not fill_obs.empty:
        fill_obs.to_csv(out_dir / "slippage_observations.csv", index=False)
    if not quote_obs.empty:
        quote_obs.to_csv(out_dir / "quote_spread_observations.csv", index=False)
    if not summary.empty:
        summary.to_csv(out_dir / "slippage_summary.csv", index=False)
    (out_dir / "slippage_recommendations.json").write_text(json.dumps(clean_json_value(recommendation), indent=2))
    write_report(out_dir, fill_obs, quote_obs, summary, recommendation)
    return recommendation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fills", type=Path, help="CSV of actual broker/paper fills or paper_trade_journal rows.")
    parser.add_argument("--quotes", type=Path, help="CSV of bid/ask or L2 quote snapshots.")
    parser.add_argument("--panel", type=Path, help="Optional panel/latest_scores CSV used for entry/exit benchmarks.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/slippage_reality_check"))
    parser.add_argument("--conservative-quantile", type=float, default=0.75)
    parser.add_argument("--entry-benchmark-column", default="next_open")
    parser.add_argument("--exit-benchmark-column", default="exit_close_20d")
    args = parser.parse_args()
    result = run(
        fills_path=args.fills,
        quotes_path=args.quotes,
        panel_path=args.panel,
        out_dir=args.out_dir,
        conservative_quantile=args.conservative_quantile,
        entry_benchmark_column=args.entry_benchmark_column,
        exit_benchmark_column=args.exit_benchmark_column,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
