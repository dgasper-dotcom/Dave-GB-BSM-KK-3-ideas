from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .early_signal_strategy import StrategySpec, backtest_spec, load_events, load_predictions


@dataclass(frozen=True)
class SleeveSpec:
    name: str
    model: str
    threshold_quantile: float
    top_k: int
    max_hold: int
    filter_name: str
    thesis: str


def default_sleeves(tune_frames: dict[str, pd.DataFrame]) -> list[tuple[SleeveSpec, StrategySpec]]:
    specs = [
        SleeveSpec(
            name="unpriced_event_probability",
            model="hgb",
            threshold_quantile=0.98,
            top_k=10,
            max_hold=20,
            filter_name="away_high",
            thesis="High M&A score, but stock remains at least 5% below its 52-week high.",
        ),
        SleeveSpec(
            name="stale_sec_repricing",
            model="hgb",
            threshold_quantile=0.95,
            top_k=10,
            max_hold=20,
            filter_name="stale_sec_not_recent",
            thesis="M&A language appeared in the last 90 days, but not the last 30, and no large run-up has occurred.",
        ),
    ]
    out: list[tuple[SleeveSpec, StrategySpec]] = []
    for sleeve in specs:
        threshold = float(tune_frames[sleeve.model]["score"].quantile(sleeve.threshold_quantile))
        out.append(
            (
                sleeve,
                StrategySpec(
                    mode="nonoverlap",
                    model=sleeve.model,
                    threshold_quantile=sleeve.threshold_quantile,
                    threshold=threshold,
                    top_k=sleeve.top_k,
                    max_hold=sleeve.max_hold,
                    filter_name=sleeve.filter_name,
                ),
            )
        )
    return out


def period_returns(trades: pd.DataFrame) -> pd.Series:
    if trades.empty:
        return pd.Series(dtype=float)
    return trades.groupby("date")["net_return"].mean().sort_index()


def summarize_portfolio(trades_by_sleeve: dict[str, pd.DataFrame]) -> tuple[dict[str, object], pd.DataFrame]:
    sleeve_returns = {name: period_returns(trades) for name, trades in trades_by_sleeve.items()}
    dates = sorted(set().union(*[set(series.index) for series in sleeve_returns.values()]))
    if not dates:
        return {
            "sleeves": len(trades_by_sleeve),
            "periods": 0,
            "trades": 0,
            "tickers": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "mean_period_return": 0.0,
            "median_period_return": 0.0,
            "positive_period_rate": np.nan,
            "mean_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "mean_ex_best": np.nan,
            "event_trade_rate": np.nan,
            "announcement_exit_rate": np.nan,
        }, pd.DataFrame()

    matrix = pd.DataFrame(index=pd.DatetimeIndex(dates, name="date"))
    for name, returns in sleeve_returns.items():
        matrix[name] = returns.reindex(matrix.index).fillna(0.0)
    matrix["portfolio_return"] = matrix[list(sleeve_returns)].mean(axis=1)
    matrix["equity"] = (1 + matrix["portfolio_return"]).cumprod()
    matrix["drawdown"] = matrix["equity"] / matrix["equity"].cummax() - 1

    all_trades = pd.concat(
        [trades.assign(sleeve=name) for name, trades in trades_by_sleeve.items() if not trades.empty],
        ignore_index=True,
    )
    summary: dict[str, object] = {
        "sleeves": len(trades_by_sleeve),
        "periods": int(len(matrix)),
        "trades": int(len(all_trades)),
        "tickers": int(all_trades["ticker"].nunique()) if not all_trades.empty else 0,
        "total_return": float(matrix["equity"].iloc[-1] - 1),
        "max_drawdown": float(matrix["drawdown"].min()),
        "mean_period_return": float(matrix["portfolio_return"].mean()),
        "median_period_return": float(matrix["portfolio_return"].median()),
        "positive_period_rate": float(matrix["portfolio_return"].gt(0).mean()),
        "mean_trade_return": float(all_trades["net_return"].mean()) if not all_trades.empty else np.nan,
        "median_trade_return": float(all_trades["net_return"].median()) if not all_trades.empty else np.nan,
        "win_rate": float(all_trades["net_return"].gt(0).mean()) if not all_trades.empty else np.nan,
        "event_trade_rate": float(all_trades["event_20d"].fillna(0).eq(1).mean()) if not all_trades.empty else np.nan,
        "announcement_exit_rate": (
            float(all_trades["exit_kind"].astype(str).str.startswith("announcement_peak").mean())
            if not all_trades.empty
            else np.nan
        ),
    }
    if len(all_trades) > 1:
        trimmed = all_trades.drop(index=all_trades["net_return"].idxmax())
        summary["mean_ex_best"] = float(trimmed["net_return"].mean())
    else:
        summary["mean_ex_best"] = np.nan
    return summary, matrix.reset_index()


def prefixed(summary: dict[str, object], prefix: str) -> dict[str, object]:
    keys = [
        "periods",
        "trades",
        "tickers",
        "total_return",
        "max_drawdown",
        "mean_period_return",
        "median_period_return",
        "positive_period_rate",
        "mean_trade_return",
        "median_trade_return",
        "win_rate",
        "mean_ex_best",
        "event_trade_rate",
        "announcement_exit_rate",
    ]
    return {f"{prefix}_{key}": summary.get(key) for key in keys}


def report_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame.empty:
        return "_No rows._"
    return frame[[col for col in columns if col in frame.columns]].to_markdown(index=False, floatfmt=".4f")


def load_baseline(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "robust_strategy_selection" / "selected_deployable_strategy.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def run(
    run_dir: Path,
    events_path: Path,
    price_dirs: list[Path],
    out_dir: Path,
    cost_bps: float,
    tune_end: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}
    validation = {model: load_predictions(run_dir, model, "validation") for model in ["hgb"]}
    test = {model: load_predictions(run_dir, model, "test") for model in ["hgb"]}
    tune_cutoff = pd.Timestamp(tune_end)
    tune_frames = {model: df[df["date"].lt(tune_cutoff)].copy() for model, df in validation.items()}
    sleeves = default_sleeves(tune_frames)

    price_cache: dict[str, pd.DataFrame] = {}
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] = {}
    sleeve_rows = []
    portfolio_rows = []
    all_trades = []
    all_period_returns = []

    samples = {
        "tune": {model: df[df["date"].lt(tune_cutoff)].copy() for model, df in validation.items()},
        "confirm": {model: df[df["date"].ge(tune_cutoff)].copy() for model, df in validation.items()},
        "test": test,
    }
    portfolio_summaries: dict[str, dict[str, object]] = {}

    for sample_name, frames in samples.items():
        trades_by_sleeve: dict[str, pd.DataFrame] = {}
        for sleeve, spec in sleeves:
            summary, trades = backtest_spec(
                frames[spec.model],
                spec,
                events_by_ticker,
                price_dirs,
                cost_bps,
                price_cache=price_cache,
                exit_cache=exit_cache,
            )
            sleeve_rows.append(
                {
                    "sample": sample_name,
                    "sleeve": sleeve.name,
                    "thesis": sleeve.thesis,
                    **summary,
                }
            )
            tagged_trades = trades.assign(sample=sample_name, sleeve=sleeve.name)
            trades_by_sleeve[sleeve.name] = tagged_trades
            if not tagged_trades.empty:
                all_trades.append(tagged_trades)

        portfolio_summary, portfolio_periods = summarize_portfolio(trades_by_sleeve)
        portfolio_summaries[sample_name] = portfolio_summary
        portfolio_rows.append({"sample": sample_name, **portfolio_summary})
        if not portfolio_periods.empty:
            portfolio_periods.insert(0, "sample", sample_name)
            all_period_returns.append(portfolio_periods)

    sleeve_summary = pd.DataFrame(sleeve_rows)
    portfolio_summary = pd.DataFrame(portfolio_rows)
    validation_test = pd.DataFrame(
        [
            {
                **prefixed(portfolio_summaries["tune"], "tune"),
                **prefixed(portfolio_summaries["confirm"], "confirm"),
                **prefixed(portfolio_summaries["test"], "test"),
            }
        ]
    )

    sleeve_summary.to_csv(out_dir / "sleeve_summary.csv", index=False)
    portfolio_summary.to_csv(out_dir / "portfolio_summary.csv", index=False)
    validation_test.to_csv(out_dir / "portfolio_validation_test_summary.csv", index=False)
    if all_trades:
        pd.concat(all_trades, ignore_index=True).to_csv(out_dir / "portfolio_trades.csv", index=False)
    if all_period_returns:
        pd.concat(all_period_returns, ignore_index=True).to_csv(out_dir / "portfolio_period_returns.csv", index=False)

    baseline = load_baseline(run_dir)
    baseline_cols = [
        "model",
        "threshold_quantile",
        "top_k",
        "max_hold",
        "filter_name",
        "tune_total_return",
        "confirm_total_return",
        "test_total_return",
        "test_median_trade_return",
        "test_win_rate",
        "test_max_drawdown",
    ]
    portfolio_cols = [
        "sample",
        "periods",
        "trades",
        "tickers",
        "total_return",
        "max_drawdown",
        "mean_period_return",
        "median_period_return",
        "positive_period_rate",
        "median_trade_return",
        "win_rate",
        "mean_ex_best",
        "event_trade_rate",
        "announcement_exit_rate",
    ]
    sleeve_cols = [
        "sample",
        "sleeve",
        "model",
        "threshold_quantile",
        "threshold",
        "top_k",
        "max_hold",
        "filter_name",
        "trades",
        "total_return",
        "median_trade_return",
        "win_rate",
        "mean_ex_best",
        "max_drawdown",
    ]
    validation_cols = [
        "tune_total_return",
        "confirm_total_return",
        "test_total_return",
        "test_max_drawdown",
        "test_median_trade_return",
        "test_win_rate",
        "test_mean_ex_best",
        "test_event_trade_rate",
        "test_announcement_exit_rate",
    ]
    report = [
        "# Return-First Two-Sleeve Strategy",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This portfolio treats merger probability as an input, not the objective. The objective is tradable return after the first selected signal.",
        "",
        "Entry rule: every 20 trading days, buy up to 10 names per sleeve at the next open after the signal date.",
        "",
        "Exit rule: sell at the audited announcement 24-hour daily-high proxy if an announcement occurs before the 20-trading-day time stop; otherwise sell at the 20-trading-day close. This keeps comparability with the announcement-peak research, but the daily-high announcement exit remains optimistic for real execution.",
        "",
        "Capital rule: equal capital to each sleeve; if one sleeve has no trades in a rebalance period, that sleeve sits in cash.",
        "",
        "## Portfolio Summary",
        report_table(portfolio_summary, portfolio_cols),
        "",
        "## Validation/Test Rollup",
        report_table(validation_test, validation_cols),
        "",
        "## Sleeve Summary",
        report_table(sleeve_summary, sleeve_cols),
        "",
        "## Baseline Robust Single-Sleeve Strategy",
        report_table(baseline, baseline_cols),
        "",
        "## Interpretation",
        "",
        "The improvement comes from combining two different return hypotheses: high event probability that is not fully reflected in price, and stale SEC language that may be rediscovered by the market. This is more aligned with demand-spike capture than maximizing merger-label accuracy.",
        "",
        "This is still a research result, not a production trading system. The test sample is small, the announcement-peak exit is optimistic, and this candidate should be revalidated with more history and a broker-executable exit before risking capital.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full_strategy_review")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/two_sleeve_return_strategy",
    )
    parser.add_argument("--price-dir", action="append", default=["data/raw/stockanalysis_prices", "data/raw/yfinance"])
    parser.add_argument("--cost-bps", type=float, default=250.0)
    parser.add_argument("--tune-end", default="2026-04-01")
    args = parser.parse_args()
    run(
        run_dir=Path(args.run_dir),
        events_path=Path(args.events),
        price_dirs=[Path(p) for p in args.price_dir],
        out_dir=Path(args.out_dir),
        cost_bps=args.cost_bps,
        tune_end=args.tune_end,
    )


if __name__ == "__main__":
    main()
