from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .early_signal_strategy import (
    StrategySpec,
    compute_trade_exit,
    load_events,
    load_predictions,
    load_prices,
    select_candidates,
)


@dataclass(frozen=True)
class BarrierSpec:
    model: str
    threshold_quantile: float
    threshold: float
    top_k: int
    max_hold: int
    filter_name: str
    take_profit: float
    stop_loss: float


def strategy_spec(spec: BarrierSpec) -> StrategySpec:
    return StrategySpec(
        mode="nonoverlap",
        model=spec.model,
        threshold_quantile=spec.threshold_quantile,
        threshold=spec.threshold,
        top_k=spec.top_k,
        max_hold=spec.max_hold,
        filter_name=spec.filter_name,
    )


def barrier_exit_for_row(
    row: pd.Series,
    events_by_ticker: dict[str, pd.Series],
    price_cache: dict[str, pd.DataFrame],
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]],
    price_dirs: list[Path],
    spec: BarrierSpec,
    cost_bps: float,
) -> dict[str, object]:
    base = compute_trade_exit(
        row,
        events_by_ticker,
        price_cache,
        exit_cache,
        price_dirs,
        spec.max_hold,
        cost_bps,
    )
    ticker = str(row["ticker"])
    if ticker not in price_cache:
        price_cache[ticker] = load_prices(ticker, price_dirs)
    prices = price_cache[ticker]
    if prices.empty or pd.isna(row["next_open"]):
        return base

    signal_date = pd.Timestamp(row["date"]).normalize()
    matches = prices.index[prices["date"].eq(signal_date)].tolist()
    if not matches:
        return base
    signal_idx = matches[0]
    entry_idx = signal_idx + 1
    time_exit_idx = signal_idx + spec.max_hold
    if entry_idx >= len(prices) or time_exit_idx >= len(prices):
        return base

    entry = float(row["next_open"])
    stop_price = entry * (1 - spec.stop_loss) if spec.stop_loss > 0 else -np.inf
    target_price = entry * (1 + spec.take_profit) if spec.take_profit > 0 else np.inf
    barrier = None
    for idx in range(entry_idx, time_exit_idx + 1):
        low = float(prices.loc[idx, "low"])
        high = float(prices.loc[idx, "high"])
        if low <= stop_price and high >= target_price:
            barrier = (idx, stop_price, "stop_loss_ambiguous_first")
            break
        if low <= stop_price:
            barrier = (idx, stop_price, "stop_loss")
            break
        if high >= target_price:
            barrier = (idx, target_price, "take_profit")
            break

    if barrier is None:
        return base

    barrier_idx, exit_price, exit_kind = barrier
    barrier_date = prices.loc[barrier_idx, "date"]
    base_exit_date = pd.Timestamp(base.get("exit_date")) if pd.notna(base.get("exit_date")) else pd.NaT
    if pd.notna(base_exit_date) and str(base.get("exit_kind", "")).startswith("announcement_peak"):
        # Keep the audited announcement exit if it occurs before the stop/target day.
        if base_exit_date < pd.Timestamp(barrier_date):
            return base

    gross = exit_price / entry - 1
    return {
        "entry_date": prices.loc[entry_idx, "date"],
        "exit_date": barrier_date,
        "exit_kind": exit_kind,
        "exit_price": float(exit_price),
        "gross_return": float(gross),
        "net_return": float(gross - cost_bps / 10000.0),
        "announcement_ts_et": base.get("announcement_ts_et", pd.NaT),
    }


def summarize(trades: pd.DataFrame, spec: BarrierSpec) -> dict[str, object]:
    if trades.empty:
        return {"trades": 0, **spec.__dict__}
    period_returns = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    out = {
        **spec.__dict__,
        "trades": int(len(trades)),
        "tickers": int(trades["ticker"].nunique()),
        "active_signal_dates": int(len(period_returns)),
        "mean_trade_return": float(trades["net_return"].mean()),
        "median_trade_return": float(trades["net_return"].median()),
        "win_rate": float(trades["net_return"].gt(0).mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "event_trade_rate": float(trades["event_20d"].fillna(0).eq(1).mean()),
        "take_profit_exit_rate": float(trades["exit_kind"].astype(str).str.startswith("take_profit").mean()),
        "stop_loss_exit_rate": float(trades["exit_kind"].astype(str).str.startswith("stop_loss").mean()),
        "announcement_exit_rate": float(trades["exit_kind"].astype(str).str.startswith("announcement_peak").mean()),
    }
    if len(trades) > 1:
        trimmed = trades.drop(index=trades["net_return"].idxmax())
        out["mean_ex_best"] = float(trimmed["net_return"].mean())
        out["median_ex_best"] = float(trimmed["net_return"].median())
    else:
        out["mean_ex_best"] = np.nan
        out["median_ex_best"] = np.nan
    return out


def backtest_barrier_spec(
    df: pd.DataFrame,
    spec: BarrierSpec,
    events_by_ticker: dict[str, pd.Series],
    price_dirs: list[Path],
    cost_bps: float,
    price_cache: dict[str, pd.DataFrame],
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]],
) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = select_candidates(df, strategy_spec(spec))
    if candidates.empty:
        return {"trades": 0, **spec.__dict__}, candidates
    exits = [
        barrier_exit_for_row(row, events_by_ticker, price_cache, exit_cache, price_dirs, spec, cost_bps)
        for _, row in candidates.iterrows()
    ]
    trades = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(exits)], axis=1)
    trades = trades.dropna(subset=["net_return"]).copy()
    return summarize(trades, spec), trades


def make_specs(tune_frames: dict[str, pd.DataFrame]) -> list[BarrierSpec]:
    models = ["rf", "hgb"]
    quantiles = [0.80, 0.90, 0.95, 0.98]
    filters = ["all", "away_high", "no_big_runup", "no_sec30", "quiet_not_priced"]
    take_profits = [0.0, 0.10, 0.15, 0.20, 0.30]
    stop_losses = [0.0, 0.08, 0.10, 0.15, 0.20]
    specs = []
    for model in models:
        thresholds = {q: float(tune_frames[model]["score"].quantile(q)) for q in quantiles}
        for q, threshold in thresholds.items():
            for max_hold in (20, 40):
                for filter_name in filters:
                    for take_profit in take_profits:
                        for stop_loss in stop_losses:
                            if take_profit == 0 and stop_loss == 0:
                                continue
                            specs.append(
                                BarrierSpec(
                                    model=model,
                                    threshold_quantile=q,
                                    threshold=threshold,
                                    top_k=10,
                                    max_hold=max_hold,
                                    filter_name=filter_name,
                                    take_profit=take_profit,
                                    stop_loss=stop_loss,
                                )
                            )
    return specs


def prefixed(summary: dict[str, object], prefix: str) -> dict[str, object]:
    keys = [
        "trades",
        "tickers",
        "active_signal_dates",
        "mean_trade_return",
        "median_trade_return",
        "win_rate",
        "total_return",
        "max_drawdown",
        "event_trade_rate",
        "take_profit_exit_rate",
        "stop_loss_exit_rate",
        "announcement_exit_rate",
        "mean_ex_best",
        "median_ex_best",
    ]
    return {f"{prefix}_{key}": summary.get(key) for key in keys}


def select_deployable(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["min_validation_total_return"] = out[["tune_total_return", "confirm_total_return"]].min(axis=1)
    out["min_validation_mean_ex_best"] = out[["tune_mean_ex_best", "confirm_mean_ex_best"]].min(axis=1)
    out["min_validation_trades"] = out[["tune_trades", "confirm_trades"]].min(axis=1)
    out["median_floor"] = out[["tune_median_trade_return", "confirm_median_trade_return"]].min(axis=1)
    candidates = out[
        out["tune_trades"].ge(25)
        & out["confirm_trades"].ge(25)
        & out["tune_total_return"].gt(0)
        & out["confirm_total_return"].gt(0)
        & out["tune_total_return"].lt(1.0)
        & out["confirm_total_return"].lt(1.0)
        & out["tune_mean_ex_best"].gt(0)
        & out["confirm_mean_ex_best"].gt(0)
        & out["confirm_median_trade_return"].gt(0)
        & out["tune_win_rate"].ge(0.40)
        & out["confirm_win_rate"].ge(0.50)
        & out["tune_max_drawdown"].gt(-0.25)
        & out["confirm_max_drawdown"].gt(-0.25)
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "min_validation_mean_ex_best",
            "min_validation_total_return",
            "median_floor",
            "min_validation_trades",
            "threshold_quantile",
        ],
        ascending=[False, False, False, False, False],
    )


def run(
    run_dir: Path,
    events_path: Path,
    price_dirs: list[Path],
    out_dir: Path,
    cost_bps: float,
    tune_end: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    validation = {model: load_predictions(run_dir, model, "validation") for model in ["rf", "hgb"]}
    test = {model: load_predictions(run_dir, model, "test") for model in ["rf", "hgb"]}
    tune_cutoff = pd.Timestamp(tune_end)
    tune_frames = {model: frame[frame["date"].lt(tune_cutoff)].copy() for model, frame in validation.items()}
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}
    price_cache: dict[str, pd.DataFrame] = {}
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] = {}
    rows = []
    test_trades_by_row: dict[int, pd.DataFrame] = {}
    for spec in make_specs(tune_frames):
        val = validation[spec.model]
        tune = val[val["date"].lt(tune_cutoff)].copy()
        confirm = val[val["date"].ge(tune_cutoff)].copy()
        tune_summary, _ = backtest_barrier_spec(
            tune, spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        if tune_summary.get("trades", 0) < 8:
            continue
        confirm_summary, _ = backtest_barrier_spec(
            confirm, spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        if confirm_summary.get("trades", 0) < 8:
            continue
        test_summary, test_trades = backtest_barrier_spec(
            test[spec.model], spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        rows.append({**spec.__dict__, **prefixed(tune_summary, "tune"), **prefixed(confirm_summary, "confirm"), **prefixed(test_summary, "test")})
        test_trades_by_row[len(rows) - 1] = test_trades

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "barrier_exit_validation_test_grid.csv", index=False)
    if grid.empty:
        (out_dir / "REPORT.md").write_text("No barrier strategies met minimum validation trade counts.")
        return

    deployable = select_deployable(grid)
    deployable.to_csv(out_dir / "barrier_exit_deployable_candidates.csv", index=False)
    selected = deployable.head(1).copy()
    if not selected.empty:
        selected.to_csv(out_dir / "selected_barrier_exit_strategy.csv", index=False)
        selected_idx = int(selected.index[0])
        test_trades_by_row[selected_idx].to_csv(out_dir / "selected_barrier_exit_test_trades.csv", index=False)

    cols = [
        "model",
        "threshold_quantile",
        "threshold",
        "top_k",
        "max_hold",
        "filter_name",
        "take_profit",
        "stop_loss",
        "tune_trades",
        "tune_total_return",
        "tune_median_trade_return",
        "tune_win_rate",
        "tune_mean_ex_best",
        "tune_max_drawdown",
        "confirm_trades",
        "confirm_total_return",
        "confirm_median_trade_return",
        "confirm_win_rate",
        "confirm_mean_ex_best",
        "confirm_max_drawdown",
        "test_trades",
        "test_total_return",
        "test_median_trade_return",
        "test_win_rate",
        "test_mean_ex_best",
        "test_max_drawdown",
        "test_take_profit_exit_rate",
        "test_stop_loss_exit_rate",
        "test_event_trade_rate",
    ]
    selected_md = (
        selected[[c for c in cols if c in selected.columns]].to_markdown(index=False, floatfmt=".4f")
        if not selected.empty
        else "_No barrier strategy met the deployable gates._"
    )
    top_md = (
        deployable[[c for c in cols if c in deployable.columns]].head(25).to_markdown(index=False, floatfmt=".4f")
        if not deployable.empty
        else "_No rows._"
    )
    diagnostics = grid[
        grid["tune_total_return"].gt(0)
        & grid["confirm_total_return"].gt(0)
        & grid["test_total_return"].gt(0)
    ].copy()
    diagnostics_md = (
        diagnostics.sort_values("test_total_return", ascending=False)[[c for c in cols if c in diagnostics.columns]]
        .head(20)
        .to_markdown(index=False, floatfmt=".4f")
        if not diagnostics.empty
        else "_No rows._"
    )
    report = [
        "# Barrier Exit Strategy",
        "",
        f"Source run: `{run_dir}`",
        "",
        "Entry thresholds are derived from the validation tune slice. Stop-loss and take-profit rules are selected on tune/confirm validation before test evaluation.",
        "",
        "If a daily bar touches both stop and target, the stop is counted first to avoid optimistic intraday ordering.",
        "",
        "## Selected Barrier Strategy",
        selected_md,
        "",
        "## Top Deployable Candidates",
        top_md,
        "",
        "## Positive Test Diagnostics",
        "Diagnostic only: sorted by test result, not used for selection.",
        "",
        diagnostics_md,
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full_strategy_review")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/barrier_exit_strategy",
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
