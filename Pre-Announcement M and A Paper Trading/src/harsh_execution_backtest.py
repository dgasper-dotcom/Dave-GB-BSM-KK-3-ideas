from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .early_signal_strategy import StrategySpec, select_candidates
from .profitability_research import MARKET_FEATURES
from .sec_market_fusion import sec_feature_columns
from .two_sleeve_return_strategy import summarize_portfolio
from .walk_forward_return_backtest import (
    DEFAULT_SLEEVES,
    load_panel,
    make_walk_forward_model,
    quarter_folds,
    robustness_rows,
    summarize_unique_ticker_portfolio,
    table,
)


def harsh_trade_filter(
    candidates: pd.DataFrame,
    account_size: float,
    position_pct: float,
    max_pct_adv: float,
    min_adv: float,
    min_price: float,
    max_open_gap_up: float,
    max_spread_proxy: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    trades = candidates.copy()
    trades["planned_position_dollars"] = account_size * position_pct
    trades["open_gap"] = trades["next_open"] / trades["close"] - 1
    trades["order_pct_adv"] = trades["planned_position_dollars"] / trades["adv_20d_dollars"].replace(0, np.nan)
    required = ["next_open", "exit_close_20d", "close", "adv_20d_dollars", "price"]
    ok = pd.Series(True, index=trades.index)
    for col in required:
        missing = trades[col].isna()
        if missing.any():
            rows.append({"skip_reason": f"missing_{col}", "count": int(missing.sum())})
        ok &= ~missing

    checks = {
        "adv_below_floor": trades["adv_20d_dollars"].fillna(0).lt(min_adv),
        "price_below_floor": trades["price"].fillna(0).lt(min_price),
        "order_too_large_vs_adv": trades["order_pct_adv"].fillna(np.inf).gt(max_pct_adv),
        "open_gap_up_too_large": trades["open_gap"].fillna(np.inf).gt(max_open_gap_up),
        "spread_proxy_too_wide": trades["intraday_range"].fillna(np.inf).gt(max_spread_proxy),
    }
    for reason, mask in checks.items():
        mask = mask & ok
        if mask.any():
            rows.append({"skip_reason": reason, "count": int(mask.sum())})
        ok &= ~mask
    return trades[ok].copy(), pd.DataFrame(rows)


def harsh_backtest_spec(
    scored: pd.DataFrame,
    spec: StrategySpec,
    account_size: float,
    position_pct: float,
    max_pct_adv: float,
    min_adv: float,
    min_price: float,
    max_open_gap_up: float,
    max_spread_proxy: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    fixed_cost_bps: float,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    candidates = select_candidates(scored, spec)
    if candidates.empty:
        return {"trades": 0, **spec.__dict__}, candidates, pd.DataFrame()
    trades, skips = harsh_trade_filter(
        candidates,
        account_size=account_size,
        position_pct=position_pct,
        max_pct_adv=max_pct_adv,
        min_adv=min_adv,
        min_price=min_price,
        max_open_gap_up=max_open_gap_up,
        max_spread_proxy=max_spread_proxy,
    )
    if trades.empty:
        return {"trades": 0, **spec.__dict__}, trades, skips

    entry = trades["next_open"] * (1 + entry_slippage_bps / 10000.0)
    exit_price = trades["exit_close_20d"] * (1 - exit_slippage_bps / 10000.0)
    trades["entry_date"] = trades["date"] + pd.offsets.BDay(1)
    trades["exit_kind"] = "harsh_20d_time_stop"
    trades["entry_slippage_bps"] = entry_slippage_bps
    trades["exit_slippage_bps"] = exit_slippage_bps
    trades["fixed_cost_bps"] = fixed_cost_bps
    trades["gross_return"] = exit_price / entry - 1
    trades["net_return"] = trades["gross_return"] - fixed_cost_bps / 10000.0

    period_returns = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
        **spec.__dict__,
        "trades": int(len(trades)),
        "candidate_trades": int(len(candidates)),
        "skipped_trades": int(len(candidates) - len(trades)),
        "tickers": int(trades["ticker"].nunique()),
        "active_signal_dates": int(len(period_returns)),
        "mean_trade_return": float(trades["net_return"].mean()),
        "median_trade_return": float(trades["net_return"].median()),
        "win_rate": float(trades["net_return"].gt(0).mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "event_trade_rate": float(trades["event_20d"].fillna(0).eq(1).mean()),
    }
    if len(trades) > 1:
        summary["mean_ex_best"] = float(trades.drop(index=trades["net_return"].idxmax())["net_return"].mean())
    else:
        summary["mean_ex_best"] = np.nan
    return summary, trades, skips


def run(
    panel_path: Path,
    out_dir: Path,
    start: str,
    target: str,
    max_iter: int,
    account_size: float,
    position_pct: float,
    max_pct_adv: float,
    min_adv: float,
    min_price: float,
    max_open_gap_up: float,
    max_spread_proxy: float,
    entry_slippage_bps: float,
    exit_slippage_bps: float,
    fixed_cost_bps: float,
    min_train_event_rows: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(panel_path)
    features = MARKET_FEATURES + sec_feature_columns(panel)

    fold_rows = []
    sleeve_rows = []
    skip_rows = []
    trades_by_sleeve: dict[str, list[pd.DataFrame]] = {sleeve.name: [] for sleeve in DEFAULT_SLEEVES}

    folds = quarter_folds(panel, start)
    for fold_number, (fold_start, fold_end) in enumerate(folds, start=1):
        train = panel[panel["date"].lt(fold_start)].dropna(subset=[target]).copy()
        test = panel[panel["date"].ge(fold_start) & panel["date"].lt(fold_end)].copy()
        train_event_rows = int(train[target].sum())
        print(
            f"fold={fold_number}/{len(folds)} start={fold_start.date()} end={fold_end.date()} "
            f"train_rows={len(train)} test_rows={len(test)} train_event_rows={train_event_rows}",
            flush=True,
        )
        if train.empty or test.empty or train[target].nunique() < 2 or train_event_rows < min_train_event_rows:
            fold_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "status": "skipped",
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": train_event_rows,
                }
            )
            continue

        model = make_walk_forward_model(max_iter)
        model.fit(train[features], train[target].astype(int))
        calibration = train[train["date"].ge(fold_start - pd.Timedelta(days=365))].copy()
        if calibration[target].nunique() < 2:
            calibration = train
        calibration_scores = model.predict_proba(calibration[features])[:, 1]
        scored = test.copy()
        scored["score"] = model.predict_proba(scored[features])[:, 1]

        fold_trades_by_sleeve = {}
        for sleeve in DEFAULT_SLEEVES:
            threshold = float(np.quantile(calibration_scores, sleeve.threshold_quantile))
            spec = StrategySpec(
                mode="nonoverlap",
                model="hgb",
                threshold_quantile=sleeve.threshold_quantile,
                threshold=threshold,
                top_k=sleeve.top_k,
                max_hold=sleeve.max_hold,
                filter_name=sleeve.filter_name,
            )
            summary, trades, skips = harsh_backtest_spec(
                scored,
                spec,
                account_size=account_size,
                position_pct=position_pct,
                max_pct_adv=max_pct_adv,
                min_adv=min_adv,
                min_price=min_price,
                max_open_gap_up=max_open_gap_up,
                max_spread_proxy=max_spread_proxy,
                entry_slippage_bps=entry_slippage_bps,
                exit_slippage_bps=exit_slippage_bps,
                fixed_cost_bps=fixed_cost_bps,
            )
            tagged = trades.assign(fold_start=fold_start.date(), fold_end=fold_end.date(), sleeve=sleeve.name)
            fold_trades_by_sleeve[sleeve.name] = tagged
            if not tagged.empty:
                trades_by_sleeve[sleeve.name].append(tagged)
            sleeve_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "sleeve": sleeve.name,
                    "threshold": threshold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": train_event_rows,
                    **summary,
                }
            )
            if not skips.empty:
                skips = skips.assign(fold_start=fold_start.date(), fold_end=fold_end.date(), sleeve=sleeve.name)
                skip_rows.append(skips)

        fold_summary, _ = summarize_portfolio(fold_trades_by_sleeve)
        fold_rows.append(
            {
                "fold_start": fold_start.date(),
                "fold_end": fold_end.date(),
                "status": "traded",
                "train_rows": len(train),
                "test_rows": len(test),
                "train_event_rows": train_event_rows,
                **fold_summary,
            }
        )

    sleeve_trades = {
        sleeve: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        for sleeve, frames in trades_by_sleeve.items()
    }
    nonempty = [trades for trades in sleeve_trades.values() if not trades.empty]
    all_trades = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()
    overall, periods = summarize_portfolio(sleeve_trades)
    unique = summarize_unique_ticker_portfolio(all_trades)
    robustness = pd.DataFrame(robustness_rows(all_trades, "harsh_execution"))

    fold_frame = pd.DataFrame(fold_rows)
    sleeve_frame = pd.DataFrame(sleeve_rows)
    fold_frame.to_csv(out_dir / "harsh_fold_summary.csv", index=False)
    sleeve_frame.to_csv(out_dir / "harsh_sleeve_summary.csv", index=False)
    pd.DataFrame([{"mode": "harsh_execution", **overall}]).to_csv(out_dir / "harsh_overall_summary.csv", index=False)
    pd.DataFrame([{"mode": "harsh_execution_unique_ticker", **unique}]).to_csv(
        out_dir / "harsh_unique_ticker_summary.csv", index=False
    )
    robustness.to_csv(out_dir / "harsh_robustness_summary.csv", index=False)
    if not all_trades.empty:
        all_trades.to_csv(out_dir / "harsh_trades.csv", index=False)
    if not periods.empty:
        periods.to_csv(out_dir / "harsh_period_returns.csv", index=False)
    if skip_rows:
        pd.concat(skip_rows, ignore_index=True).to_csv(out_dir / "harsh_skip_summary.csv", index=False)

    params = pd.DataFrame(
        [
            {
                "account_size": account_size,
                "position_pct": position_pct,
                "max_pct_adv": max_pct_adv,
                "min_adv": min_adv,
                "min_price": min_price,
                "max_open_gap_up": max_open_gap_up,
                "max_spread_proxy": max_spread_proxy,
                "entry_slippage_bps": entry_slippage_bps,
                "exit_slippage_bps": exit_slippage_bps,
                "fixed_cost_bps": fixed_cost_bps,
                "max_iter": max_iter,
            }
        ]
    )
    params.to_csv(out_dir / "harsh_parameters.csv", index=False)

    overall_cols = [
        "mode",
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
    ]
    fold_cols = [
        "fold_start",
        "fold_end",
        "trades",
        "total_return",
        "max_drawdown",
        "median_trade_return",
        "win_rate",
        "event_trade_rate",
    ]
    report = [
        "# Harsh Execution Walk-Forward Backtest",
        "",
        f"Panel: `{panel_path}`",
        f"Date range: `{panel['date'].min().date()}` to `{panel['date'].max().date()}`",
        "",
        "This test uses the same two-sleeve signal, but applies harsher execution assumptions before computing 20-trading-day returns.",
        "",
        "## Harsh Assumptions",
        table(params, list(params.columns)),
        "",
        "## Overall Sleeve-Weighted",
        table(pd.DataFrame([{"mode": "harsh_execution", **overall}]), overall_cols),
        "",
        "## Overall Unique-Ticker",
        "This is the cleaner production view: a ticker can only be held once per rebalance date.",
        "",
        table(pd.DataFrame([{"mode": "harsh_execution_unique_ticker", **unique}]), overall_cols),
        "",
        "## Tail Robustness",
        table(robustness, ["mode", "test", "periods", "trades", "total_return", "median_period_return"]),
        "",
        "## Fold Results",
        table(fold_frame[fold_frame["status"].eq("traded")], fold_cols),
        "",
        "## Skip Summary",
        table(pd.concat(skip_rows, ignore_index=True) if skip_rows else pd.DataFrame(), ["sleeve", "skip_reason", "count"], max_rows=40),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        default="reports/profitability_sec_market_fusion_full_strategy_review/fused_market_sec_panel.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/harsh_execution_backtest",
    )
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--target", default="event_20d")
    parser.add_argument("--max-iter", type=int, default=60)
    parser.add_argument("--account-size", type=float, default=1_000_000)
    parser.add_argument("--position-pct", type=float, default=0.01)
    parser.add_argument("--max-pct-adv", type=float, default=0.01)
    parser.add_argument("--min-adv", type=float, default=1_000_000)
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--max-open-gap-up", type=float, default=0.05)
    parser.add_argument("--max-spread-proxy", type=float, default=0.25)
    parser.add_argument("--entry-slippage-bps", type=float, default=200.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=100.0)
    parser.add_argument("--fixed-cost-bps", type=float, default=250.0)
    parser.add_argument("--min-train-event-rows", type=int, default=100)
    args = parser.parse_args()
    run(
        panel_path=Path(args.panel),
        out_dir=Path(args.out_dir),
        start=args.start,
        target=args.target,
        max_iter=args.max_iter,
        account_size=args.account_size,
        position_pct=args.position_pct,
        max_pct_adv=args.max_pct_adv,
        min_adv=args.min_adv,
        min_price=args.min_price,
        max_open_gap_up=args.max_open_gap_up,
        max_spread_proxy=args.max_spread_proxy,
        entry_slippage_bps=args.entry_slippage_bps,
        exit_slippage_bps=args.exit_slippage_bps,
        fixed_cost_bps=args.fixed_cost_bps,
        min_train_event_rows=args.min_train_event_rows,
    )


if __name__ == "__main__":
    main()
