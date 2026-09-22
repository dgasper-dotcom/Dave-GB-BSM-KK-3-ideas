from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .early_signal_strategy import apply_filter
from .execution_resilient_strategy import (
    add_harsh_execution_columns,
    choose_spec,
    make_event_model,
    make_positive_model,
    make_return_regressor,
    rank_values,
    sample_training_rows,
    score_frame,
)
from .profitability_research import MARKET_FEATURES
from .sec_market_fusion import sec_feature_columns
from .walk_forward_return_backtest import load_panel, table


def latest_signal_date(panel: pd.DataFrame, signal_date: str | None) -> pd.Timestamp:
    if signal_date:
        requested = pd.Timestamp(signal_date).normalize()
        available = panel.loc[panel["date"].le(requested), "date"].max()
        if pd.isna(available):
            raise ValueError(f"No panel rows on or before signal_date={signal_date}")
        return pd.Timestamp(available).normalize()
    return pd.Timestamp(panel["date"].max()).normalize()


def add_entry_eligibility(
    frame: pd.DataFrame,
    account_size: float,
    position_pct: float,
    max_pct_adv: float,
    min_adv: float,
    min_price: float,
    max_open_gap_up: float,
    max_spread_proxy: float,
) -> pd.DataFrame:
    out = frame.copy()
    out["planned_position_dollars"] = account_size * position_pct
    out["order_pct_adv"] = out["planned_position_dollars"] / out["adv_20d_dollars"].replace(0, np.nan)
    out["open_gap"] = out["next_open"] / out["close"] - 1
    known_now = out[["close", "adv_20d_dollars", "price", "intraday_range"]].notna().all(axis=1)
    entry_ok = (
        known_now
        & out["adv_20d_dollars"].ge(min_adv)
        & out["price"].ge(min_price)
        & out["order_pct_adv"].le(max_pct_adv)
        & out["intraday_range"].le(max_spread_proxy)
    )
    open_known = out["next_open"].notna()
    entry_ok &= (~open_known) | out["open_gap"].le(max_open_gap_up)
    out["paper_entry_eligible"] = entry_ok
    out["paper_entry_condition"] = np.where(
        open_known,
        "next_open_gap_checked",
        f"pending_next_open_gap_check_le_{max_open_gap_up:.2%}",
    )
    return out


def select_paper_orders(scored: pd.DataFrame, spec, top_k_override: int | None = None) -> pd.DataFrame:
    if spec is None or scored.empty:
        return scored.iloc[[]].copy()
    eligible = scored[scored["paper_entry_eligible"].fillna(False)].copy()
    if eligible.empty:
        return eligible
    eligible["score"] = eligible["event_score"]
    eligible = apply_filter(eligible, spec.filter_name)
    eligible = eligible[
        eligible["event_score"].ge(spec.event_threshold)
        & eligible["expected_net_return_20d"].ge(spec.return_threshold)
        & eligible["positive_return_score"].ge(spec.positive_threshold)
    ].copy()
    if eligible.empty:
        return eligible
    eligible["rank_score"] = rank_values(eligible, spec.rank_by)
    eligible["paper_rank"] = eligible["rank_score"].rank(ascending=False, method="first")
    top_k = top_k_override if top_k_override is not None else spec.top_k
    orders = eligible[eligible["paper_rank"].le(top_k)].copy()
    orders["paper_action"] = "PAPER_BUY_NEXT_OPEN_CONDITIONAL"
    orders["entry_rule"] = "next open only if gap/liquidity/spread checks still pass"
    orders["exit_rule"] = "paper exit at 20 trading-day close; log actual fills and slippage"
    return orders.sort_values("paper_rank")


def make_watchlist(scored: pd.DataFrame, orders: pd.DataFrame, watchlist_size: int, no_order_reason: str) -> pd.DataFrame:
    watch = scored.copy()
    watch["rank_score"] = watch["execution_score"].fillna(0)
    watch = watch.sort_values(["paper_entry_eligible", "rank_score"], ascending=[False, False]).head(watchlist_size).copy()
    order_keys = set(zip(orders.get("ticker", []), pd.to_datetime(orders.get("date", []))))
    watch["paper_action"] = [
        "PAPER_BUY_NEXT_OPEN_CONDITIONAL" if (row.ticker, pd.Timestamp(row.date)) in order_keys else "WATCH_ONLY"
        for row in watch.itertuples(index=False)
    ]
    watch["no_order_reason"] = np.where(watch["paper_action"].eq("WATCH_ONLY"), no_order_reason, "")
    return watch


def write_model_card(
    out_dir: Path,
    panel_path: Path,
    summary: dict[str, object],
    orders: pd.DataFrame,
    watchlist: pd.DataFrame,
) -> None:
    order_table = table(
        orders,
        [
            "ticker",
            "date",
            "close",
            "next_open",
            "event_score",
            "expected_net_return_20d",
            "positive_return_score",
            "execution_score",
            "paper_rank",
            "planned_position_dollars",
            "paper_entry_condition",
        ],
        max_rows=25,
    )
    watch_table = table(
        watchlist,
        [
            "ticker",
            "date",
            "close",
            "next_open",
            "paper_action",
            "event_score",
            "expected_net_return_20d",
            "positive_return_score",
            "execution_score",
            "paper_entry_eligible",
            "no_order_reason",
        ],
        max_rows=25,
    )
    report = [
        "# Paper Trading Model",
        "",
        f"Panel: `{panel_path}`",
        f"Signal date: `{summary['signal_date']}`",
        f"Training cutoff: outcomes known before `{summary['training_cutoff']}`",
        "",
        "This is a paper-testing scanner, not a live trading recommendation. It is designed to abstain unless the return-aware model expects nonnegative harsh net return after adverse entry, exit, and fixed-cost assumptions.",
        "",
        "## Decision",
        "",
        f"- Paper orders: `{len(orders)}`",
        f"- Watchlist rows: `{len(watchlist)}`",
        f"- Decision reason: {summary['decision_reason']}",
        "",
        "## Paper Orders",
        order_table,
        "",
        "## Watchlist",
        watch_table,
        "",
        "## Operating Rules",
        "",
        "- Score after the signal-date close.",
        "- Paper-buy at the next open only if gap, liquidity, price, and spread-proxy checks still pass.",
        "- Position size is 1% of the paper account by default.",
        "- Exit at the 20-trading-day close for apples-to-apples evaluation.",
        "- Log actual entry/exit prices, bid/ask/mid snapshots, order type, limit price, partial fills, rejects, borrow/halts/news, and actual exit price.",
        "- Do not promote to live trading until forward paper results are positive after costs and robust after removing top winners.",
        "",
    ]
    (out_dir / "MODEL_CARD.md").write_text("\n".join(report))


def run(
    panel_path: Path,
    out_dir: Path,
    signal_date_arg: str | None,
    train_lookback_days: int,
    calibration_days: int,
    max_event_train_rows: int,
    max_return_train_rows: int,
    max_iter: int,
    min_train_event_rows: int,
    min_calibration_trades: int,
    min_predicted_net_return: float,
    watchlist_size: int,
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
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(panel_path)
    panel = add_harsh_execution_columns(
        panel,
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
    signal_date = latest_signal_date(panel, signal_date_arg)
    panel["outcome_known_date"] = panel["date"] + pd.offsets.BDay(22)
    train = panel[panel["outcome_known_date"].lt(signal_date)].dropna(subset=["event_20d"]).copy()
    if train_lookback_days > 0:
        train = train[train["date"].ge(signal_date - pd.Timedelta(days=train_lookback_days))].copy()
    signal_frame = panel[panel["date"].eq(signal_date)].copy()
    if signal_frame.empty:
        raise ValueError(f"No rows for signal_date={signal_date.date()}")
    if train.empty or train["event_20d"].nunique() < 2 or int(train["event_20d"].sum()) < min_train_event_rows:
        raise ValueError("Insufficient training history for paper model.")

    features = MARKET_FEATURES + sec_feature_columns(panel)
    event_train = sample_training_rows(train, max_event_train_rows, random_state=42)
    return_train = train[train["execution_tradable"].fillna(False) & train["harsh_net_return_20d"].notna()].copy()
    return_train = sample_training_rows(return_train, max_return_train_rows, random_state=4242)
    if return_train.empty or return_train["harsh_net_return_20d"].gt(0).nunique() < 2:
        raise ValueError("Insufficient execution-return outcomes for return model.")

    event_model = make_event_model(max_iter)
    event_model.fit(event_train[features], event_train["event_20d"].astype(int))
    lo, hi = return_train["harsh_net_return_20d"].quantile([0.01, 0.99])
    return_target = return_train["harsh_net_return_20d"].clip(lower=lo, upper=hi)
    positive_target = return_train["harsh_net_return_20d"].gt(0).astype(int)
    return_model = make_return_regressor(max_iter)
    positive_model = make_positive_model(max_iter)
    return_model.fit(return_train[features], return_target)
    positive_model.fit(return_train[features], positive_target)

    calibration = train[
        train["date"].ge(signal_date - pd.Timedelta(days=calibration_days))
        & train["execution_tradable"].fillna(False)
    ].copy()
    if calibration.empty:
        calibration = return_train.copy()
    scored_calibration = score_frame(calibration, features, event_model, return_model, positive_model)
    spec, selected_summary = choose_spec(
        scored_calibration,
        min_calibration_trades=min_calibration_trades,
        min_predicted_net_return=min_predicted_net_return,
    )

    scored_signal = score_frame(signal_frame, features, event_model, return_model, positive_model)
    scored_signal = add_entry_eligibility(
        scored_signal,
        account_size=account_size,
        position_pct=position_pct,
        max_pct_adv=max_pct_adv,
        min_adv=min_adv,
        min_price=min_price,
        max_open_gap_up=max_open_gap_up,
        max_spread_proxy=max_spread_proxy,
    )
    if spec is None:
        orders = scored_signal.iloc[[]].copy()
        decision_reason = "No recent calibration strategy passed positive harsh-return robustness gates."
        selected_spec = None
    else:
        orders = select_paper_orders(scored_signal, spec)
        decision_reason = (
            "Selected calibration strategy passed; no current names met all gates."
            if orders.empty
            else "Selected calibration strategy passed and current names met all gates."
        )
        selected_spec = asdict(spec)
    watchlist = make_watchlist(scored_signal, orders, watchlist_size, decision_reason)

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(event_model, model_dir / "event_model.joblib")
    joblib.dump(return_model, model_dir / "return_model.joblib")
    joblib.dump(positive_model, model_dir / "positive_return_model.joblib")
    (model_dir / "feature_columns.json").write_text(json.dumps(features, indent=2))

    orders.to_csv(out_dir / "paper_orders.csv", index=False)
    watchlist.to_csv(out_dir / "paper_watchlist.csv", index=False)
    scored_signal.to_csv(out_dir / "latest_scores.csv", index=False)
    scored_calibration.to_csv(out_dir / "calibration_scores.csv", index=False)
    pd.DataFrame(
        columns=[
            "ticker",
            "signal_date",
            "paper_action",
            "entry_order_type",
            "entry_limit_price",
            "planned_entry_date",
            "planned_position_dollars",
            "planned_entry_benchmark_price",
            "actual_entry_time",
            "actual_entry_price",
            "actual_entry_bid",
            "actual_entry_ask",
            "actual_entry_mid",
            "actual_entry_spread_bps",
            "actual_entry_shares",
            "actual_entry_status",
            "actual_entry_reject_reason",
            "actual_exit_date",
            "actual_exit_time",
            "planned_exit_benchmark_price",
            "actual_exit_price",
            "actual_exit_bid",
            "actual_exit_ask",
            "actual_exit_mid",
            "actual_exit_spread_bps",
            "actual_exit_shares",
            "actual_exit_status",
            "actual_net_return",
            "borrow_halt_news_notes",
            "notes",
        ]
    ).to_csv(out_dir / "paper_trade_journal_template.csv", index=False)

    summary = {
        "signal_date": str(signal_date.date()),
        "training_cutoff": str(signal_date.date()),
        "panel_path": str(panel_path),
        "train_rows": int(len(train)),
        "event_train_rows": int(len(event_train)),
        "return_train_rows": int(len(return_train)),
        "train_event_rows": int(train["event_20d"].sum()),
        "signal_rows": int(len(signal_frame)),
        "paper_orders": int(len(orders)),
        "watchlist_rows": int(len(watchlist)),
        "decision_reason": decision_reason,
        "selected_spec": selected_spec,
        "selected_calibration_summary": selected_summary,
        "harsh_execution_assumptions": {
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
            "min_predicted_net_return": min_predicted_net_return,
        },
    }
    (out_dir / "paper_model_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    write_model_card(out_dir, panel_path, summary, orders, watchlist)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv")
    parser.add_argument("--out-dir", default="reports/paper_trading_model")
    parser.add_argument("--signal-date")
    parser.add_argument("--train-lookback-days", type=int, default=730)
    parser.add_argument("--calibration-days", type=int, default=365)
    parser.add_argument("--max-event-train-rows", type=int, default=100_000)
    parser.add_argument("--max-return-train-rows", type=int, default=100_000)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument("--min-train-event-rows", type=int, default=100)
    parser.add_argument("--min-calibration-trades", type=int, default=8)
    parser.add_argument("--min-predicted-net-return", type=float, default=0.0)
    parser.add_argument("--watchlist-size", type=int, default=50)
    parser.add_argument("--account-size", type=float, default=1_000_000)
    parser.add_argument("--position-pct", type=float, default=0.01)
    parser.add_argument("--max-pct-adv", type=float, default=0.02)
    parser.add_argument("--min-adv", type=float, default=500_000)
    parser.add_argument("--min-price", type=float, default=1.0)
    parser.add_argument("--max-open-gap-up", type=float, default=0.10)
    parser.add_argument("--max-spread-proxy", type=float, default=0.35)
    parser.add_argument("--entry-slippage-bps", type=float, default=100.0)
    parser.add_argument("--exit-slippage-bps", type=float, default=50.0)
    parser.add_argument("--fixed-cost-bps", type=float, default=150.0)
    args = parser.parse_args()
    summary = run(
        panel_path=Path(args.panel),
        out_dir=Path(args.out_dir),
        signal_date_arg=args.signal_date,
        train_lookback_days=args.train_lookback_days,
        calibration_days=args.calibration_days,
        max_event_train_rows=args.max_event_train_rows,
        max_return_train_rows=args.max_return_train_rows,
        max_iter=args.max_iter,
        min_train_event_rows=args.min_train_event_rows,
        min_calibration_trades=args.min_calibration_trades,
        min_predicted_net_return=args.min_predicted_net_return,
        watchlist_size=args.watchlist_size,
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
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
