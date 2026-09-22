from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .early_signal_strategy import apply_filter
from .profitability_research import MARKET_FEATURES
from .sec_market_fusion import sec_feature_columns
from .walk_forward_return_backtest import (
    load_panel,
    quarter_folds,
    robustness_rows,
    summarize_unique_ticker_portfolio,
    table,
)


@dataclass(frozen=True)
class ExecutionSpec:
    event_quantile: float
    return_quantile: float
    positive_quantile: float
    event_threshold: float
    return_threshold: float
    positive_threshold: float
    top_k: int
    filter_name: str
    rank_by: str


def make_event_model(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def make_return_regressor(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=max_iter,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def make_positive_model(max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=max_iter,
                    learning_rate=0.05,
                    max_leaf_nodes=15,
                    l2_regularization=1.0,
                    early_stopping=True,
                    random_state=42,
                ),
            ),
        ]
    )


def add_harsh_execution_columns(
    panel: pd.DataFrame,
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
) -> pd.DataFrame:
    out = panel.copy()
    for col in ["next_open", "exit_close_20d", "close", "adv_20d_dollars", "price", "intraday_range"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["planned_position_dollars"] = account_size * position_pct
    out["open_gap"] = out["next_open"] / out["close"] - 1
    out["order_pct_adv"] = out["planned_position_dollars"] / out["adv_20d_dollars"].replace(0, np.nan)
    entry = out["next_open"] * (1 + entry_slippage_bps / 10000.0)
    exit_price = out["exit_close_20d"] * (1 - exit_slippage_bps / 10000.0)
    out["harsh_net_return_20d"] = exit_price / entry - 1 - fixed_cost_bps / 10000.0
    out["execution_tradable"] = (
        out[["next_open", "exit_close_20d", "close", "adv_20d_dollars", "price"]].notna().all(axis=1)
        & out["adv_20d_dollars"].ge(min_adv)
        & out["price"].ge(min_price)
        & out["order_pct_adv"].le(max_pct_adv)
        & out["open_gap"].le(max_open_gap_up)
        & out["intraday_range"].le(max_spread_proxy)
    )
    return out


def sample_training_rows(frame: pd.DataFrame, max_rows: int, random_state: int) -> pd.DataFrame:
    if max_rows <= 0 or len(frame) <= max_rows:
        return frame
    return frame.sample(n=max_rows, random_state=random_state).sort_values(["date", "ticker"])


def score_frame(
    frame: pd.DataFrame,
    features: list[str],
    event_model: Pipeline,
    return_model: Pipeline,
    positive_model: Pipeline,
) -> pd.DataFrame:
    scored = frame.copy()
    scored["event_score"] = event_model.predict_proba(scored[features])[:, 1]
    scored["expected_net_return_20d"] = return_model.predict(scored[features])
    scored["positive_return_score"] = positive_model.predict_proba(scored[features])[:, 1]
    scored["execution_score"] = (
        scored["expected_net_return_20d"].clip(lower=0)
        * scored["positive_return_score"].clip(lower=0)
        * np.sqrt(scored["event_score"].clip(lower=0))
    )
    return scored


def rank_values(frame: pd.DataFrame, rank_by: str) -> pd.Series:
    if rank_by == "execution_score":
        return frame["execution_score"]
    if rank_by == "expected_return":
        return frame["expected_net_return_20d"]
    if rank_by == "positive_return":
        return frame["positive_return_score"]
    if rank_by == "event_x_return":
        return frame["event_score"] * frame["expected_net_return_20d"].clip(lower=0)
    raise ValueError(rank_by)


def select_trades(scored: pd.DataFrame, spec: ExecutionSpec) -> pd.DataFrame:
    eligible = scored[scored["execution_tradable"].fillna(False)].copy()
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
    all_dates = pd.Index(sorted(pd.to_datetime(scored["date"]).dropna().unique()))
    rebalance_dates = set(all_dates[::20])
    eligible = eligible[eligible["date"].isin(rebalance_dates)].copy()
    if eligible.empty:
        return eligible
    eligible["rank_score"] = rank_values(eligible, spec.rank_by)
    eligible["rank"] = eligible.groupby("date")["rank_score"].rank(ascending=False, method="first")
    trades = eligible[eligible["rank"].le(spec.top_k)].copy()
    trades["entry_date"] = trades["date"] + pd.offsets.BDay(1)
    trades["exit_kind"] = "execution_resilient_20d_time_stop"
    trades["gross_return"] = trades["harsh_net_return_20d"]
    trades["net_return"] = trades["harsh_net_return_20d"]
    return trades


def summarize(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "periods": 0,
            "tickers": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "mean_trade_return": np.nan,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "event_trade_rate": np.nan,
            "mean_ex_best": np.nan,
            "total_return_ex_best": np.nan,
        }
    period_returns = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    out = {
        "trades": int(len(trades)),
        "periods": int(len(period_returns)),
        "tickers": int(trades["ticker"].nunique()),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "mean_period_return": float(period_returns.mean()),
        "median_period_return": float(period_returns.median()),
        "positive_period_rate": float(period_returns.gt(0).mean()),
        "mean_trade_return": float(trades["net_return"].mean()),
        "median_trade_return": float(trades["net_return"].median()),
        "win_rate": float(trades["net_return"].gt(0).mean()),
        "event_trade_rate": float(trades["event_20d"].fillna(0).eq(1).mean()),
    }
    if len(trades) > 1:
        trimmed = trades.drop(index=trades["net_return"].idxmax())
        trimmed_periods = trimmed.groupby("date")["net_return"].mean().sort_index()
        out["mean_ex_best"] = float(trimmed["net_return"].mean())
        out["total_return_ex_best"] = float((1 + trimmed_periods).prod() - 1)
    else:
        out["mean_ex_best"] = np.nan
        out["total_return_ex_best"] = np.nan
    return out


def make_specs(calibration: pd.DataFrame, min_predicted_net_return: float) -> list[ExecutionSpec]:
    event_quantiles = [0.90, 0.95, 0.98]
    return_quantiles = [0.90, 0.95, 0.98]
    positive_quantiles = [0.50, 0.70]
    top_ks = [1, 3, 5]
    filters = ["all", "away_high", "no_big_runup", "quiet_not_priced", "stale_sec_not_recent", "strategic_review_90"]
    rank_bys = ["execution_score", "expected_return", "event_x_return"]
    specs: list[ExecutionSpec] = []
    tradable = calibration[calibration["execution_tradable"].fillna(False)].copy()
    if tradable.empty:
        return specs
    for event_q in event_quantiles:
        event_threshold = float(tradable["event_score"].quantile(event_q))
        for return_q in return_quantiles:
            return_threshold = max(
                float(tradable["expected_net_return_20d"].quantile(return_q)),
                min_predicted_net_return,
            )
            for positive_q in positive_quantiles:
                positive_threshold = float(tradable["positive_return_score"].quantile(positive_q))
                for top_k in top_ks:
                    for filter_name in filters:
                        for rank_by in rank_bys:
                            specs.append(
                                ExecutionSpec(
                                    event_quantile=event_q,
                                    return_quantile=return_q,
                                    positive_quantile=positive_q,
                                    event_threshold=event_threshold,
                                    return_threshold=return_threshold,
                                    positive_threshold=positive_threshold,
                                    top_k=top_k,
                                    filter_name=filter_name,
                                    rank_by=rank_by,
                                )
                            )
    return specs


def choose_spec(
    calibration: pd.DataFrame,
    min_calibration_trades: int,
    min_predicted_net_return: float,
) -> tuple[ExecutionSpec | None, dict[str, object]]:
    rows = []
    for spec in make_specs(calibration, min_predicted_net_return=min_predicted_net_return):
        trades = select_trades(calibration, spec)
        summary = summarize(trades)
        if summary["trades"] < min_calibration_trades or summary["periods"] < 4:
            continue
        if (
            summary["total_return"] <= 0
            or summary["total_return_ex_best"] <= 0
            or summary["mean_ex_best"] <= 0
            or summary["median_trade_return"] <= 0
            or summary["max_drawdown"] <= -0.35
            or summary["win_rate"] < 0.42
        ):
            continue
        rows.append({**spec.__dict__, **summary})
    if not rows:
        return None, {}
    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["total_return_ex_best", "median_trade_return", "mean_ex_best", "max_drawdown", "trades"],
        ascending=[False, False, False, False, False],
    )
    selected = grid.iloc[0].to_dict()
    spec = ExecutionSpec(
        event_quantile=float(selected["event_quantile"]),
        return_quantile=float(selected["return_quantile"]),
        positive_quantile=float(selected["positive_quantile"]),
        event_threshold=float(selected["event_threshold"]),
        return_threshold=float(selected["return_threshold"]),
        positive_threshold=float(selected["positive_threshold"]),
        top_k=int(selected["top_k"]),
        filter_name=str(selected["filter_name"]),
        rank_by=str(selected["rank_by"]),
    )
    return spec, selected


def run(
    panel_path: Path,
    out_dir: Path,
    start: str,
    target: str,
    max_iter: int,
    train_lookback_days: int,
    calibration_days: int,
    max_event_train_rows: int,
    max_return_train_rows: int,
    min_train_event_rows: int,
    min_calibration_trades: int,
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
    min_predicted_net_return: float,
) -> None:
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
    panel["outcome_known_date"] = panel["date"] + pd.offsets.BDay(22)
    features = MARKET_FEATURES + sec_feature_columns(panel)

    fold_rows = []
    selection_rows = []
    trade_frames = []
    scored_sample_frames = []
    folds = quarter_folds(panel, start)
    for fold_number, (fold_start, fold_end) in enumerate(folds, start=1):
        known_cutoff = fold_start
        train = panel[panel["outcome_known_date"].lt(known_cutoff)].dropna(subset=[target]).copy()
        if train_lookback_days > 0:
            train = train[train["date"].ge(fold_start - pd.Timedelta(days=train_lookback_days))].copy()
        test = panel[panel["date"].ge(fold_start) & panel["date"].lt(fold_end)].copy()
        train_event_rows = int(train[target].sum()) if not train.empty else 0
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

        event_train = sample_training_rows(train, max_event_train_rows, random_state=fold_number)
        return_train = train[train["execution_tradable"].fillna(False) & train["harsh_net_return_20d"].notna()].copy()
        return_train = sample_training_rows(return_train, max_return_train_rows, random_state=10_000 + fold_number)
        if return_train.empty or return_train["harsh_net_return_20d"].gt(0).nunique() < 2:
            fold_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "status": "skipped_return_model",
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": train_event_rows,
                }
            )
            continue

        event_model = make_event_model(max_iter)
        event_model.fit(event_train[features], event_train[target].astype(int))

        lo, hi = return_train["harsh_net_return_20d"].quantile([0.01, 0.99])
        return_target = return_train["harsh_net_return_20d"].clip(lower=lo, upper=hi)
        positive_target = return_train["harsh_net_return_20d"].gt(0).astype(int)
        return_model = make_return_regressor(max_iter)
        positive_model = make_positive_model(max_iter)
        return_model.fit(return_train[features], return_target)
        positive_model.fit(return_train[features], positive_target)

        calibration = train[train["date"].ge(fold_start - pd.Timedelta(days=calibration_days))].copy()
        calibration = calibration[calibration["execution_tradable"].fillna(False)].copy()
        if calibration.empty:
            calibration = return_train.copy()
        scored_calibration = score_frame(calibration, features, event_model, return_model, positive_model)
        spec, selected = choose_spec(
            scored_calibration,
            min_calibration_trades=min_calibration_trades,
            min_predicted_net_return=min_predicted_net_return,
        )
        if spec is None:
            fold_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "status": "no_calibration_strategy",
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": train_event_rows,
                }
            )
            continue

        scored_test = score_frame(test, features, event_model, return_model, positive_model)
        if fold_number >= len(folds) - 1:
            scored_sample_frames.append(
                scored_test[
                    [
                        "ticker",
                        "date",
                        "event_score",
                        "expected_net_return_20d",
                        "positive_return_score",
                        "execution_score",
                        "execution_tradable",
                        "harsh_net_return_20d",
                    ]
                ].copy()
            )
        trades = select_trades(scored_test, spec)
        trades = trades.dropna(subset=["net_return"]).copy()
        if not trades.empty:
            trades["fold_start"] = fold_start.date()
            trades["fold_end"] = fold_end.date()
            trade_frames.append(trades)
        test_summary = summarize(trades)
        selection_rows.append(
            {
                "fold_start": fold_start.date(),
                "fold_end": fold_end.date(),
                "train_rows": len(train),
                "test_rows": len(test),
                "train_event_rows": train_event_rows,
                **selected,
            }
        )
        fold_rows.append(
            {
                "fold_start": fold_start.date(),
                "fold_end": fold_end.date(),
                "status": "traded" if not trades.empty else "selected_no_trades",
                "train_rows": len(train),
                "test_rows": len(test),
                "train_event_rows": train_event_rows,
                **test_summary,
            }
        )

    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    overall = summarize(all_trades)
    unique = summarize_unique_ticker_portfolio(all_trades)
    robustness = pd.DataFrame(robustness_rows(all_trades, "execution_resilient"))
    fold_frame = pd.DataFrame(fold_rows)
    selection_frame = pd.DataFrame(selection_rows)
    if robustness.empty:
        robustness = pd.DataFrame(columns=["mode", "test", "periods", "trades", "total_return", "median_period_return"])
    if selection_frame.empty:
        selection_frame = pd.DataFrame(
            columns=[
                "fold_start",
                "fold_end",
                "train_rows",
                "test_rows",
                "train_event_rows",
                "event_quantile",
                "return_quantile",
                "positive_quantile",
                "top_k",
                "filter_name",
                "rank_by",
                "trades",
                "total_return_ex_best",
                "median_trade_return",
                "win_rate",
            ]
        )

    fold_frame.to_csv(out_dir / "execution_resilient_fold_summary.csv", index=False)
    selection_frame.to_csv(out_dir / "execution_resilient_selected_specs.csv", index=False)
    pd.DataFrame([{"mode": "execution_resilient", **overall}]).to_csv(
        out_dir / "execution_resilient_overall_summary.csv", index=False
    )
    pd.DataFrame([{"mode": "execution_resilient_unique_ticker", **unique}]).to_csv(
        out_dir / "execution_resilient_unique_ticker_summary.csv", index=False
    )
    robustness.to_csv(out_dir / "execution_resilient_robustness_summary.csv", index=False)
    if not all_trades.empty:
        all_trades.to_csv(out_dir / "execution_resilient_trades.csv", index=False)
    if scored_sample_frames:
        pd.concat(scored_sample_frames, ignore_index=True).to_csv(out_dir / "execution_resilient_recent_scores_sample.csv", index=False)

    params = pd.DataFrame(
        [
            {
                "train_lookback_days": train_lookback_days,
                "calibration_days": calibration_days,
                "max_event_train_rows": max_event_train_rows,
                "max_return_train_rows": max_return_train_rows,
                "min_train_event_rows": min_train_event_rows,
                "min_calibration_trades": min_calibration_trades,
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
                "max_iter": max_iter,
            }
        ]
    )
    params.to_csv(out_dir / "execution_resilient_parameters.csv", index=False)

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
        "total_return_ex_best",
        "event_trade_rate",
    ]
    fold_cols = [
        "fold_start",
        "fold_end",
        "status",
        "trades",
        "total_return",
        "total_return_ex_best",
        "max_drawdown",
        "median_trade_return",
        "win_rate",
        "event_trade_rate",
    ]
    report = [
        "# Execution-Resilient Return Strategy",
        "",
        f"Panel: `{panel_path}`",
        f"Date range: `{panel['date'].min().date()}` to `{panel['date'].max().date()}`",
        "",
        "This strategy optimizes for harsh 20-trading-day net return, not merger-label accuracy. Each fold trains only on rows whose 20-day outcome would already be known before the fold starts.",
        "",
        "Entry rule: score after the signal-date close; attempt the next open only if liquidity, open-gap, price, and spread-proxy filters still pass. Exit rule: 20-trading-day close with adverse exit slippage.",
        "",
        "## Harsh Assumptions",
        table(params, list(params.columns)),
        "",
        "## Overall",
        table(pd.DataFrame([{"mode": "execution_resilient", **overall}]), overall_cols),
        "",
        "## Unique-Ticker Overall",
        table(pd.DataFrame([{"mode": "execution_resilient_unique_ticker", **unique}]), overall_cols),
        "",
        "## Tail Robustness",
        table(robustness, ["mode", "test", "periods", "trades", "total_return", "median_period_return"]),
        "",
        "## Fold Results",
        table(fold_frame, fold_cols),
        "",
        "## Selected Specs",
        table(
            selection_frame,
            [
                "fold_start",
                "fold_end",
                "event_quantile",
                "return_quantile",
                "positive_quantile",
                "top_k",
                "filter_name",
                "rank_by",
                "trades",
                "total_return_ex_best",
                "median_trade_return",
                "win_rate",
            ],
            max_rows=30,
        ),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", default="reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv")
    parser.add_argument("--out-dir", default="reports/backfill_2018_2026_sec_fusion_cap8/execution_resilient_strategy")
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--target", default="event_20d")
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--train-lookback-days", type=int, default=1095)
    parser.add_argument("--calibration-days", type=int, default=730)
    parser.add_argument("--max-event-train-rows", type=int, default=350_000)
    parser.add_argument("--max-return-train-rows", type=int, default=250_000)
    parser.add_argument("--min-train-event-rows", type=int, default=100)
    parser.add_argument("--min-calibration-trades", type=int, default=20)
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
    parser.add_argument("--min-predicted-net-return", type=float, default=0.0)
    args = parser.parse_args()
    run(
        panel_path=Path(args.panel),
        out_dir=Path(args.out_dir),
        start=args.start,
        target=args.target,
        max_iter=args.max_iter,
        train_lookback_days=args.train_lookback_days,
        calibration_days=args.calibration_days,
        max_event_train_rows=args.max_event_train_rows,
        max_return_train_rows=args.max_return_train_rows,
        min_train_event_rows=args.min_train_event_rows,
        min_calibration_trades=args.min_calibration_trades,
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
        min_predicted_net_return=args.min_predicted_net_return,
    )


if __name__ == "__main__":
    main()
