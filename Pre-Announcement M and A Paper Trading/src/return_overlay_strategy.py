from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .early_signal_strategy import apply_filter, compute_trade_exit, load_events
from .profitability_research import MARKET_FEATURES, chronological_splits
from .sec_market_fusion import sec_feature_columns


@dataclass(frozen=True)
class OverlaySpec:
    event_model: str
    event_quantile: float
    event_threshold: float
    return_gate: str
    return_threshold: float
    positive_gate: str
    positive_threshold: float
    rank_by: str
    top_k: int
    max_hold: int
    filter_name: str


def build_return_models(panel: pd.DataFrame, features: list[str], cost_bps: float) -> dict[str, Pipeline]:
    train, _, _ = chronological_splits(panel)
    fit = train.dropna(subset=["fwd_return_20d"]).copy()
    fit["net_return_20d"] = fit["fwd_return_20d"] - cost_bps / 10000.0
    # Winsorize the regression target so one-off acquisition spikes do not dominate the return overlay.
    lo, hi = fit["net_return_20d"].quantile([0.01, 0.99])
    clipped_target = fit["net_return_20d"].clip(lower=lo, upper=hi)
    positive_target = fit["net_return_20d"].gt(0).astype(int)
    return {
        "hgb_reg": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=250,
                        learning_rate=0.03,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ).fit(fit[features], clipped_target),
        "hgb_pos": Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.03,
                        l2_regularization=1.0,
                        random_state=42,
                    ),
                ),
            ]
        ).fit(fit[features], positive_target),
    }


def score_return_models(
    run_dir: Path,
    out_dir: Path,
    cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = pd.read_csv(run_dir / "fused_market_sec_panel.csv", parse_dates=["date", "event_date"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    features = MARKET_FEATURES + sec_feature_columns(panel)
    models = build_return_models(panel, features, cost_bps)
    _, validation, test = chronological_splits(panel)

    scored_frames: dict[str, pd.DataFrame] = {}
    for sample_name, sample in {"validation": validation, "test": test}.items():
        scored = sample[
            [
                "ticker",
                "date",
                "event_date",
                "event_20d",
                "next_open",
                "close",
                "adv_20d_dollars",
                "fwd_return_20d",
            ]
        ].copy()
        scored["expected_net_return_20d"] = models["hgb_reg"].predict(sample[features])
        scored["positive_return_score"] = models["hgb_pos"].predict_proba(sample[features])[:, 1]
        scored["return_overlay_score"] = scored["expected_net_return_20d"] * scored["positive_return_score"]
        scored.to_csv(out_dir / f"return_model_{sample_name}_scores.csv", index=False)
        scored_frames[sample_name] = scored
    return scored_frames["validation"], scored_frames["test"]


def load_overlay_frame(run_dir: Path, return_scores: pd.DataFrame, event_model: str, sample: str) -> pd.DataFrame:
    event = pd.read_csv(run_dir / f"{event_model}_{sample}_predictions.csv", parse_dates=["date"])
    event = event.rename(columns={"score": "event_score"})
    cols = [
        "ticker",
        "date",
        "return_20d",
        "return_60d",
        "distance_from_52w_high",
        "distance_from_52w_low",
        "volume_to_20d_avg",
        "sec_any_event_language_hit_30d",
        "sec_any_event_language_hit_90d",
        "sec_transaction_hit_30d",
        "sec_transaction_hit_90d",
        "sec_strategic_alternatives_hit_30d",
        "sec_strategic_alternatives_hit_90d",
        "sec_investment_bank_hit_90d",
        "sec_committee_hit_90d",
        "sec_change_of_control_hit_90d",
        "sec_confidentiality_hit_90d",
        "sec_unsolicited_interest_hit_90d",
        "sec_multiple_party_interest_hit_90d",
        "sec_days_since_event_language",
    ]
    panel = pd.read_csv(
        run_dir / "fused_market_sec_panel.csv",
        usecols=lambda col: col in cols,
        parse_dates=["date"],
    )
    for col in cols:
        if col not in panel.columns:
            panel[col] = np.nan
    frame = event.merge(
        return_scores[
            [
                "ticker",
                "date",
                "expected_net_return_20d",
                "positive_return_score",
                "return_overlay_score",
            ]
        ],
        on=["ticker", "date"],
        how="left",
    ).merge(panel, on=["ticker", "date"], how="left")
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["score"] = frame["event_score"]
    all_dates = pd.Index(sorted(pd.to_datetime(frame["date"]).dropna().unique()))
    rebalance_20 = set(all_dates[::20])
    rebalance_40 = set(all_dates[::40])
    frame["is_rebalance_20"] = frame["date"].isin(rebalance_20)
    frame["is_rebalance_40"] = frame["date"].isin(rebalance_40)
    return frame


def rank_score(frame: pd.DataFrame, rank_by: str) -> pd.Series:
    if rank_by == "event_score":
        return frame["event_score"]
    if rank_by == "expected_return":
        return frame["expected_net_return_20d"]
    if rank_by == "positive_return":
        return frame["positive_return_score"]
    if rank_by == "event_x_positive":
        return frame["event_score"] * frame["positive_return_score"]
    if rank_by == "event_x_overlay":
        return frame["event_score"] * frame["return_overlay_score"].clip(lower=0)
    raise ValueError(rank_by)


def select_overlay_candidates(df: pd.DataFrame, spec: OverlaySpec) -> pd.DataFrame:
    eligible = apply_filter(df, spec.filter_name)
    eligible = eligible[eligible["adv_20d_dollars"].fillna(0).ge(100_000)]
    eligible = eligible[eligible["event_score"].ge(spec.event_threshold)]
    if spec.return_gate != "none":
        eligible = eligible[eligible["expected_net_return_20d"].ge(spec.return_threshold)]
    if spec.positive_gate != "none":
        eligible = eligible[eligible["positive_return_score"].ge(spec.positive_threshold)]
    if eligible.empty:
        return eligible
    rebalance_col = f"is_rebalance_{spec.max_hold}"
    if rebalance_col not in eligible.columns:
        raise ValueError(f"Unsupported max_hold: {spec.max_hold}")
    eligible = eligible[eligible[rebalance_col].fillna(False)].copy()
    if eligible.empty:
        return eligible
    eligible["rank_score"] = rank_score(eligible, spec.rank_by)
    eligible["rank"] = eligible.groupby("date")["rank_score"].rank(ascending=False, method="first")
    return eligible[eligible["rank"].le(spec.top_k)].copy()


def summarize_trades(trades: pd.DataFrame, spec: OverlaySpec) -> dict[str, object]:
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
        "announcement_exit_rate": float(trades["exit_kind"].astype(str).str.startswith("announcement_peak").mean()),
        "event_trade_rate": float(trades["event_20d"].fillna(0).eq(1).mean()),
    }
    if len(trades) > 1:
        trimmed = trades.drop(index=trades["net_return"].idxmax())
        out["mean_ex_best"] = float(trimmed["net_return"].mean())
        out["median_ex_best"] = float(trimmed["net_return"].median())
    else:
        out["mean_ex_best"] = np.nan
        out["median_ex_best"] = np.nan
    return out


def backtest_overlay_spec(
    df: pd.DataFrame,
    spec: OverlaySpec,
    events_by_ticker: dict[str, pd.Series],
    price_dirs: list[Path],
    cost_bps: float,
    price_cache: dict[str, pd.DataFrame],
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]],
) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = select_overlay_candidates(df, spec)
    if candidates.empty:
        return {"trades": 0, **spec.__dict__}, candidates
    exits = [
        compute_trade_exit(row, events_by_ticker, price_cache, exit_cache, price_dirs, spec.max_hold, cost_bps)
        for _, row in candidates.iterrows()
    ]
    trades = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(exits)], axis=1)
    trades = trades.dropna(subset=["net_return"]).copy()
    return summarize_trades(trades, spec), trades


def make_specs(tune_frames: dict[str, pd.DataFrame]) -> list[OverlaySpec]:
    event_quantiles = [0.80, 0.90, 0.95, 0.98]
    return_gates = ["p50", "p70", "p85"]
    positive_gates = ["none", "p50"]
    rank_bys = ["expected_return", "event_x_positive", "event_x_overlay"]
    filters = ["all", "away_high", "no_big_runup", "no_sec30", "quiet_not_priced"]
    specs: list[OverlaySpec] = []
    for event_model, tune in tune_frames.items():
        event_thresholds = {q: float(tune["event_score"].quantile(q)) for q in event_quantiles}
        return_thresholds = {
            "none": -np.inf,
            "p50": float(tune["expected_net_return_20d"].quantile(0.50)),
            "p70": float(tune["expected_net_return_20d"].quantile(0.70)),
            "p85": float(tune["expected_net_return_20d"].quantile(0.85)),
            "p90": float(tune["expected_net_return_20d"].quantile(0.90)),
        }
        positive_thresholds = {
            "none": -np.inf,
            "p50": float(tune["positive_return_score"].quantile(0.50)),
            "p70": float(tune["positive_return_score"].quantile(0.70)),
        }
        for event_q, event_threshold in event_thresholds.items():
            for return_gate, return_threshold in return_thresholds.items():
                for positive_gate, positive_threshold in positive_thresholds.items():
                    if return_gate == "none" and positive_gate == "none":
                        continue
                    for rank_by in rank_bys:
                        for max_hold in (20, 40):
                            for filter_name in filters:
                                specs.append(
                                    OverlaySpec(
                                        event_model=event_model,
                                        event_quantile=event_q,
                                        event_threshold=event_threshold,
                                        return_gate=return_gate,
                                        return_threshold=return_threshold,
                                        positive_gate=positive_gate,
                                        positive_threshold=positive_threshold,
                                        rank_by=rank_by,
                                        top_k=10,
                                        max_hold=max_hold,
                                        filter_name=filter_name,
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
        "announcement_exit_rate",
        "event_trade_rate",
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
        out["tune_trades"].ge(20)
        & out["confirm_trades"].ge(20)
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
            "event_quantile",
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
    validation_scores, test_scores = score_return_models(run_dir, out_dir, cost_bps)
    event_models = ["rf", "hgb"]
    validation_frames = {
        model: load_overlay_frame(run_dir, validation_scores, model, "validation")
        for model in event_models
    }
    test_frames = {
        model: load_overlay_frame(run_dir, test_scores, model, "test")
        for model in event_models
    }
    tune_cutoff = pd.Timestamp(tune_end)
    tune_frames = {model: frame[frame["date"].lt(tune_cutoff)].copy() for model, frame in validation_frames.items()}
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}
    price_cache: dict[str, pd.DataFrame] = {}
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] = {}
    rows = []
    test_trades_by_row: dict[int, pd.DataFrame] = {}

    for spec in make_specs(tune_frames):
        val = validation_frames[spec.event_model]
        tune = val[val["date"].lt(tune_cutoff)].copy()
        confirm = val[val["date"].ge(tune_cutoff)].copy()
        tune_summary, _ = backtest_overlay_spec(
            tune, spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        if tune_summary.get("trades", 0) < 8:
            continue
        confirm_summary, _ = backtest_overlay_spec(
            confirm, spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        if confirm_summary.get("trades", 0) < 8:
            continue
        test_summary, test_trades = backtest_overlay_spec(
            test_frames[spec.event_model], spec, events_by_ticker, price_dirs, cost_bps, price_cache, exit_cache
        )
        row = {**spec.__dict__, **prefixed(tune_summary, "tune"), **prefixed(confirm_summary, "confirm"), **prefixed(test_summary, "test")}
        rows.append(row)
        test_trades_by_row[len(rows) - 1] = test_trades

    grid = pd.DataFrame(rows)
    grid.to_csv(out_dir / "return_overlay_validation_test_grid.csv", index=False)
    if grid.empty:
        (out_dir / "REPORT.md").write_text("No return-overlay strategies met the minimum validation trade count.")
        return

    deployable = select_deployable(grid)
    deployable.to_csv(out_dir / "return_overlay_deployable_candidates.csv", index=False)
    selected = deployable.head(1).copy()
    if not selected.empty:
        selected.to_csv(out_dir / "selected_return_overlay_strategy.csv", index=False)
        selected_idx = int(selected.index[0])
        test_trades_by_row[selected_idx].to_csv(out_dir / "selected_return_overlay_test_trades.csv", index=False)

    cols = [
        "event_model",
        "event_quantile",
        "event_threshold",
        "return_gate",
        "return_threshold",
        "positive_gate",
        "positive_threshold",
        "rank_by",
        "top_k",
        "max_hold",
        "filter_name",
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
        "test_event_trade_rate",
    ]
    selected_md = (
        selected[[c for c in cols if c in selected.columns]].to_markdown(index=False, floatfmt=".4f")
        if not selected.empty
        else "_No return-overlay strategy met the deployable gates._"
    )
    top_md = (
        deployable[[c for c in cols if c in deployable.columns]].head(25).to_markdown(index=False, floatfmt=".4f")
        if not deployable.empty
        else "_No rows._"
    )
    diagnostic = grid[
        grid["tune_total_return"].gt(0)
        & grid["confirm_total_return"].gt(0)
        & grid["test_total_return"].gt(0)
    ].copy()
    diagnostic_md = (
        diagnostic.sort_values("test_total_return", ascending=False)[[c for c in cols if c in diagnostic.columns]]
        .head(20)
        .to_markdown(index=False, floatfmt=".4f")
        if not diagnostic.empty
        else "_No rows._"
    )
    report = [
        "# Return Overlay Strategy",
        "",
        f"Source run: `{run_dir}`",
        "",
        "The return overlay is trained only on the pre-2026 train split. Event and return thresholds are derived from the validation tune slice before confirm/test evaluation.",
        "",
        "Exit rule: buy next open after a selected signal; sell at audited announcement 24-hour daily-high proxy if an announcement occurs before the time stop, otherwise sell at the fixed time-stop close. Daily-high announcement exits remain optimistic.",
        "",
        "## Selected Return-Overlay Strategy",
        selected_md,
        "",
        "## Top Deployable Candidates",
        top_md,
        "",
        "## Positive Test Diagnostics",
        "Diagnostic only: these rows are sorted by test result and are not the selection rule.",
        "",
        diagnostic_md,
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full_strategy_review")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/return_overlay_strategy",
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
