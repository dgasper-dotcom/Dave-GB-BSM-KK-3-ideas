from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .early_signal_strategy import (
    FILTER_NAMES,
    StrategySpec,
    backtest_spec,
    load_events,
    load_predictions,
)


def strategy_specs(validation: pd.DataFrame) -> list[StrategySpec]:
    models = ["logistic", "rf", "hgb"]
    modes = ["nonoverlap", "first_signal"]
    quantiles = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    top_ks = [3, 5, 10, 20]
    max_holds = [20, 40, 60]
    filters = FILTER_NAMES
    specs = []
    for model in models:
        model_scores = validation[model]["score"]
        thresholds = {q: float(model_scores.quantile(q)) for q in quantiles}
        for mode in modes:
            for q, threshold in thresholds.items():
                for max_hold in max_holds:
                    for filter_name in filters:
                        candidate_top_ks = top_ks if mode == "nonoverlap" else [9999]
                        for top_k in candidate_top_ks:
                            specs.append(
                                StrategySpec(
                                    mode=mode,
                                    model=model,
                                    threshold_quantile=q,
                                    threshold=threshold,
                                    top_k=top_k,
                                    max_hold=max_hold,
                                    filter_name=filter_name,
                                )
                            )
    return specs


def prefixed(summary: dict[str, object], prefix: str) -> dict[str, object]:
    keep = [
        "trades",
        "tickers",
        "active_signal_dates",
        "mean_trade_return",
        "median_trade_return",
        "win_rate",
        "total_return",
        "max_drawdown",
        "announcement_exit_rate",
        "mean_ex_best",
        "median_ex_best",
    ]
    return {f"{prefix}_{key}": summary.get(key) for key in keep}


def add_validation_selection_scores(grid: pd.DataFrame) -> pd.DataFrame:
    out = grid.copy()
    out["min_validation_total_return"] = out[["tune_total_return", "confirm_total_return"]].min(axis=1)
    out["max_validation_total_return"] = out[["tune_total_return", "confirm_total_return"]].max(axis=1)
    out["min_validation_win_rate"] = out[["tune_win_rate", "confirm_win_rate"]].min(axis=1)
    out["min_validation_mean_ex_best"] = out[["tune_mean_ex_best", "confirm_mean_ex_best"]].min(axis=1)
    out["min_validation_trades"] = out[["tune_trades", "confirm_trades"]].min(axis=1)
    out["validation_drawdown_floor"] = out[["tune_max_drawdown", "confirm_max_drawdown"]].min(axis=1)
    return out


def rank_validation_only(candidates: pd.DataFrame) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "min_validation_total_return",
            "median_floor",
            "min_validation_mean_ex_best",
            "confirm_total_return",
            "tune_total_return",
            "min_validation_trades",
        ],
        ascending=[False, False, False, False, False, False],
    )


def deployable_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    candidates = grid[
        grid["mode"].eq("nonoverlap")
        & grid["top_k"].le(10)
        & grid["max_hold"].eq(20)
        & grid["tune_trades"].ge(25)
        & grid["confirm_trades"].ge(25)
        & grid["tune_total_return"].gt(0)
        & grid["confirm_total_return"].gt(0)
        & grid["tune_total_return"].lt(1.0)
        & grid["confirm_total_return"].lt(1.0)
        & grid["tune_mean_ex_best"].gt(0)
        & grid["confirm_mean_ex_best"].gt(0)
        & grid["confirm_median_trade_return"].gt(0)
        & grid["tune_win_rate"].ge(0.40)
        & grid["confirm_win_rate"].ge(0.50)
        & grid["tune_max_drawdown"].gt(-0.20)
        & grid["confirm_max_drawdown"].gt(-0.20)
    ].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        [
            "min_validation_mean_ex_best",
            "min_validation_total_return",
            "min_validation_trades",
            "median_floor",
            "threshold_quantile",
        ],
        ascending=[False, False, False, False, True],
    )


def strategic_review_candidates(grid: pd.DataFrame) -> pd.DataFrame:
    strategic_filter = grid["filter_name"].astype(str).str.contains(
        "strategic|adviser|control", regex=True, na=False
    )
    candidates = grid[
        strategic_filter
        & grid["mode"].eq("nonoverlap")
        & grid["tune_total_return"].gt(0)
        & grid["confirm_total_return"].gt(0)
        & grid["tune_mean_ex_best"].gt(0)
        & grid["confirm_mean_ex_best"].gt(0)
        & grid["tune_median_trade_return"].gt(0)
        & grid["confirm_median_trade_return"].gt(0)
        & grid["tune_win_rate"].ge(0.50)
        & grid["confirm_win_rate"].ge(0.50)
        & grid["tune_max_drawdown"].gt(-0.55)
        & grid["confirm_max_drawdown"].gt(-0.55)
    ].copy()
    return rank_validation_only(candidates)


def run(
    run_dir: Path,
    events_path: Path,
    price_dirs: list[Path],
    out_dir: Path,
    cost_bps: float,
    tune_end: str,
    min_tune_trades: int,
    min_confirm_trades: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}

    validation = {model: load_predictions(run_dir, model, "validation") for model in ["logistic", "rf", "hgb"]}
    test = {model: load_predictions(run_dir, model, "test") for model in ["logistic", "rf", "hgb"]}
    tune_cutoff = pd.Timestamp(tune_end)

    price_cache = {}
    exit_cache = {}
    tune_for_thresholds = {model: df[df["date"].lt(tune_cutoff)].copy() for model, df in validation.items()}
    rows = []
    selected_trades = {}
    for spec in strategy_specs(tune_for_thresholds):
        val_df = validation[spec.model]
        tune_df = val_df[val_df["date"].lt(tune_cutoff)].copy()
        confirm_df = val_df[val_df["date"].ge(tune_cutoff)].copy()
        tune_summary, _ = backtest_spec(
            tune_df,
            spec,
            events_by_ticker,
            price_dirs,
            cost_bps,
            price_cache=price_cache,
            exit_cache=exit_cache,
        )
        if tune_summary.get("trades", 0) < min_tune_trades:
            continue
        confirm_summary, _ = backtest_spec(
            confirm_df,
            spec,
            events_by_ticker,
            price_dirs,
            cost_bps,
            price_cache=price_cache,
            exit_cache=exit_cache,
        )
        if confirm_summary.get("trades", 0) < min_confirm_trades:
            continue
        test_summary, test_trades = backtest_spec(
            test[spec.model],
            spec,
            events_by_ticker,
            price_dirs,
            cost_bps,
            price_cache=price_cache,
            exit_cache=exit_cache,
        )
        row = {
            **spec.__dict__,
            **prefixed(tune_summary, "tune"),
            **prefixed(confirm_summary, "confirm"),
            **prefixed(test_summary, "test"),
        }
        row["robust_score"] = min(
            float(row.get("tune_total_return", -999)),
            float(row.get("confirm_total_return", -999)),
        )
        row["median_floor"] = min(
            float(row.get("tune_median_trade_return", -999)),
            float(row.get("confirm_median_trade_return", -999)),
        )
        rows.append(row)
        selected_trades[len(rows) - 1] = test_trades

    grid = add_validation_selection_scores(pd.DataFrame(rows))
    grid.to_csv(out_dir / "robust_validation_test_grid.csv", index=False)
    if grid.empty:
        (out_dir / "REPORT.md").write_text("No strategies met the internal validation trade-count gates.")
        return

    robust = grid[
        grid["tune_total_return"].gt(0)
        & grid["confirm_total_return"].gt(0)
        & grid["tune_mean_ex_best"].gt(0)
        & grid["confirm_mean_ex_best"].gt(0)
        & grid["tune_win_rate"].ge(0.45)
        & grid["confirm_win_rate"].ge(0.45)
        & grid["tune_max_drawdown"].gt(-0.55)
        & grid["confirm_max_drawdown"].gt(-0.55)
    ].copy()
    if robust.empty:
        robust = grid[
            grid["tune_total_return"].gt(0)
            & grid["confirm_total_return"].gt(0)
            & grid["tune_mean_ex_best"].gt(0)
            & grid["confirm_mean_ex_best"].gt(0)
        ].copy()
    if robust.empty:
        robust = grid.copy()

    robust = add_validation_selection_scores(robust)
    robust = robust.sort_values(
        ["robust_score", "median_floor", "confirm_total_return", "tune_total_return"],
        ascending=[False, False, False, False],
    )
    robust.to_csv(out_dir / "robust_candidates_ranked.csv", index=False)
    selected_broad = robust.head(1).copy()
    selected_broad.to_csv(out_dir / "selected_broad_robust_strategy.csv", index=False)
    selected_broad_idx = int(selected_broad.index[0])
    selected_trades[selected_broad_idx].to_csv(out_dir / "selected_broad_robust_test_trades.csv", index=False)

    deployable = deployable_candidates(grid)
    deployable.to_csv(out_dir / "deployable_candidates_ranked.csv", index=False)
    selected_deployable = deployable.head(1).copy()
    if not selected_deployable.empty:
        selected_deployable.to_csv(out_dir / "selected_deployable_strategy.csv", index=False)
        selected_deployable_idx = int(selected_deployable.index[0])
        selected_trades[selected_deployable_idx].to_csv(out_dir / "selected_deployable_test_trades.csv", index=False)

    strategic = strategic_review_candidates(grid)
    strategic.to_csv(out_dir / "strategic_review_candidates_ranked.csv", index=False)
    selected_strategic = strategic.head(1).copy()
    if not selected_strategic.empty:
        selected_strategic.to_csv(out_dir / "selected_strategic_review_strategy.csv", index=False)
        selected_strategic_idx = int(selected_strategic.index[0])
        selected_trades[selected_strategic_idx].to_csv(out_dir / "selected_strategic_review_test_trades.csv", index=False)

    report_cols = [
        "mode",
        "model",
        "threshold_quantile",
        "threshold",
        "top_k",
        "max_hold",
        "filter_name",
        "tune_trades",
        "tune_total_return",
        "tune_median_trade_return",
        "tune_win_rate",
        "tune_max_drawdown",
        "confirm_trades",
        "confirm_total_return",
        "confirm_median_trade_return",
        "confirm_win_rate",
        "confirm_max_drawdown",
        "test_trades",
        "test_total_return",
        "test_median_trade_return",
        "test_win_rate",
        "test_max_drawdown",
        "test_mean_ex_best",
    ]
    selected_deployable_md = (
        selected_deployable[[c for c in report_cols if c in selected_deployable.columns]].to_markdown(
            index=False, floatfmt=".4f"
        )
        if not selected_deployable.empty
        else "_No deployable strategy met the stricter gates._"
    )
    selected_strategic_md = (
        selected_strategic[[c for c in report_cols if c in selected_strategic.columns]].to_markdown(
            index=False, floatfmt=".4f"
        )
        if not selected_strategic.empty
        else "_No strategic-review strategy met the stricter strategic gates._"
    )
    top_deployable_md = (
        deployable[[c for c in report_cols if c in deployable.columns]].head(20).to_markdown(
            index=False, floatfmt=".4f"
        )
        if not deployable.empty
        else "_No rows._"
    )
    top_strategic_md = (
        strategic[[c for c in report_cols if c in strategic.columns]].head(20).to_markdown(
            index=False, floatfmt=".4f"
        )
        if not strategic.empty
        else "_No rows._"
    )
    report = [
        "# Robust Early Strategy Selection",
        "",
        f"Validation tune/confirm split: tune before `{tune_end}`, confirm on/after `{tune_end}`.",
        "",
        "The selected rule is chosen without using test performance. It must survive both validation slices before being applied to test.",
        "",
        "## Selected Deployable Strategy",
        "",
        "This selection is limited to 20-day non-overlap portfolios with `top_k <= 10`, at least 25 trades in each validation slice, positive tune/confirm returns after removing the best trade, capped validation compounding, and controlled validation drawdown.",
        "",
        selected_deployable_md,
        "",
        "## Selected Strategic-Review Strategy",
        "",
        "This selection only uses strategic-review-related filters. It is reported separately because those disclosures can be public enough to be partly priced in.",
        "",
        selected_strategic_md,
        "",
        "## Broad Robust Winner",
        "",
        "This is the highest broad validation ranking before the deployability gates. It is useful as an overfitting/tail-risk check.",
        "",
        selected_broad[[c for c in report_cols if c in selected_broad.columns]].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Deployable Candidates",
        top_deployable_md,
        "",
        "## Top Strategic-Review Candidates",
        top_strategic_md,
        "",
        "## Top Robust Candidates",
        robust[[c for c in report_cols if c in robust.columns]].head(20).to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument("--price-dir", action="append", default=["data/raw/stockanalysis_prices", "data/raw/yfinance"])
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion_full/robust_strategy_selection")
    parser.add_argument("--cost-bps", type=float, default=250.0)
    parser.add_argument("--tune-end", default="2026-04-01")
    parser.add_argument("--min-tune-trades", type=int, default=8)
    parser.add_argument("--min-confirm-trades", type=int, default=8)
    args = parser.parse_args()
    run(
        run_dir=Path(args.run_dir),
        events_path=Path(args.events),
        price_dirs=[Path(p) for p in args.price_dir],
        out_dir=Path(args.out_dir),
        cost_bps=args.cost_bps,
        tune_end=args.tune_end,
        min_tune_trades=args.min_tune_trades,
        min_confirm_trades=args.min_confirm_trades,
    )


if __name__ == "__main__":
    main()
