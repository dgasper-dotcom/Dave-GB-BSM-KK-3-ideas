from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .early_signal_strategy import (
    StrategySpec,
    backtest_spec,
    load_events,
    select_candidates,
)
from .profitability_research import MARKET_FEATURES
from .sec_market_fusion import sec_feature_columns
from .two_sleeve_return_strategy import SleeveSpec, summarize_portfolio


DEFAULT_SLEEVES = [
    SleeveSpec(
        name="unpriced_event_probability",
        model="hgb",
        threshold_quantile=0.98,
        top_k=10,
        max_hold=20,
        filter_name="away_high",
        thesis="High M&A score, but price remains at least 5% below its 52-week high.",
    ),
    SleeveSpec(
        name="stale_sec_repricing",
        model="hgb",
        threshold_quantile=0.95,
        top_k=10,
        max_hold=20,
        filter_name="stale_sec_not_recent",
        thesis="M&A language appeared in the last 90 days, but not the last 30, with no large run-up.",
    ),
]


def load_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, parse_dates=["date", "event_date"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def quarter_folds(panel: pd.DataFrame, start: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    first = pd.Timestamp(start)
    last = pd.Timestamp(panel["date"].max()).normalize() + pd.Timedelta(days=1)
    starts = list(pd.date_range(first, last, freq="QS"))
    if not starts or starts[0] != first:
        starts = [first] + starts
    folds = []
    for fold_start in starts:
        fold_end = fold_start + pd.DateOffset(months=3)
        if fold_start >= last:
            continue
        folds.append((fold_start, min(pd.Timestamp(fold_end), last)))
    return folds


def score_frame(model, frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    scored = frame.copy()
    scored["score"] = model.predict_proba(scored[features])[:, 1]
    return scored


def make_walk_forward_model(max_iter: int) -> Pipeline:
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


def executable_time_stop_backtest(
    scored: pd.DataFrame,
    spec: StrategySpec,
    cost_bps: float,
) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = select_candidates(scored, spec)
    if candidates.empty:
        return {"trades": 0, **spec.__dict__}, candidates
    trades = candidates.dropna(subset=["fwd_return_20d"]).copy()
    if trades.empty:
        return {"trades": 0, **spec.__dict__}, trades
    trades["entry_date"] = trades["date"] + pd.offsets.BDay(1)
    trades["exit_kind"] = "time_stop_20d_executable"
    trades["gross_return"] = trades["fwd_return_20d"]
    trades["net_return"] = trades["gross_return"] - cost_bps / 10000.0
    period_returns = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
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
        "announcement_exit_rate": 0.0,
    }
    if len(trades) > 1:
        trimmed = trades.drop(index=trades["net_return"].idxmax())
        summary["mean_ex_best"] = float(trimmed["net_return"].mean())
        summary["median_ex_best"] = float(trimmed["net_return"].median())
    else:
        summary["mean_ex_best"] = np.nan
        summary["median_ex_best"] = np.nan
    return summary, trades


def flatten_summary(summary: dict[str, object], prefix: str) -> dict[str, object]:
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


def summarize_unique_ticker_portfolio(trades: pd.DataFrame) -> dict[str, object]:
    if trades.empty:
        return {
            "trades": 0,
            "tickers": 0,
            "periods": 0,
            "total_return": 0.0,
            "max_drawdown": 0.0,
            "median_trade_return": np.nan,
            "win_rate": np.nan,
            "mean_ex_best": np.nan,
        }
    unique = trades.sort_values(["date", "ticker", "net_return"]).drop_duplicates(["date", "ticker"], keep="last")
    period_returns = unique.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    summary = {
        "trades": int(len(unique)),
        "tickers": int(unique["ticker"].nunique()),
        "periods": int(len(period_returns)),
        "total_return": float(equity.iloc[-1] - 1),
        "max_drawdown": float(drawdown.min()),
        "mean_period_return": float(period_returns.mean()),
        "median_period_return": float(period_returns.median()),
        "positive_period_rate": float(period_returns.gt(0).mean()),
        "mean_trade_return": float(unique["net_return"].mean()),
        "median_trade_return": float(unique["net_return"].median()),
        "win_rate": float(unique["net_return"].gt(0).mean()),
        "event_trade_rate": float(unique["event_20d"].fillna(0).eq(1).mean()),
    }
    if len(unique) > 1:
        summary["mean_ex_best"] = float(unique.drop(index=unique["net_return"].idxmax())["net_return"].mean())
    else:
        summary["mean_ex_best"] = np.nan
    return summary


def robustness_rows(trades: pd.DataFrame, mode: str) -> list[dict[str, object]]:
    if trades.empty:
        return []
    rows = []
    unique = trades.sort_values(["date", "ticker", "net_return"]).drop_duplicates(["date", "ticker"], keep="last")
    for n in (1, 2, 3, 5, 10):
        cut = unique.drop(index=unique.sort_values("net_return", ascending=False).head(n).index)
        if cut.empty:
            continue
        period_returns = cut.groupby("date")["net_return"].mean().sort_index()
        rows.append(
            {
                "mode": mode,
                "test": f"dedup_ex_top_{n}_trades",
                "periods": int(len(period_returns)),
                "trades": int(len(cut)),
                "total_return": float((1 + period_returns).prod() - 1),
                "median_period_return": float(period_returns.median()),
            }
        )
    period_returns = unique.groupby("date")["net_return"].mean().sort_index()
    for n in (1, 2, 3):
        cut = period_returns.sort_values(ascending=False).iloc[n:]
        if cut.empty:
            continue
        rows.append(
            {
                "mode": mode,
                "test": f"dedup_ex_top_{n}_periods",
                "periods": int(len(cut)),
                "trades": np.nan,
                "total_return": float((1 + cut).prod() - 1),
                "median_period_return": float(cut.median()),
            }
        )
    return rows


def table(frame: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    out = frame[[c for c in cols if c in frame.columns]]
    if max_rows:
        out = out.head(max_rows)
    return out.to_markdown(index=False, floatfmt=".4f")


def run(
    panel_path: Path,
    events_path: Path,
    price_dirs: list[Path],
    out_dir: Path,
    start: str,
    target: str,
    cost_bps: float,
    min_train_event_rows: int,
    include_announcement_proxy: bool,
    max_iter: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(panel_path)
    features = MARKET_FEATURES + sec_feature_columns(panel)
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}
    price_cache: dict[str, pd.DataFrame] = {}
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] = {}

    fold_rows = []
    sleeve_rows = []
    modes = ["executable_20d_time_stop"]
    if include_announcement_proxy:
        modes.insert(0, "announcement_peak_proxy")
    all_trades_by_mode: dict[str, dict[str, list[pd.DataFrame]]] = {
        mode: {sleeve.name: [] for sleeve in DEFAULT_SLEEVES} for mode in modes
    }

    folds = quarter_folds(panel, start)
    for fold_number, (fold_start, fold_end) in enumerate(folds, start=1):
        train = panel[panel["date"].lt(fold_start)].dropna(subset=[target]).copy()
        test = panel[panel["date"].ge(fold_start) & panel["date"].lt(fold_end)].copy()
        event_rows = int(train[target].sum())
        print(
            f"fold={fold_number}/{len(folds)} start={fold_start.date()} end={fold_end.date()} "
            f"train_rows={len(train)} test_rows={len(test)} train_event_rows={event_rows}",
            flush=True,
        )
        if train.empty or test.empty or train[target].nunique() < 2 or event_rows < min_train_event_rows:
            fold_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "mode": "skipped",
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": event_rows,
                }
            )
            continue

        model = make_walk_forward_model(max_iter)
        model.fit(train[features], train[target].astype(int))
        calibration = train[train["date"].ge(fold_start - pd.Timedelta(days=365))].copy()
        if calibration[target].nunique() < 2:
            calibration = train
        calibration_scores = model.predict_proba(calibration[features])[:, 1]
        scored = score_frame(model, test, features)

        fold_trades_by_mode: dict[str, dict[str, pd.DataFrame]] = {mode: {} for mode in modes}
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
            executable_summary, executable_trades = executable_time_stop_backtest(scored, spec, cost_bps)
            mode_results = [("executable_20d_time_stop", executable_summary, executable_trades)]
            if include_announcement_proxy:
                research_summary, research_trades = backtest_spec(
                    scored,
                    spec,
                    events_by_ticker,
                    price_dirs,
                    cost_bps,
                    price_cache=price_cache,
                    exit_cache=exit_cache,
                )
                mode_results.insert(0, ("announcement_peak_proxy", research_summary, research_trades))
            for mode, summary, trades in mode_results:
                tagged = trades.assign(
                    fold_start=fold_start.date(),
                    fold_end=fold_end.date(),
                    sleeve=sleeve.name,
                    mode=mode,
                    threshold=threshold,
                )
                fold_trades_by_mode[mode][sleeve.name] = tagged
                if not tagged.empty:
                    all_trades_by_mode[mode][sleeve.name].append(tagged)
                sleeve_rows.append(
                    {
                        "fold_start": fold_start.date(),
                        "fold_end": fold_end.date(),
                        "mode": mode,
                        "sleeve": sleeve.name,
                        "thesis": sleeve.thesis,
                        "threshold": threshold,
                        "train_rows": len(train),
                        "test_rows": len(test),
                        "train_event_rows": event_rows,
                        **summary,
                    }
                )

        for mode, trades_by_sleeve in fold_trades_by_mode.items():
            portfolio_summary, _ = summarize_portfolio(trades_by_sleeve)
            fold_rows.append(
                {
                    "fold_start": fold_start.date(),
                    "fold_end": fold_end.date(),
                    "mode": mode,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_event_rows": event_rows,
                    **portfolio_summary,
                }
            )

    fold_summary = pd.DataFrame(fold_rows)
    sleeve_summary = pd.DataFrame(sleeve_rows)
    fold_summary.to_csv(out_dir / "walk_forward_fold_summary.csv", index=False)
    sleeve_summary.to_csv(out_dir / "walk_forward_sleeve_summary.csv", index=False)

    overall_rows = []
    unique_rows = []
    robust_rows = []
    trade_frames = []
    period_frames = []
    for mode, sleeve_map in all_trades_by_mode.items():
        trades_by_sleeve = {
            sleeve: pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            for sleeve, frames in sleeve_map.items()
        }
        summary, periods = summarize_portfolio(trades_by_sleeve)
        overall_rows.append({"mode": mode, **summary})
        nonempty_trades = [trades for trades in trades_by_sleeve.values() if not trades.empty]
        mode_trades = pd.concat(nonempty_trades, ignore_index=True) if nonempty_trades else pd.DataFrame()
        unique_rows.append({"mode": mode, **summarize_unique_ticker_portfolio(mode_trades)})
        robust_rows.extend(robustness_rows(mode_trades, mode))
        if not periods.empty:
            periods.insert(0, "mode", mode)
            period_frames.append(periods)
        for sleeve, trades in trades_by_sleeve.items():
            if not trades.empty:
                trade_frames.append(trades.assign(mode=mode, sleeve=sleeve))
    overall = pd.DataFrame(overall_rows)
    unique_overall = pd.DataFrame(unique_rows)
    robustness = pd.DataFrame(robust_rows)
    overall.to_csv(out_dir / "walk_forward_overall_summary.csv", index=False)
    unique_overall.to_csv(out_dir / "walk_forward_unique_ticker_summary.csv", index=False)
    robustness.to_csv(out_dir / "walk_forward_robustness_summary.csv", index=False)
    if trade_frames:
        pd.concat(trade_frames, ignore_index=True).to_csv(out_dir / "walk_forward_trades.csv", index=False)
    if period_frames:
        pd.concat(period_frames, ignore_index=True).to_csv(out_dir / "walk_forward_period_returns.csv", index=False)

    fold_cols = [
        "fold_start",
        "fold_end",
        "mode",
        "trades",
        "total_return",
        "max_drawdown",
        "median_trade_return",
        "win_rate",
        "mean_ex_best",
        "event_trade_rate",
        "announcement_exit_rate",
    ]
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
        "announcement_exit_rate",
    ]
    unique_cols = [
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
    robustness_cols = ["mode", "test", "periods", "trades", "total_return", "median_period_return"]
    sleeve_cols = [
        "mode",
        "sleeve",
        "trades",
        "total_return",
        "max_drawdown",
        "median_trade_return",
        "win_rate",
        "mean_ex_best",
        "event_trade_rate",
        "announcement_exit_rate",
    ]
    sleeve_overall = []
    for mode, sleeve_map in all_trades_by_mode.items():
        for sleeve, frames in sleeve_map.items():
            trades = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
            summary, _ = summarize_portfolio({sleeve: trades})
            sleeve_overall.append({"mode": mode, "sleeve": sleeve, **summary})
    sleeve_overall_frame = pd.DataFrame(sleeve_overall)
    sleeve_overall_frame.to_csv(out_dir / "walk_forward_sleeve_overall_summary.csv", index=False)

    report = [
        "# Larger Walk-Forward Return Backtest",
        "",
        f"Panel: `{panel_path}`",
        f"Date range: `{panel['date'].min().date()}` to `{panel['date'].max().date()}`",
        f"Tickers: `{panel['ticker'].nunique()}`; rows: `{len(panel)}`; target: `{target}`",
        "",
        "Each fold retrains the HGB model using only data before the fold, recalibrates sleeve thresholds from prior scores, then trades the next calendar quarter.",
        "",
        "The `executable_20d_time_stop` mode buys the next open and exits at the 20-trading-day close, with no announcement high. Use `--include-announcement-proxy` to also run the slower research-only announcement-peak exit.",
        "",
        "## Overall",
        table(overall, overall_cols),
        "",
        "## Unique-Ticker Overall",
        "This view caps a ticker at one position per rebalance date even if both sleeves select it.",
        "",
        table(unique_overall, unique_cols),
        "",
        "## Tail Robustness",
        "These checks use the unique-ticker portfolio and remove the largest winners.",
        "",
        table(robustness, robustness_cols),
        "",
        "## Sleeve Overall",
        table(sleeve_overall_frame, sleeve_cols),
        "",
        "## Fold Results",
        table(fold_summary[fold_summary["mode"].ne("skipped")], fold_cols),
        "",
        "## Skipped Folds",
        table(fold_summary[fold_summary["mode"].eq("skipped")], ["fold_start", "fold_end", "train_rows", "test_rows", "train_event_rows"]),
        "",
        "## Caveats",
        "",
        "This is larger than the prior holdout, but still limited by the local audited-event dataset and by historical SEC coverage beginning in 2023. The executable result is the more relevant number for a tradable strategy.",
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--panel",
        default="reports/profitability_sec_market_fusion_full_strategy_review/fused_market_sec_panel.csv",
    )
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/walk_forward_return_backtest",
    )
    parser.add_argument("--price-dir", action="append", default=["data/raw/stockanalysis_prices", "data/raw/yfinance"])
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--target", default="event_20d")
    parser.add_argument("--cost-bps", type=float, default=250.0)
    parser.add_argument("--min-train-event-rows", type=int, default=100)
    parser.add_argument("--include-announcement-proxy", action="store_true")
    parser.add_argument("--max-iter", type=int, default=60)
    args = parser.parse_args()
    run(
        panel_path=Path(args.panel),
        events_path=Path(args.events),
        price_dirs=[Path(p) for p in args.price_dir],
        out_dir=Path(args.out_dir),
        start=args.start,
        target=args.target,
        cost_bps=args.cost_bps,
        min_train_event_rows=args.min_train_event_rows,
        include_announcement_proxy=args.include_announcement_proxy,
        max_iter=args.max_iter,
    )


if __name__ == "__main__":
    main()
