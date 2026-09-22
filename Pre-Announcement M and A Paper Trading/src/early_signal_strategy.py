from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

FEATURE_COLUMNS = [
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
    "sec_investment_bank_hit_30d",
    "sec_investment_bank_hit_90d",
    "sec_committee_hit_30d",
    "sec_committee_hit_90d",
    "sec_unsolicited_interest_hit_30d",
    "sec_unsolicited_interest_hit_90d",
    "sec_multiple_party_interest_hit_30d",
    "sec_multiple_party_interest_hit_90d",
    "sec_change_of_control_hit_30d",
    "sec_change_of_control_hit_90d",
    "sec_confidentiality_hit_30d",
    "sec_confidentiality_hit_90d",
    "sec_days_since_event_language",
]

FILTER_NAMES = [
    "all",
    "no_sec30",
    "away_high",
    "no_big_runup",
    "no_sec30_no_big_runup",
    "early_combo",
    "very_early_combo",
    "stale_sec_not_recent",
    "quiet_not_priced",
    "strategic_review_90",
    "strategic_review_not_tx30",
    "strategic_review_complex",
    "strategic_review_complex_early",
    "fresh_strategic_review_no_tx",
    "adviser_committee_review",
    "control_or_confidential_review",
    "strategic_review_interest",
    "strategic_review_interest_early",
    "strategic_review_strong_combo",
    "strategic_review_strong_early",
]


@dataclass(frozen=True)
class StrategySpec:
    mode: str
    model: str
    threshold_quantile: float
    threshold: float
    top_k: int
    max_hold: int
    filter_name: str


def load_events(events_path: Path) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    events = events[events["audited_announcement_ts"].notna()].copy()
    events["announcement_ts_et"] = pd.to_datetime(events["audited_announcement_ts"], utc=True).dt.tz_convert("America/New_York")
    events["announcement_date_et"] = events["announcement_ts_et"].dt.tz_localize(None).dt.normalize()
    keep = ["ticker", "event_type", "announcement_ts_et", "announcement_date_et", "audit_status", "audit_score"]
    return events[keep].drop_duplicates("ticker", keep="first")


def load_predictions(run_dir: Path, model: str, sample: str) -> pd.DataFrame:
    pred = pd.read_csv(run_dir / f"{model}_{sample}_predictions.csv", parse_dates=["date"])
    panel = pd.read_csv(
        run_dir / "fused_market_sec_panel.csv",
        usecols=lambda col: col in FEATURE_COLUMNS,
        parse_dates=["date"],
    )
    df = pred.merge(panel, on=["ticker", "date"], how="left")
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def load_prices(ticker: str, price_dirs: list[Path]) -> pd.DataFrame:
    for price_dir in price_dirs:
        path = price_dir / f"{ticker}.csv"
        if path.exists():
            prices = pd.read_csv(path, parse_dates=["date"])
            for col in ["open", "high", "low", "close"]:
                prices[col] = pd.to_numeric(prices[col], errors="coerce")
            prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
            return prices.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    return pd.DataFrame()


def daily_peak_window(announcement_ts_et: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    naive = announcement_ts_et.tz_localize(None)
    announce_date = naive.normalize()
    close_time = announce_date + pd.Timedelta(hours=MARKET_CLOSE_HOUR, minutes=MARKET_CLOSE_MINUTE)
    open_time = announce_date + pd.Timedelta(hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE)
    end_time = naive + pd.Timedelta(hours=24)
    if naive >= close_time:
        return naive, end_time, "announcement_peak_after_close"
    if naive < open_time:
        return naive, end_time, "announcement_peak_premarket"
    return naive, end_time, "announcement_peak_intraday_optimistic"


def daily_sessions_overlapping(prices: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    sessions = prices.copy()
    sessions["session_start"] = sessions["date"] + pd.Timedelta(hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE)
    sessions["session_end"] = sessions["date"] + pd.Timedelta(hours=MARKET_CLOSE_HOUR, minutes=MARKET_CLOSE_MINUTE)
    return sessions[(sessions["session_end"].ge(start_time)) & (sessions["session_start"].le(end_time))].copy()


def compute_trade_exit(
    row: pd.Series,
    events_by_ticker: dict[str, pd.Series],
    price_cache: dict[str, pd.DataFrame],
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]],
    price_dirs: list[Path],
    max_hold: int,
    cost_bps: float,
) -> dict[str, object]:
    ticker = row["ticker"]
    cache_key = (str(ticker), str(pd.Timestamp(row["date"]).date()), int(max_hold), round(float(row["next_open"]), 6))
    if cache_key in exit_cache:
        return exit_cache[cache_key].copy()
    if ticker not in price_cache:
        price_cache[ticker] = load_prices(ticker, price_dirs)
    prices = price_cache[ticker]
    if prices.empty or pd.isna(row["next_open"]):
        result = {"gross_return": np.nan, "net_return": np.nan, "exit_kind": "missing_price"}
        exit_cache[cache_key] = result
        return result.copy()

    signal_date = pd.Timestamp(row["date"]).normalize()
    matches = prices.index[prices["date"].eq(signal_date)].tolist()
    if not matches:
        result = {"gross_return": np.nan, "net_return": np.nan, "exit_kind": "missing_signal_date"}
        exit_cache[cache_key] = result
        return result.copy()
    signal_idx = matches[0]
    entry_idx = signal_idx + 1
    time_exit_idx = signal_idx + max_hold
    if entry_idx >= len(prices) or time_exit_idx >= len(prices):
        result = {"gross_return": np.nan, "net_return": np.nan, "exit_kind": "insufficient_forward_prices"}
        exit_cache[cache_key] = result
        return result.copy()

    entry_date = prices.loc[entry_idx, "date"]
    time_exit_date = prices.loc[time_exit_idx, "date"]
    exit_price = float(prices.loc[time_exit_idx, "close"])
    exit_date = time_exit_date
    exit_kind = f"time_stop_{max_hold}d"
    announcement_ts = pd.NaT

    event = events_by_ticker.get(ticker)
    if event is not None and pd.notna(event.get("announcement_ts_et")):
        announcement_ts = event["announcement_ts_et"]
        start_time, end_time, peak_note = daily_peak_window(announcement_ts)
        window = daily_sessions_overlapping(prices, start_time, end_time)
        window = window[(window["date"].ge(entry_date)) & (window["date"].le(time_exit_date))]
        if not window.empty:
            peak = window.loc[window["high"].idxmax()]
            exit_price = float(peak["high"])
            exit_date = peak["date"]
            exit_kind = peak_note

    gross = exit_price / float(row["next_open"]) - 1
    result = {
        "entry_date": entry_date,
        "exit_date": exit_date,
        "exit_kind": exit_kind,
        "exit_price": exit_price,
        "gross_return": float(gross),
        "net_return": float(gross - cost_bps / 10000.0),
        "announcement_ts_et": announcement_ts,
    }
    exit_cache[cache_key] = result
    return result.copy()


def apply_filter(df: pd.DataFrame, filter_name: str) -> pd.DataFrame:
    out = df.copy()
    no_sec30 = out["sec_any_event_language_hit_30d"].fillna(0).eq(0)
    no_sec90 = out["sec_any_event_language_hit_90d"].fillna(0).eq(0)
    stale_sec = out["sec_any_event_language_hit_90d"].fillna(0).gt(0) & no_sec30
    strategic_review_90 = out["sec_strategic_alternatives_hit_90d"].fillna(0).gt(0)
    strategic_review_30 = out["sec_strategic_alternatives_hit_30d"].fillna(0).gt(0)
    adviser_90 = out["sec_investment_bank_hit_90d"].fillna(0).gt(0)
    committee_90 = out["sec_committee_hit_90d"].fillna(0).gt(0)
    unsolicited_90 = out["sec_unsolicited_interest_hit_90d"].fillna(0).gt(0)
    multi_party_90 = out["sec_multiple_party_interest_hit_90d"].fillna(0).gt(0)
    change_control_90 = out["sec_change_of_control_hit_90d"].fillna(0).gt(0)
    confidentiality_90 = out["sec_confidentiality_hit_90d"].fillna(0).gt(0)
    transaction_30 = out["sec_transaction_hit_30d"].fillna(0).gt(0)
    transaction_90 = out["sec_transaction_hit_90d"].fillna(0).gt(0)
    review_complex = strategic_review_90 & (adviser_90 | committee_90 | change_control_90 | confidentiality_90)
    review_interest = strategic_review_90 & (unsolicited_90 | multi_party_90)
    review_strong_cue_count = (
        adviser_90.astype(int)
        + committee_90.astype(int)
        + unsolicited_90.astype(int)
        + multi_party_90.astype(int)
        + change_control_90.astype(int)
        + confidentiality_90.astype(int)
    )
    review_strong_combo = strategic_review_90 & review_strong_cue_count.ge(2)
    no_big_runup = out["return_60d"].fillna(0).lt(0.25)
    no_modest_runup = out["return_60d"].fillna(0).lt(0.15)
    away_high = out["distance_from_52w_high"].fillna(-1).lt(-0.05)
    deep_away_high = out["distance_from_52w_high"].fillna(-1).lt(-0.10)
    liquid_volume_not_spiking = out["volume_to_20d_avg"].fillna(1).lt(3.0)

    masks = {
        "all": pd.Series(True, index=out.index),
        "no_sec30": no_sec30,
        "away_high": away_high,
        "no_big_runup": no_big_runup,
        "no_sec30_no_big_runup": no_sec30 & no_big_runup,
        "early_combo": no_sec30 & no_big_runup & away_high,
        "very_early_combo": no_sec90 & no_modest_runup & deep_away_high,
        "stale_sec_not_recent": stale_sec & no_big_runup,
        "quiet_not_priced": no_sec30 & no_big_runup & away_high & liquid_volume_not_spiking,
        "strategic_review_90": strategic_review_90,
        "strategic_review_not_tx30": strategic_review_90 & ~transaction_30,
        "strategic_review_complex": review_complex,
        "strategic_review_complex_early": review_complex & ~transaction_30 & no_big_runup & away_high,
        "fresh_strategic_review_no_tx": strategic_review_30 & ~transaction_30,
        "adviser_committee_review": strategic_review_90 & (adviser_90 | committee_90) & ~transaction_30,
        "control_or_confidential_review": strategic_review_90 & (change_control_90 | confidentiality_90) & ~transaction_90,
        "strategic_review_interest": review_interest,
        "strategic_review_interest_early": review_interest & ~transaction_30 & no_big_runup & away_high,
        "strategic_review_strong_combo": review_strong_combo,
        "strategic_review_strong_early": review_strong_combo & ~transaction_30 & no_big_runup & away_high,
    }
    if filter_name not in masks:
        raise ValueError(filter_name)
    return out[masks[filter_name]].copy()


def select_candidates(df: pd.DataFrame, spec: StrategySpec) -> pd.DataFrame:
    eligible = apply_filter(df, spec.filter_name)
    eligible = eligible[eligible["adv_20d_dollars"].fillna(0).ge(100_000)]
    eligible = eligible[eligible["score"].ge(spec.threshold)].copy()
    if eligible.empty:
        return eligible
    if spec.mode == "first_signal":
        return eligible.sort_values(["ticker", "date", "score"], ascending=[True, True, False]).groupby("ticker", as_index=False).head(1)
    if spec.mode == "nonoverlap":
        all_dates = pd.Index(sorted(pd.to_datetime(df["date"]).dropna().unique()))
        rebalance_dates = set(all_dates[:: spec.max_hold])
        eligible = eligible[eligible["date"].isin(rebalance_dates)].copy()
        if eligible.empty:
            return eligible
        eligible["rank"] = eligible.groupby("date")["score"].rank(ascending=False, method="first")
        return eligible[eligible["rank"].le(spec.top_k)].copy()
    raise ValueError(spec.mode)


def backtest_spec(
    df: pd.DataFrame,
    spec: StrategySpec,
    events_by_ticker: dict[str, pd.Series],
    price_dirs: list[Path],
    cost_bps: float,
    price_cache: dict[str, pd.DataFrame] | None = None,
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    candidates = select_candidates(df, spec)
    if candidates.empty:
        return {"trades": 0, **spec.__dict__}, candidates

    if price_cache is None:
        price_cache = {}
    if exit_cache is None:
        exit_cache = {}
    exits = [
        compute_trade_exit(row, events_by_ticker, price_cache, exit_cache, price_dirs, spec.max_hold, cost_bps)
        for _, row in candidates.iterrows()
    ]
    trades = pd.concat([candidates.reset_index(drop=True), pd.DataFrame(exits)], axis=1)
    trades = trades.dropna(subset=["net_return"]).copy()
    if trades.empty:
        return {"trades": 0, **spec.__dict__}, trades

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
        "announcement_exit_rate": float(trades["exit_kind"].astype(str).str.startswith("announcement_peak").mean()),
    }
    if len(trades) > 1:
        trimmed = trades.drop(index=trades["net_return"].idxmax())
        summary["mean_ex_best"] = float(trimmed["net_return"].mean())
        summary["median_ex_best"] = float(trimmed["net_return"].median())
    else:
        summary["mean_ex_best"] = np.nan
        summary["median_ex_best"] = np.nan
    return summary, trades


def run_grid(
    run_dir: Path,
    events_path: Path,
    price_dirs: list[Path],
    out_dir: Path,
    cost_bps: float,
    min_validation_trades: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = load_events(events_path)
    events_by_ticker = {row["ticker"]: row for _, row in events.iterrows()}

    models = ["logistic", "rf", "hgb"]
    modes = ["nonoverlap", "first_signal"]
    quantiles = [0.50, 0.70, 0.80, 0.90, 0.95, 0.98]
    top_ks = [3, 5, 10, 20]
    max_holds = [20, 40, 60]
    filters = FILTER_NAMES

    validation_frames = {model: load_predictions(run_dir, model, "validation") for model in models}
    test_frames = {model: load_predictions(run_dir, model, "test") for model in models}

    validation_rows = []
    price_cache: dict[str, pd.DataFrame] = {}
    exit_cache: dict[tuple[str, str, int, float], dict[str, object]] = {}
    for model, val_df in validation_frames.items():
        thresholds = {q: float(val_df["score"].quantile(q)) for q in quantiles}
        for mode in modes:
            for q, threshold in thresholds.items():
                for max_hold in max_holds:
                    for filter_name in filters:
                        candidate_top_ks = top_ks if mode == "nonoverlap" else [9999]
                        for top_k in candidate_top_ks:
                            spec = StrategySpec(
                                mode=mode,
                                model=model,
                                threshold_quantile=q,
                                threshold=threshold,
                                top_k=top_k,
                                max_hold=max_hold,
                                filter_name=filter_name,
                            )
                            summary, _ = backtest_spec(
                                val_df,
                                spec,
                                events_by_ticker,
                                price_dirs,
                                cost_bps,
                                price_cache=price_cache,
                                exit_cache=exit_cache,
                            )
                            summary["sample"] = "validation"
                            validation_rows.append(summary)

    validation_grid = pd.DataFrame(validation_rows)
    validation_grid.to_csv(out_dir / "validation_strategy_grid.csv", index=False)
    eligible = validation_grid[
        validation_grid["trades"].ge(min_validation_trades)
        & validation_grid["median_trade_return"].gt(0)
        & validation_grid["mean_ex_best"].gt(0)
    ].copy()
    if eligible.empty:
        eligible = validation_grid[validation_grid["trades"].ge(min_validation_trades)].copy()
    best = eligible.sort_values(
        ["total_return", "median_trade_return", "mean_ex_best"],
        ascending=[False, False, False],
    ).head(1)

    if best.empty:
        (out_dir / "REPORT.md").write_text("No eligible validation strategies.")
        return

    best_row = best.iloc[0]
    best_spec = StrategySpec(
        mode=str(best_row["mode"]),
        model=str(best_row["model"]),
        threshold_quantile=float(best_row["threshold_quantile"]),
        threshold=float(best_row["threshold"]),
        top_k=int(best_row["top_k"]),
        max_hold=int(best_row["max_hold"]),
        filter_name=str(best_row["filter_name"]),
    )
    val_summary, val_trades = backtest_spec(
        validation_frames[best_spec.model],
        best_spec,
        events_by_ticker,
        price_dirs,
        cost_bps,
        price_cache=price_cache,
        exit_cache=exit_cache,
    )
    test_summary, test_trades = backtest_spec(
        test_frames[best_spec.model],
        best_spec,
        events_by_ticker,
        price_dirs,
        cost_bps,
        price_cache=price_cache,
        exit_cache=exit_cache,
    )
    val_summary["sample"] = "validation_selected"
    test_summary["sample"] = "test_selected"

    selected_summary = pd.DataFrame([val_summary, test_summary])
    selected_summary.to_csv(out_dir / "selected_strategy_summary.csv", index=False)
    val_trades.to_csv(out_dir / "selected_validation_trades.csv", index=False)
    test_trades.to_csv(out_dir / "selected_test_trades.csv", index=False)

    top_validation = validation_grid[validation_grid["trades"].ge(min_validation_trades)].sort_values(
        "total_return", ascending=False
    ).head(20)
    top_validation.to_csv(out_dir / "top_validation_strategies.csv", index=False)

    report = [
        "# Early Signal Strategy Search",
        "",
        f"Source run: `{run_dir}`",
        "",
        "The selected strategy is chosen on validation only, requiring a minimum trade count. Test results are then produced with the exact same threshold and rule.",
        "",
        "Exit rule: enter next open after signal; if an audited announcement occurs before the time stop, exit at the highest daily high in the 24-hour announcement window; otherwise exit at the fixed time-stop close.",
        "",
        "Daily OHLC makes announcement-peak exits optimistic for intraday announcements.",
        "",
        "## Selected Strategy Summary",
        selected_summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Top Validation Strategies",
        top_validation.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Selected Test Trades",
        test_trades[
            [
                "ticker",
                "date",
                "score",
                "event_20d",
                "next_open",
                "exit_date",
                "exit_kind",
                "net_return",
                "return_60d",
                "distance_from_52w_high",
                "sec_any_event_language_hit_30d",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion_full/early_signal_strategy")
    parser.add_argument("--price-dir", action="append", default=["data/raw/stockanalysis_prices", "data/raw/yfinance"])
    parser.add_argument("--cost-bps", type=float, default=250.0)
    parser.add_argument("--min-validation-trades", type=int, default=12)
    args = parser.parse_args()
    run_grid(
        run_dir=Path(args.run_dir),
        events_path=Path(args.events),
        price_dirs=[Path(p) for p in args.price_dir],
        out_dir=Path(args.out_dir),
        cost_bps=args.cost_bps,
        min_validation_trades=args.min_validation_trades,
    )


if __name__ == "__main__":
    main()
