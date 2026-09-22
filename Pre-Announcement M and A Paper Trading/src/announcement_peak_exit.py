from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0


def load_events(events_path: Path) -> pd.DataFrame:
    events = pd.read_csv(events_path)
    events = events[events["audited_announcement_ts"].notna()].copy()
    events["announcement_ts_utc"] = pd.to_datetime(events["audited_announcement_ts"], utc=True)
    events["announcement_ts_et"] = events["announcement_ts_utc"].dt.tz_convert("America/New_York")
    events["announcement_date_et"] = events["announcement_ts_et"].dt.tz_localize(None).dt.normalize()
    keep = [
        "ticker",
        "event_type",
        "company_name",
        "announcement_ts_utc",
        "announcement_ts_et",
        "announcement_date_et",
        "audit_status",
        "audit_score",
        "audit_form",
        "audit_document_url",
    ]
    return events[keep].drop_duplicates("ticker", keep="first")


def load_prices(ticker: str, price_dirs: list[Path]) -> pd.DataFrame:
    for price_dir in price_dirs:
        path = price_dir / f"{ticker}.csv"
        if not path.exists():
            continue
        prices = pd.read_csv(path, parse_dates=["date"])
        for col in ["open", "high", "low", "close"]:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")
        prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
        return prices.sort_values("date").drop_duplicates("date", keep="last")
    return pd.DataFrame()


def next_trading_day(prices: pd.DataFrame, signal_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    future = prices[prices["date"].gt(pd.Timestamp(signal_date).normalize())]
    if future.empty:
        return pd.NaT
    return future.iloc[0]["date"]


def daily_peak_window(announcement_ts_et: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    naive = announcement_ts_et.tz_localize(None)
    announce_date = naive.normalize()
    close_time = announce_date + pd.Timedelta(hours=MARKET_CLOSE_HOUR, minutes=MARKET_CLOSE_MINUTE)
    open_time = announce_date + pd.Timedelta(hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE)
    end_time = naive + pd.Timedelta(hours=24)

    if naive >= close_time:
        return naive, end_time, "after_close_daily_high"
    if naive < open_time:
        return naive, end_time, "premarket_daily_high"
    return naive, end_time, "intraday_daily_high_optimistic"


def daily_sessions_overlapping(prices: pd.DataFrame, start_time: pd.Timestamp, end_time: pd.Timestamp) -> pd.DataFrame:
    sessions = prices.copy()
    sessions["session_start"] = sessions["date"] + pd.Timedelta(hours=MARKET_OPEN_HOUR, minutes=MARKET_OPEN_MINUTE)
    sessions["session_end"] = sessions["date"] + pd.Timedelta(hours=MARKET_CLOSE_HOUR, minutes=MARKET_CLOSE_MINUTE)
    return sessions[(sessions["session_end"].ge(start_time)) & (sessions["session_start"].le(end_time))].copy()


def announcement_peak_for_row(row: pd.Series, prices: pd.DataFrame) -> dict[str, object]:
    if prices.empty or pd.isna(row.get("announcement_ts_et")):
        return {
            "entry_date": pd.NaT,
            "announcement_peak_date": pd.NaT,
            "announcement_peak_high": np.nan,
            "announcement_peak_return": np.nan,
            "peak_window_note": "missing_prices_or_announcement",
        }

    signal_date = pd.Timestamp(row["date"]).normalize()
    entry_date = next_trading_day(prices, signal_date)
    start_time, end_time, note = daily_peak_window(row["announcement_ts_et"])
    window = daily_sessions_overlapping(prices, start_time, end_time)
    if window.empty or pd.isna(entry_date):
        return {
            "entry_date": entry_date,
            "announcement_peak_date": pd.NaT,
            "announcement_peak_high": np.nan,
            "announcement_peak_return": np.nan,
            "peak_window_note": f"{note};no_daily_bar_in_window",
        }
    peak_idx = window["high"].idxmax()
    peak = window.loc[peak_idx]
    entry_price = row["next_open"]
    peak_return = peak["high"] / entry_price - 1 if entry_price and not pd.isna(entry_price) else np.nan
    if pd.notna(entry_date) and peak["date"] < entry_date:
        peak_return = np.nan
        note = f"{note};peak_before_entry"
    return {
        "entry_date": entry_date,
        "announcement_peak_date": peak["date"],
        "announcement_peak_high": float(peak["high"]),
        "announcement_peak_return": float(peak_return) if not pd.isna(peak_return) else np.nan,
        "peak_window_note": note,
    }


def select_nonoverlap_trades(
    predictions: pd.DataFrame,
    top_k: int,
    threshold: float,
    min_adv: float,
    holding_days: int = 20,
) -> pd.DataFrame:
    df = predictions.copy()
    df = df[df["adv_20d_dollars"].fillna(0).ge(min_adv)]
    df = df[df["score"].ge(threshold)]
    df = df.dropna(subset=["fwd_return_20d"])
    if df.empty:
        return df
    all_dates = pd.Index(sorted(pd.to_datetime(predictions["date"]).dropna().unique()))
    rebalance_dates = set(all_dates[::holding_days])
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["date"].isin(rebalance_dates)].copy()
    if df.empty:
        return df
    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="first")
    return df[df["rank"].le(top_k)].copy().sort_values(["date", "rank"])


def first_signal_entries(predictions: pd.DataFrame, threshold: float, min_adv: float) -> pd.DataFrame:
    df = predictions.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df[df["adv_20d_dollars"].fillna(0).ge(min_adv)]
    df = df[df["score"].ge(threshold)]
    df = df.sort_values(["ticker", "date", "score"], ascending=[True, True, False])
    return df.groupby("ticker", as_index=False).head(1).copy()


def add_peak_exit(
    trades: pd.DataFrame,
    events: pd.DataFrame,
    price_dirs: list[Path],
    only_announced: bool = True,
) -> pd.DataFrame:
    out = trades.merge(events, on="ticker", how="left")
    if only_announced:
        out = out[out["announcement_ts_et"].notna()].copy()
    if out.empty:
        return out

    cache: dict[str, pd.DataFrame] = {}
    annotations = []
    for _, row in out.iterrows():
        ticker = row["ticker"]
        if ticker not in cache:
            cache[ticker] = load_prices(ticker, price_dirs)
        annotations.append(announcement_peak_for_row(row, cache[ticker]))
    ann = pd.DataFrame(annotations, index=out.index)
    out = pd.concat([out, ann], axis=1)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["days_signal_to_announcement"] = (
        pd.to_datetime(out["announcement_date_et"]) - pd.to_datetime(out["date"])
    ).dt.days
    out["old_20d_return_raw"] = out["fwd_return_20d"]
    out["old_20d_return_net_250bps"] = out["old_20d_return_raw"] - 0.025
    out["announcement_peak_return_net_250bps"] = out["announcement_peak_return"] - 0.025
    out["peak_minus_old_20d_net"] = out["announcement_peak_return_net_250bps"] - out["old_20d_return_net_250bps"]
    return out


def summarize(label: str, rows: pd.DataFrame) -> dict[str, object]:
    valid = rows.dropna(subset=["announcement_peak_return"])
    if valid.empty:
        best_ticker = ""
        trimmed = valid
    else:
        best_idx = valid["announcement_peak_return_net_250bps"].idxmax()
        best_ticker = str(valid.loc[best_idx, "ticker"])
        trimmed = valid.drop(index=best_idx)
    return {
        "scenario": label,
        "rows": int(len(rows)),
        "rows_with_peak_exit": int(len(valid)),
        "tickers": int(rows["ticker"].nunique()) if "ticker" in rows else 0,
        "best_peak_ticker": best_ticker,
        "mean_old_20d_net": float(valid["old_20d_return_net_250bps"].mean()) if not valid.empty else np.nan,
        "median_old_20d_net": float(valid["old_20d_return_net_250bps"].median()) if not valid.empty else np.nan,
        "old_20d_net_win_rate": float(valid["old_20d_return_net_250bps"].gt(0).mean()) if not valid.empty else np.nan,
        "mean_peak_net": float(valid["announcement_peak_return_net_250bps"].mean()) if not valid.empty else np.nan,
        "median_peak_net": float(valid["announcement_peak_return_net_250bps"].median()) if not valid.empty else np.nan,
        "peak_net_win_rate": float(valid["announcement_peak_return_net_250bps"].gt(0).mean()) if not valid.empty else np.nan,
        "mean_peak_net_ex_best": float(trimmed["announcement_peak_return_net_250bps"].mean()) if not trimmed.empty else np.nan,
        "median_peak_net_ex_best": float(trimmed["announcement_peak_return_net_250bps"].median()) if not trimmed.empty else np.nan,
        "peak_net_win_rate_ex_best": float(trimmed["announcement_peak_return_net_250bps"].gt(0).mean()) if not trimmed.empty else np.nan,
        "mean_peak_minus_old_net": float(valid["peak_minus_old_20d_net"].mean()) if not valid.empty else np.nan,
        "median_peak_minus_old_net": float(valid["peak_minus_old_20d_net"].median()) if not valid.empty else np.nan,
        "median_days_signal_to_announcement": float(valid["days_signal_to_announcement"].median()) if not valid.empty else np.nan,
    }


def markdown_table(df: pd.DataFrame, max_rows: int = 30) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".4f")


def build_report(
    run_dir: Path,
    out_dir: Path,
    events_path: Path,
    price_dirs: list[Path],
    model: str,
    sample: str,
    threshold: float,
    top_k: int,
    min_adv: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(run_dir / f"{model}_{sample}_predictions.csv", parse_dates=["date"])
    events = load_events(events_path)

    selected = select_nonoverlap_trades(predictions, top_k=top_k, threshold=threshold, min_adv=min_adv)
    selected_peak = add_peak_exit(selected, events, price_dirs, only_announced=True)
    first_signal = first_signal_entries(predictions, threshold=threshold, min_adv=min_adv)
    first_signal_peak = add_peak_exit(first_signal, events, price_dirs, only_announced=True)

    selected_peak.to_csv(out_dir / f"{model}_{sample}_selected_nonoverlap_announcement_peak.csv", index=False)
    first_signal_peak.to_csv(out_dir / f"{model}_{sample}_first_signal_announcement_peak.csv", index=False)

    summary = pd.DataFrame(
        [
            summarize("selected_nonoverlap_announced_only", selected_peak),
            summarize("first_signal_announced_only", first_signal_peak),
        ]
    )
    summary.to_csv(out_dir / "summary.csv", index=False)

    selected_cols = [
        "ticker",
        "date",
        "entry_date",
        "score",
        "rank",
        "days_signal_to_announcement",
        "announcement_ts_et",
        "old_20d_return_net_250bps",
        "announcement_peak_return_net_250bps",
        "peak_minus_old_20d_net",
        "announcement_peak_date",
        "announcement_peak_high",
        "peak_window_note",
    ]
    first_cols = [c for c in selected_cols if c != "rank"]
    report = [
        "# Announcement Peak Exit Diagnostic",
        "",
        f"Source run: `{run_dir}`",
        f"Model/sample: `{model}` / `{sample}`",
        f"Signal threshold: `{threshold:.6f}`; top_k for selected non-overlap trades: `{top_k}`; min ADV: `${min_adv:,.0f}`",
        "",
        "Entry is the next available open after the signal date. The peak exit uses the highest daily high in the 24-hour window after the audited SEC announcement timestamp.",
        "",
        "Because the cache is daily OHLC, this is an optimistic proxy for a true intraday peak. For intraday announcements, the full announcement-day high may include prices before the filing time.",
        "",
        "## Summary",
        markdown_table(summary),
        "",
        "## Selected Non-Overlap Trades With Announcements",
        markdown_table(selected_peak[[c for c in selected_cols if c in selected_peak.columns]]),
        "",
        "## First Signal Per Ticker With Announcements",
        markdown_table(first_signal_peak[[c for c in first_cols if c in first_signal_peak.columns]]),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def selected_hgb_threshold(run_dir: Path) -> tuple[float, int, float]:
    selected_path = run_dir / "selected_nonoverlap_test_backtest.csv"
    if not selected_path.exists():
        return 0.24510393487101204, 10, 100_000.0
    row = pd.read_csv(selected_path).iloc[0]
    return float(row["threshold"]), int(row["top_k"]), float(row["min_adv"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full")
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion_full/announcement_peak_exit")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument("--price-dir", action="append", default=["data/raw/stockanalysis_prices", "data/raw/yfinance"])
    parser.add_argument("--model", default="hgb")
    parser.add_argument("--sample", default="test")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--min-adv", type=float)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    default_threshold, default_top_k, default_min_adv = selected_hgb_threshold(run_dir)
    threshold = args.threshold if args.threshold is not None else default_threshold
    top_k = args.top_k if args.top_k is not None else default_top_k
    min_adv = args.min_adv if args.min_adv is not None else default_min_adv

    build_report(
        run_dir=run_dir,
        out_dir=Path(args.out_dir),
        events_path=Path(args.events),
        price_dirs=[Path(p) for p in args.price_dir],
        model=args.model,
        sample=args.sample,
        threshold=threshold,
        top_k=top_k,
        min_adv=min_adv,
    )


if __name__ == "__main__":
    main()
