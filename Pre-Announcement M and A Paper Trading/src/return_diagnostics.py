from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PANEL_COLUMNS = [
    "ticker",
    "date",
    "return_20d",
    "return_60d",
    "volume_to_20d_avg",
    "gap",
    "intraday_range",
    "distance_from_52w_high",
    "distance_from_52w_low",
    "price",
    "adv_20d_dollars",
    "days_to_event_calendar",
    "event_20d",
    "fwd_return_5d",
    "fwd_return_10d",
    "fwd_return_20d",
    "sec_days_since_last_filing",
    "sec_days_since_event_language",
    "sec_any_event_language_hit_30d",
    "sec_any_event_language_hit_90d",
    "sec_transaction_hit_30d",
    "sec_transaction_hit_90d",
    "sec_strategic_alternatives_hit_30d",
    "sec_strategic_alternatives_hit_90d",
    "sec_change_of_control_hit_30d",
    "sec_change_of_control_hit_90d",
]


def _win_rate(series: pd.Series, hurdle: float = 0.0) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.gt(hurdle).mean())


def _mean(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def _median(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.median())


def _pct(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return np.nan
    return float(values.mean())


def load_joined(run_dir: Path, model: str, sample: str) -> pd.DataFrame:
    predictions = pd.read_csv(run_dir / f"{model}_{sample}_predictions.csv", parse_dates=["date"])
    panel_cols = [c for c in PANEL_COLUMNS if c not in {"event_20d", "fwd_return_20d", "adv_20d_dollars"}]
    panel = pd.read_csv(
        run_dir / "fused_market_sec_panel.csv",
        usecols=lambda col: col in panel_cols,
        parse_dates=["date"],
    )
    df = predictions.merge(panel, on=["ticker", "date"], how="left")
    df["net_return_20d_250bps"] = df["fwd_return_20d"] - 0.025
    df["near_52w_high"] = df["distance_from_52w_high"].ge(-0.05)
    df["big_60d_runup"] = df["return_60d"].ge(0.25)
    df["recent_sec_event_language"] = df["sec_any_event_language_hit_30d"].fillna(0).ge(1)
    df["public_priced_in_proxy"] = (
        df["near_52w_high"].fillna(False)
        | df["big_60d_runup"].fillna(False)
        | df["recent_sec_event_language"].fillna(False)
    )
    return df


def summarize_frame(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    grouped = df.groupby(group_col, dropna=False, observed=False)
    rows = []
    for key, g in grouped:
        event_rows = g[g["event_20d"].eq(1)]
        rows.append(
            {
                group_col: key,
                "rows": int(len(g)),
                "return_obs": int(g["fwd_return_20d"].notna().sum()),
                "tickers": int(g["ticker"].nunique()),
                "event_rate": _mean(g["event_20d"]),
                "mean_raw_fwd_20d": _mean(g["fwd_return_20d"]),
                "median_raw_fwd_20d": _median(g["fwd_return_20d"]),
                "raw_win_rate": _win_rate(g["fwd_return_20d"]),
                "mean_net_fwd_20d_250bps": _mean(g["net_return_20d_250bps"]),
                "median_net_fwd_20d_250bps": _median(g["net_return_20d_250bps"]),
                "net_win_rate_250bps": _win_rate(g["net_return_20d_250bps"]),
                "event_mean_raw_fwd_20d": _mean(event_rows["fwd_return_20d"]),
                "event_median_raw_fwd_20d": _median(event_rows["fwd_return_20d"]),
                "event_net_win_rate_250bps": _win_rate(event_rows["net_return_20d_250bps"]),
                "mean_return_20d": _mean(g["return_20d"]),
                "median_return_20d": _median(g["return_20d"]),
                "mean_return_60d": _mean(g["return_60d"]),
                "median_return_60d": _median(g["return_60d"]),
                "mean_distance_from_52w_high": _mean(g["distance_from_52w_high"]),
                "near_52w_high_rate": _pct(g["near_52w_high"]),
                "big_60d_runup_rate": _pct(g["big_60d_runup"]),
                "recent_sec_event_language_rate": _pct(g["recent_sec_event_language"]),
                "priced_in_proxy_rate": _pct(g["public_priced_in_proxy"]),
                "median_days_to_event_calendar": _median(event_rows["days_to_event_calendar"]),
            }
        )
    return pd.DataFrame(rows)


def score_deciles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["score_decile"] = pd.qcut(out["score"], 10, labels=False, duplicates="drop") + 1
    return summarize_frame(out, "score_decile").sort_values("score_decile")


def topk_summary(df: pd.DataFrame, ks: tuple[int, ...] = (10, 25, 50, 100, 250, 500, 1000)) -> pd.DataFrame:
    ranked = df.sort_values("score", ascending=False).copy()
    rows = []
    for k in ks:
        rows.append(summarize_frame(ranked.head(k).assign(bucket=f"top_{k}"), "bucket").iloc[0].to_dict())
    return pd.DataFrame(rows)


def event_timing_summary(df: pd.DataFrame) -> pd.DataFrame:
    events = df[df["event_20d"].eq(1)].copy()
    bins = [-np.inf, 3, 7, 14, 32, np.inf]
    labels = ["1-3d", "4-7d", "8-14d", "15-32d", ">32d_or_bad_label"]
    events["days_to_event_bucket"] = pd.cut(events["days_to_event_calendar"], bins=bins, labels=labels)
    return summarize_frame(events, "days_to_event_bucket")


def priced_in_proxy_summary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["priced_in_proxy"] = np.where(out["public_priced_in_proxy"], "proxy_true", "proxy_false")
    return summarize_frame(out, "priced_in_proxy").sort_values("priced_in_proxy")


def selected_nonoverlap_trades(
    df: pd.DataFrame,
    top_k: int,
    threshold: float,
    min_adv: float = 100_000,
    holding_days: int = 20,
) -> pd.DataFrame:
    eligible = df[df["adv_20d_dollars"].fillna(0).ge(min_adv)]
    eligible = eligible[eligible["score"].ge(threshold)].copy()
    eligible = eligible.dropna(subset=["fwd_return_20d"])
    if eligible.empty:
        return eligible
    all_dates = pd.Index(sorted(pd.to_datetime(df["date"]).dropna().unique()))
    rebalance_dates = set(all_dates[::holding_days])
    eligible = eligible[eligible["date"].isin(rebalance_dates)].copy()
    if eligible.empty:
        return eligible
    eligible["rank"] = eligible.groupby("date")["score"].rank(ascending=False, method="first")
    trades = eligible[eligible["rank"].le(top_k)].copy()
    trades["net_return_20d_250bps"] = trades["fwd_return_20d"] - 0.025
    keep = [
        "date",
        "ticker",
        "rank",
        "score",
        "event_20d",
        "days_to_event_calendar",
        "fwd_return_20d",
        "net_return_20d_250bps",
        "return_20d",
        "return_60d",
        "distance_from_52w_high",
        "near_52w_high",
        "big_60d_runup",
        "recent_sec_event_language",
        "public_priced_in_proxy",
        "sec_days_since_event_language",
        "sec_any_event_language_hit_30d",
        "sec_transaction_hit_30d",
        "sec_strategic_alternatives_hit_30d",
        "adv_20d_dollars",
        "price",
    ]
    return trades[[c for c in keep if c in trades.columns]].sort_values(["date", "rank"])


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No rows._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".4f")


def build_report(run_dir: Path, out_dir: Path, samples: tuple[str, ...], models: tuple[str, ...]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    report_sections = [
        "# Return Diagnostics: Is The Merger Signal Already Priced In?",
        "",
        f"Source run: `{run_dir}`",
        "",
        "A `public_priced_in_proxy` is true when a row is within 5% of its 52-week high, has gained at least 25% over 60 trading days, or has SEC event language in the prior 30 days.",
        "",
    ]
    overview_rows = []
    for sample in samples:
        for model in models:
            pred_path = run_dir / f"{model}_{sample}_predictions.csv"
            if not pred_path.exists():
                continue
            df = load_joined(run_dir, model, sample)
            prefix = f"{model}_{sample}"

            deciles = score_deciles(df)
            topk = topk_summary(df)
            timing = event_timing_summary(df)
            proxy = priced_in_proxy_summary(df)

            deciles.to_csv(out_dir / f"{prefix}_score_deciles.csv", index=False)
            topk.to_csv(out_dir / f"{prefix}_topk_summary.csv", index=False)
            timing.to_csv(out_dir / f"{prefix}_event_timing_summary.csv", index=False)
            proxy.to_csv(out_dir / f"{prefix}_priced_in_proxy_summary.csv", index=False)

            selected = None
            if model == "hgb" and sample == "test":
                selected = selected_nonoverlap_trades(df, top_k=10, threshold=0.24510393487101204)
                selected.to_csv(out_dir / "hgb_test_selected_nonoverlap_trades.csv", index=False)
                selected_summary = summarize_frame(selected.assign(bucket="selected_nonoverlap"), "bucket")
                selected_summary.to_csv(out_dir / "hgb_test_selected_nonoverlap_summary.csv", index=False)

            top100 = topk.loc[topk["bucket"].eq("top_100")].iloc[0]
            top500 = topk.loc[topk["bucket"].eq("top_500")].iloc[0]
            overview_rows.append(
                {
                    "sample": sample,
                    "model": model,
                    "rows": int(len(df)),
                    "base_event_rate": float(df["event_20d"].mean()),
                    "base_mean_raw_fwd_20d": float(df["fwd_return_20d"].mean()),
                    "top100_event_rate": top100["event_rate"],
                    "top100_mean_net_250bps": top100["mean_net_fwd_20d_250bps"],
                    "top100_median_net_250bps": top100["median_net_fwd_20d_250bps"],
                    "top100_priced_in_proxy_rate": top100["priced_in_proxy_rate"],
                    "top100_median_days_to_event": top100["median_days_to_event_calendar"],
                    "top500_event_rate": top500["event_rate"],
                    "top500_mean_net_250bps": top500["mean_net_fwd_20d_250bps"],
                    "top500_priced_in_proxy_rate": top500["priced_in_proxy_rate"],
                }
            )

            if sample == "test" and model in {"rf", "hgb"}:
                report_sections.extend(
                    [
                        f"## {model.upper()} Test",
                        "",
                        "### Top-K Summary",
                        markdown_table(topk),
                        "",
                        "### Score Deciles",
                        markdown_table(deciles),
                        "",
                        "### Event Timing Summary",
                        markdown_table(timing),
                        "",
                        "### Priced-In Proxy Summary",
                        markdown_table(proxy),
                        "",
                    ]
                )
                if selected is not None:
                    report_sections.extend(
                        [
                            "### Validation-Selected HGB Non-Overlapping Test Trades",
                            markdown_table(selected.head(40)),
                            "",
                        ]
                    )

    overview = pd.DataFrame(overview_rows)
    overview.to_csv(out_dir / "overview.csv", index=False)
    report_sections.insert(6, markdown_table(overview))
    report_sections.insert(6, "## Overview")
    report_sections.insert(6, "")
    (out_dir / "REPORT.md").write_text("\n".join(report_sections))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full")
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion_full/return_diagnostics")
    parser.add_argument("--samples", nargs="+", default=["validation", "test"])
    parser.add_argument("--models", nargs="+", default=["logistic", "rf", "hgb"])
    args = parser.parse_args()
    build_report(Path(args.run_dir), Path(args.out_dir), tuple(args.samples), tuple(args.models))


if __name__ == "__main__":
    main()
