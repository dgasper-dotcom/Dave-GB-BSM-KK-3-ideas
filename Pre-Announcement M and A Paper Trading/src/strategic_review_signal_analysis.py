from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SIGNAL_COLUMNS = [
    "ticker",
    "date",
    "event_20d",
    "days_to_event_calendar",
    "fwd_return_20d",
    "return_60d",
    "distance_from_52w_high",
    "volume_to_20d_avg",
    "sec_any_event_language_hit_30d",
    "sec_transaction_hit_30d",
    "sec_transaction_hit_90d",
    "sec_strategic_alternatives_hit_90d",
    "sec_investment_bank_hit_90d",
    "sec_committee_hit_90d",
    "sec_change_of_control_hit_90d",
    "sec_confidentiality_hit_90d",
    "sec_unsolicited_interest_hit_90d",
    "sec_multiple_party_interest_hit_90d",
]


def _mean(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.mean()) if not values.empty else np.nan


def _median(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.median()) if not values.empty else np.nan


def _win_rate(series: pd.Series) -> float:
    values = series.dropna()
    return float(values.gt(0).mean()) if not values.empty else np.nan


def add_signal_masks(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    out["sample"] = np.select(
        [
            out["date"].lt(pd.Timestamp("2026-01-01")),
            out["date"].lt(pd.Timestamp("2026-06-01")),
        ],
        ["train", "validation"],
        default="test",
    )
    for col in SIGNAL_COLUMNS:
        if col not in out.columns:
            out[col] = np.nan

    review_90 = out["sec_strategic_alternatives_hit_90d"].fillna(0).gt(0)
    adviser_90 = out["sec_investment_bank_hit_90d"].fillna(0).gt(0)
    committee_90 = out["sec_committee_hit_90d"].fillna(0).gt(0)
    control_90 = out["sec_change_of_control_hit_90d"].fillna(0).gt(0)
    confidentiality_90 = out["sec_confidentiality_hit_90d"].fillna(0).gt(0)
    unsolicited_90 = out["sec_unsolicited_interest_hit_90d"].fillna(0).gt(0)
    multi_party_90 = out["sec_multiple_party_interest_hit_90d"].fillna(0).gt(0)
    transaction_30 = out["sec_transaction_hit_30d"].fillna(0).gt(0)
    transaction_90 = out["sec_transaction_hit_90d"].fillna(0).gt(0)

    no_big_runup = out["return_60d"].fillna(0).lt(0.25)
    away_high = out["distance_from_52w_high"].fillna(-1).lt(-0.05)
    no_volume_spike = out["volume_to_20d_avg"].fillna(1).lt(3.0)
    near_high = out["distance_from_52w_high"].ge(-0.05)
    big_runup = out["return_60d"].ge(0.25)
    recent_event_language = out["sec_any_event_language_hit_30d"].fillna(0).gt(0)

    cue_count = (
        adviser_90.astype(int)
        + committee_90.astype(int)
        + control_90.astype(int)
        + confidentiality_90.astype(int)
        + unsolicited_90.astype(int)
        + multi_party_90.astype(int)
    )
    out["public_priced_in_proxy"] = near_high | big_runup | recent_event_language
    out["strategic_review_90"] = review_90
    out["strategic_review_complex"] = review_90 & (adviser_90 | committee_90 | control_90 | confidentiality_90)
    out["strategic_review_interest"] = review_90 & (unsolicited_90 | multi_party_90)
    out["strategic_review_strong_combo"] = review_90 & cue_count.ge(2)
    out["strategic_review_no_recent_tx"] = review_90 & ~transaction_30
    out["strategic_review_not_priced"] = review_90 & ~transaction_30 & no_big_runup & away_high & no_volume_spike
    out["strategic_review_interest_not_priced"] = (
        review_90 & (unsolicited_90 | multi_party_90) & ~transaction_90 & no_big_runup & away_high
    )
    return out


def summarize(rows: pd.DataFrame, label: str, sample: str) -> dict[str, object]:
    event_rows = rows[rows["event_20d"].eq(1)]
    net_return = rows["fwd_return_20d"] - 0.025
    event_net_return = event_rows["fwd_return_20d"] - 0.025
    return {
        "sample": sample,
        "signal": label,
        "rows": int(len(rows)),
        "tickers": int(rows["ticker"].nunique()) if "ticker" in rows else 0,
        "event_rate_20d": _mean(rows["event_20d"]),
        "mean_net_return_20d_250bps": _mean(net_return),
        "median_net_return_20d_250bps": _median(net_return),
        "net_win_rate_20d_250bps": _win_rate(net_return),
        "event_rows": int(len(event_rows)),
        "event_mean_net_return_20d_250bps": _mean(event_net_return),
        "event_median_net_return_20d_250bps": _median(event_net_return),
        "median_days_to_event_calendar": _median(event_rows["days_to_event_calendar"]),
        "priced_in_proxy_rate": _mean(rows["public_priced_in_proxy"].astype(float)),
        "recent_sec_event_language_rate": _mean(rows["sec_any_event_language_hit_30d"].fillna(0).astype(float)),
        "mean_return_60d": _mean(rows["return_60d"]),
        "median_distance_from_52w_high": _median(rows["distance_from_52w_high"]),
    }


def build_report(run_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(
        run_dir / "fused_market_sec_panel.csv",
        usecols=lambda col: col in SIGNAL_COLUMNS,
        parse_dates=["date"],
    )
    panel = add_signal_masks(panel)
    signal_names = [
        "strategic_review_90",
        "strategic_review_complex",
        "strategic_review_interest",
        "strategic_review_strong_combo",
        "strategic_review_no_recent_tx",
        "strategic_review_not_priced",
        "strategic_review_interest_not_priced",
    ]
    rows = []
    for sample, sample_rows in panel[panel["sample"].isin(["validation", "test"])].groupby("sample"):
        rows.append(summarize(sample_rows, "baseline_all_rows", sample))
        for signal in signal_names:
            rows.append(summarize(sample_rows[sample_rows[signal]], signal, sample))
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "strategic_review_signal_summary.csv", index=False)

    test = summary[summary["sample"].eq("test")].copy()
    validation = summary[summary["sample"].eq("validation")].copy()
    report = [
        "# Strategic Review Signal Analysis",
        "",
        f"Source run: `{run_dir}`",
        "",
        "This report evaluates point-in-time SEC phrase signals directly, before model ranking. Returns are 20 trading-day forward returns net of 250 bps.",
        "",
        "A `public_priced_in_proxy` is true when the stock is within 5% of its 52-week high, has a 60-day runup of at least 25%, or had event-language SEC text in the prior 30 days.",
        "",
        "## Test Period",
        test.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## Validation Period",
        validation.to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_sec_market_fusion_full_strategy_review")
    parser.add_argument(
        "--out-dir",
        default="reports/profitability_sec_market_fusion_full_strategy_review/strategic_review_signal_analysis",
    )
    args = parser.parse_args()
    build_report(Path(args.run_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
