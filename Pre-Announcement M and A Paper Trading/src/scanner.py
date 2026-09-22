from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd


def explain_row(row: pd.Series, top_n: int = 5) -> str:
    reasons = []
    checks = [
        ("strategic alternatives language", row.get("strategic_alternatives_hit_90d", 0)),
        ("investment-bank/advisor language", row.get("investment_bank_hit_90d", 0)),
        ("change-of-control language", row.get("change_of_control_hit_90d", 0)),
        ("special committee language", row.get("committee_hit_90d", 0)),
        ("cash runway below 12 months", row.get("cash_runway_months", 999) < 12),
        ("market cap below $25M", row.get("market_cap", 1e12) < 25_000_000),
        ("unusual volume", row.get("volume_to_20d_avg", 0) > 3),
        ("recent financing language", row.get("financing_hit_90d", 0)),
    ]
    for label, flag in checks:
        if bool(flag):
            reasons.append(label)
    if not reasons:
        reasons.append("elevated model score from combined historical features")
    return "; ".join(reasons[:top_n])


def score_daily(feature_panel: pd.DataFrame, model_path: Path, feature_names: list[str], out_path: Path) -> pd.DataFrame:
    model = joblib.load(model_path)
    today = feature_panel.copy()
    scores = model.predict_proba(today[feature_names])[:, 1]
    today["predicted_probability_20d"] = scores
    today["model_rank"] = today["predicted_probability_20d"].rank(ascending=False, method="first")
    today["explanation"] = today.apply(explain_row, axis=1)
    today = today.sort_values("model_rank")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    today.to_csv(out_path, index=False)
    return today


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--feature-list", required=True)
    parser.add_argument("--out", default="reports/live_scanner.csv")
    args = parser.parse_args()
    panel = pd.read_parquet(args.features)
    feature_names = json.loads(Path(args.feature_list).read_text())
    score_daily(panel, Path(args.model), feature_names, Path(args.out))


if __name__ == "__main__":
    main()
