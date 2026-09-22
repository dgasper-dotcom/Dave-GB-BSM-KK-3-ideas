from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from .nlp_features import filing_phrase_features, rolling_nlp_features
from .point_in_time import filing_available_timestamp, prediction_timestamp
from .scanner import explain_row
from .sec_ingestion import SECClient, parse_filing_text


DEFAULT_FORMS = ["8-K", "10-Q", "10-K", "DEF 14A", "PRE 14A", "S-1", "S-3", "S-4", "424B"]


def previous_business_day(value: object) -> pd.Timestamp:
    date = pd.Timestamp(value).normalize()
    return pd.bdate_range(end=date - pd.Timedelta(days=1), periods=1)[0]


def load_feature_names(path: Path) -> list[str]:
    return json.loads(path.read_text())


def cohort_observations(cohort: pd.DataFrame, control_date: str) -> pd.DataFrame:
    rows = []
    for row in cohort.to_dict("records"):
        event_date = pd.to_datetime(row.get("event_date"), errors="coerce")
        if pd.notna(event_date):
            obs_date = previous_business_day(event_date)
            obs_kind = "pre_event_minus_1_bday"
        else:
            obs_date = pd.Timestamp(control_date).normalize()
            obs_kind = "control_asof"
        rows.append(
            {
                "cik": str(row["cik"]).zfill(10),
                "ticker": row["symbol"],
                "company": row.get("company_name"),
                "exchange": row.get("exchange"),
                "date": obs_date,
                "prediction_ts": prediction_timestamp(obs_date),
                "observation_kind": obs_kind,
                "event_type": row.get("event_type"),
                "event_date": row.get("event_date"),
                "label_quality": row.get("label_quality"),
            }
        )
    return pd.DataFrame(rows)


def download_company_filings(
    client: SECClient,
    obs_row: dict,
    forms: list[str],
    calendar: pd.Series,
    max_filings: int,
    lookback_days: int,
) -> list[dict]:
    try:
        index = client.filing_index(obs_row["cik"], forms=forms)
    except Exception as exc:
        return [
            {
                "cik": obs_row["cik"],
                "ticker": obs_row["ticker"],
                "error": f"filing_index:{type(exc).__name__}:{exc}",
            }
        ]

    if index.empty:
        return []
    index["filing_timestamp"] = pd.to_datetime(index["filing_timestamp"], utc=True, errors="coerce")
    index["information_available_timestamp"] = index["filing_timestamp"].apply(
        lambda ts: filing_available_timestamp(ts, calendar) if pd.notna(ts) else pd.NaT
    )
    pred_ts = pd.Timestamp(obs_row["prediction_ts"])
    min_ts = pred_ts - pd.Timedelta(days=lookback_days)
    selected = index[
        index["information_available_timestamp"].notna()
        & (index["information_available_timestamp"] <= pred_ts)
        & (index["information_available_timestamp"] >= min_ts)
    ].sort_values("information_available_timestamp", ascending=False)
    selected = selected.head(max_filings)

    rows = []
    for filing in selected.to_dict("records"):
        try:
            path = client.download_filing(filing["cik"], filing["accessionNumber"], filing["document_url"])
            raw = path.read_text(errors="replace")
            filing["raw_path"] = str(path)
            filing["parsed_text"] = parse_filing_text(raw)
            rows.append(filing)
        except Exception as exc:
            rows.append(
                {
                    "cik": obs_row["cik"],
                    "ticker": obs_row["ticker"],
                    "accessionNumber": filing.get("accessionNumber"),
                    "information_available_timestamp": filing.get("information_available_timestamp"),
                    "error": f"download_or_parse:{type(exc).__name__}:{exc}",
                }
            )
    return rows


def score_cohort(
    cohort_path: Path,
    model_path: Path,
    feature_list_path: Path,
    out_dir: Path,
    control_date: str,
    max_filings_per_company: int,
    lookback_days: int,
    forms: list[str],
    user_agent: str,
) -> dict[str, object]:
    cohort = pd.read_csv(cohort_path, dtype={"cik": str})
    observations = cohort_observations(cohort, control_date)
    calendar = pd.bdate_range("1998-01-01", pd.Timestamp(control_date) + pd.Timedelta(days=10))

    client = SECClient(user_agent=user_agent, cache_dir=Path("data/raw/sec"))
    filing_rows: list[dict] = []
    error_rows: list[dict] = []
    for i, obs in enumerate(observations.to_dict("records"), start=1):
        rows = download_company_filings(
            client,
            obs,
            forms=forms,
            calendar=calendar,
            max_filings=max_filings_per_company,
            lookback_days=lookback_days,
        )
        for row in rows:
            if "error" in row:
                error_rows.append(row)
            if "parsed_text" in row:
                filing_rows.append(row)
        if i % 50 == 0:
            print(f"processed_companies={i} parsed_filings={len(filing_rows)} errors={len(error_rows)}")

    filings = pd.DataFrame(filing_rows)
    if filings.empty:
        raise ValueError("No filings were parsed; cannot score cohort.")
    filing_features = filing_phrase_features(filings)
    features = rolling_nlp_features(filing_features, observations)
    for col in ("cash_runway_months", "market_cap", "volume_to_20d_avg"):
        if col not in features.columns:
            features[col] = pd.NA

    model_features = load_feature_names(feature_list_path)
    for col in model_features:
        if col not in features.columns:
            features[col] = 0
    features[model_features] = features[model_features].apply(pd.to_numeric, errors="coerce").fillna(0)

    model = joblib.load(model_path)
    scores = model.predict_proba(features[model_features])[:, 1]
    scored = features.merge(
        observations[
            [
                "cik",
                "ticker",
                "date",
                "observation_kind",
                "event_type",
                "event_date",
                "label_quality",
                "company",
                "exchange",
            ]
        ],
        on=["cik", "ticker", "date"],
        how="left",
        suffixes=("", "_obs"),
    )
    scored["predicted_probability_20d"] = scores
    scored["model_rank"] = scored["predicted_probability_20d"].rank(ascending=False, method="first")
    scored["explanation"] = scored.apply(explain_row, axis=1)
    scored = scored.sort_values("model_rank")

    out_dir.mkdir(parents=True, exist_ok=True)
    scored_path = out_dir / "cohort_scores_logistic_event_20d.csv"
    features_path = out_dir / "cohort_sec_text_features.csv"
    filings_path = out_dir / "cohort_parsed_filings_index.csv"
    errors_path = out_dir / "cohort_sec_errors.csv"
    summary_path = out_dir / "cohort_scoring_summary.json"
    scored.to_csv(scored_path, index=False)
    features.to_csv(features_path, index=False)
    filings.drop(columns=["parsed_text"], errors="ignore").to_csv(filings_path, index=False)
    pd.DataFrame(error_rows).to_csv(errors_path, index=False)

    summary = {
        "companies_requested": int(len(cohort)),
        "companies_scored": int(len(scored)),
        "parsed_filings": int(len(filings)),
        "errors": int(len(error_rows)),
        "model_path": str(model_path),
        "feature_list_path": str(feature_list_path),
        "scored_path": str(scored_path),
        "features_path": str(features_path),
        "filings_path": str(filings_path),
        "errors_path": str(errors_path),
        "control_date": control_date,
        "max_filings_per_company": max_filings_per_company,
        "lookback_days": lookback_days,
        "warning": "Scores use the AEMD-only in-sample smoke model and are not validated predictive probabilities.",
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="data/processed/cohort/training_company_cohort.csv")
    parser.add_argument("--model", default="models/aemd_sec_text_smoke/logistic_event_20d.joblib")
    parser.add_argument("--feature-list", default="models/aemd_sec_text_smoke/logistic_event_20d_features.json")
    parser.add_argument("--out-dir", default="reports/cohort_scoring")
    parser.add_argument("--control-date", default="2026-09-17")
    parser.add_argument("--max-filings-per-company", type=int, default=12)
    parser.add_argument("--lookback-days", type=int, default=730)
    parser.add_argument("--forms", nargs="*", default=DEFAULT_FORMS)
    parser.add_argument("--user-agent", default="transformative-tx-research/0.1 davidgasper@example.com")
    args = parser.parse_args()
    summary = score_cohort(
        cohort_path=Path(args.cohort),
        model_path=Path(args.model),
        feature_list_path=Path(args.feature_list),
        out_dir=Path(args.out_dir),
        control_date=args.control_date,
        max_filings_per_company=args.max_filings_per_company,
        lookback_days=args.lookback_days,
        forms=args.forms,
        user_agent=args.user_agent,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
