from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .nlp_features import PHRASE_GROUPS, phrase_hits
from .point_in_time import filing_available_timestamp, prediction_timestamp
from .profitability_diagnostics import nonoverlap_20d_backtest
from .profitability_research import MARKET_FEATURES, chronological_splits, make_model, rank_backtest, score_model
from .sec_ingestion import SECClient, parse_filing_text


DEFAULT_FORMS = ["8-K", "10-Q", "10-K", "DEF 14A", "PRE 14A", "DEFA14A", "DEFM14A", "S-4", "S-1", "S-3", "424B"]


def select_company_filings(
    filings: pd.DataFrame,
    max_filings_per_company: int,
    filing_selection: str = "recent",
) -> pd.DataFrame:
    filings = filings.sort_values("filing_timestamp", ascending=True).copy()
    if max_filings_per_company <= 0 or len(filings) <= max_filings_per_company:
        return filings
    if filing_selection == "recent":
        return filings.tail(max_filings_per_company)
    if filing_selection == "oldest":
        return filings.head(max_filings_per_company)
    if filing_selection == "even":
        positions = np.linspace(0, len(filings) - 1, max_filings_per_company)
        positions = np.rint(positions).astype(int)
        positions = np.unique(positions)
        if len(positions) < max_filings_per_company:
            missing = max_filings_per_company - len(positions)
            fillers = [pos for pos in range(len(filings)) if pos not in set(positions)]
            positions = np.sort(np.concatenate([positions, np.array(fillers[:missing], dtype=int)]))
        return filings.iloc[positions[:max_filings_per_company]].copy()
    raise ValueError(f"Unknown filing_selection={filing_selection!r}")


def load_market_panel(path: Path) -> pd.DataFrame:
    panel = pd.read_csv(path, dtype={"cik": str}, parse_dates=["date", "event_date"])
    panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
    return panel


def collect_filing_texts(
    panel: pd.DataFrame,
    cohort_path: Path,
    cache_dir: Path,
    out_dir: Path,
    user_agent: str,
    forms: list[str],
    max_filings_per_company: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    max_companies: int | None = None,
    max_document_chars: int = 1_000_000,
    filing_selection: str = "recent",
    checkpoint_every: int = 50,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cohort = pd.read_csv(cohort_path, dtype={"cik": str})
    needed_symbols = set(panel["ticker"].dropna().astype(str).str.upper().unique())
    companies = cohort[cohort["symbol"].astype(str).str.upper().isin(needed_symbols)].copy()
    companies["cik"] = companies["cik"].astype(str).str.zfill(10)
    companies = companies.drop_duplicates("cik")
    if max_companies:
        companies = companies.head(max_companies)

    client = SECClient(user_agent=user_agent, cache_dir=cache_dir)
    calendar = pd.bdate_range(start - pd.Timedelta(days=10), end + pd.Timedelta(days=10))
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []
    for i, company in enumerate(companies.to_dict("records"), start=1):
        cik = str(company["cik"]).zfill(10)
        symbol = str(company["symbol"]).upper()
        try:
            idx = client.filing_index(cik, forms=forms)
        except Exception as exc:
            errors.append({"ticker": symbol, "cik": cik, "error": f"filing_index:{type(exc).__name__}:{exc}"})
            continue
        if idx.empty:
            continue
        idx["filing_timestamp"] = pd.to_datetime(idx["filing_timestamp"], utc=True, errors="coerce")
        idx = idx[idx["filing_timestamp"].between(start.tz_localize("UTC"), end.tz_localize("UTC"), inclusive="both")]
        idx = select_company_filings(idx, max_filings_per_company, filing_selection=filing_selection)
        for filing in idx.to_dict("records"):
            try:
                path = client.download_filing(cik, filing["accessionNumber"], filing["document_url"])
                with path.open("r", errors="replace") as handle:
                    raw = handle.read(max_document_chars)
                available_ts = filing_available_timestamp(filing["filing_timestamp"], calendar)
                index_row = {
                    "ticker": symbol,
                    "cik": cik,
                    "form_type": filing.get("form_type"),
                    "accessionNumber": filing.get("accessionNumber"),
                    "filing_timestamp": filing.get("filing_timestamp"),
                    "information_available_timestamp": available_ts,
                    "document_url": filing.get("document_url"),
                    "raw_path": str(path),
                }
                text = parse_filing_text(raw)
                hits = phrase_hits(text)
                feature_row = {
                    "ticker": symbol,
                    "cik": cik,
                    "form_type": filing.get("form_type"),
                    "accession_number": filing.get("accessionNumber"),
                    "information_available_timestamp": available_ts,
                }
                for group in PHRASE_GROUPS:
                    group_hits = [hit for hit in hits if hit.group == group]
                    feature_row[f"{group}_hit"] = int(bool(group_hits))
                    feature_row[f"{group}_count"] = int(sum(hit.count for hit in group_hits))
                feature_row["any_event_language_hit"] = int(any(hit.group != "financing" for hit in hits))
                index_rows.append(index_row)
                feature_rows.append(feature_row)
            except Exception as exc:
                errors.append({"ticker": symbol, "cik": cik, "error": f"download_parse:{type(exc).__name__}:{exc}"})
        if i % max(1, checkpoint_every) == 0:
            if index_rows:
                pd.DataFrame(index_rows).to_csv(out_dir / "sec_filings_index.partial.csv", index=False)
            if feature_rows:
                pd.DataFrame(feature_rows).to_csv(out_dir / "sec_filing_phrase_features.partial.csv", index=False)
            pd.DataFrame(errors).to_csv(out_dir / "sec_collection_errors.partial.csv", index=False)
            print(f"sec_companies={i} parsed_filings={len(index_rows)} errors={len(errors)}", flush=True)

    filing_features = pd.DataFrame(feature_rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    if index_rows:
        pd.DataFrame(index_rows).to_csv(out_dir / "sec_filings_index.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "sec_collection_errors.csv", index=False)
    if not filing_features.empty:
        filing_features["information_available_timestamp"] = pd.to_datetime(
            filing_features["information_available_timestamp"], utc=True
        )
    summary = {
        "companies_requested": int(len(companies)),
        "parsed_filings": int(len(filing_features)),
        "errors": int(len(errors)),
        "forms": forms,
        "max_filings_per_company": max_filings_per_company,
        "max_document_chars": max_document_chars,
        "filing_selection": filing_selection,
        "checkpoint_every": checkpoint_every,
    }
    return filing_features, summary


def rolling_sec_features(panel: pd.DataFrame, filing_features: pd.DataFrame) -> pd.DataFrame:
    base = panel[["ticker", "date"]].copy()
    base["ticker"] = base["ticker"].astype(str).str.upper()
    base["prediction_ts"] = base["date"].apply(prediction_timestamp)
    feature_cols = [c for c in filing_features.columns if c.endswith("_hit") or c.endswith("_count")]
    output_parts = []
    ff = filing_features.copy()
    ff["ticker"] = ff["ticker"].astype(str).str.upper()
    ff["information_available_timestamp"] = pd.to_datetime(ff["information_available_timestamp"], utc=True, errors="coerce")

    for ticker, obs in base.groupby("ticker", sort=False):
        obs = obs.copy()
        obs_ts = pd.to_datetime(obs["prediction_ts"], utc=True).astype("int64").to_numpy()
        filings = ff[ff["ticker"].eq(ticker)].sort_values("information_available_timestamp")
        out = pd.DataFrame(index=obs.index)
        out["ticker"] = obs["ticker"].to_numpy()
        out["date"] = obs["date"].to_numpy()
        if filings.empty:
            out["sec_filings_180d"] = 0
            out["sec_days_since_event_language"] = np.nan
            for col in feature_cols:
                for suffix in ("30d", "90d", "180d", "ever"):
                    out[f"sec_{col}_{suffix}"] = 0
            output_parts.append(out)
            continue
        filing_ts = pd.to_datetime(filings["information_available_timestamp"], utc=True).astype("int64").to_numpy()
        right = np.searchsorted(filing_ts, obs_ts, side="right")
        last_idx = right - 1
        valid_last = last_idx >= 0
        day_ns = 24 * 60 * 60 * 1_000_000_000
        out["sec_days_since_last_filing"] = np.where(
            valid_last,
            (obs_ts - filing_ts[np.maximum(last_idx, 0)]) / day_ns,
            np.nan,
        )
        event_hit = filings["any_event_language_hit"].fillna(0).astype(int).to_numpy()
        event_positions = np.flatnonzero(event_hit > 0)
        if len(event_positions):
            last_event_pos = event_positions[np.searchsorted(event_positions, last_idx, side="right") - 1]
            has_event = np.searchsorted(event_positions, last_idx, side="right") > 0
            out["sec_days_since_event_language"] = np.where(
                has_event,
                (obs_ts - filing_ts[last_event_pos]) / day_ns,
                np.nan,
            )
        else:
            out["sec_days_since_event_language"] = np.nan

        for days in (30, 90, 180):
            left = np.searchsorted(filing_ts, obs_ts - days * day_ns, side="right")
            out[f"sec_filings_{days}d"] = right - left
        out["sec_filings_ever"] = right

        for col in feature_cols:
            values = filings[col].fillna(0).astype(float).to_numpy()
            csum = np.concatenate([[0.0], np.cumsum(values)])
            for days in (30, 90, 180):
                left = np.searchsorted(filing_ts, obs_ts - days * day_ns, side="right")
                vals = csum[right] - csum[left]
                if col.endswith("_hit"):
                    vals = (vals > 0).astype(int)
                out[f"sec_{col}_{days}d"] = vals
            vals = csum[right]
            if col.endswith("_hit"):
                vals = (vals > 0).astype(int)
            out[f"sec_{col}_ever"] = vals
        output_parts.append(out)
    sec = pd.concat(output_parts).sort_index()
    return sec


def sec_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c.startswith("sec_") and c not in {"sec_days_since_event_language", "sec_days_since_last_filing"}] + [
        c for c in ("sec_days_since_event_language", "sec_days_since_last_filing") if c in frame.columns
    ]


def train_and_backtest_fusion(panel: pd.DataFrame, out_dir: Path, target: str = "event_20d") -> dict[str, object]:
    train, val, test = chronological_splits(panel)
    features = MARKET_FEATURES + sec_feature_columns(panel)
    rows = []
    selected_models = {}
    for name in ("logistic", "rf", "hgb"):
        model = make_model(name)
        fit_train = train.dropna(subset=[target]).copy()
        if fit_train[target].nunique() < 2:
            continue
        model.fit(fit_train[features], fit_train[target].astype(int))
        selected_models[name] = model
        for sample_name, sample in (("validation", val), ("test", test)):
            if sample.empty or sample[target].nunique() < 2:
                continue
            scored = sample[["ticker", "date", "event_date", target, "next_open", "close", "adv_20d_dollars", "fwd_return_20d"]].copy()
            scored["score"] = model.predict_proba(sample[features])[:, 1]
            scored.to_csv(out_dir / f"{name}_{sample_name}_predictions.csv", index=False)
            from .evaluation import rare_event_metrics

            metrics = rare_event_metrics(scored[target].to_numpy(), scored["score"].to_numpy())
            metrics.update({"model": name, "sample": sample_name})
            rows.append(metrics)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)

    search_rows = []
    best = None
    best_model = None
    for name in selected_models:
        pred = pd.read_csv(out_dir / f"{name}_validation_predictions.csv", parse_dates=["date"])
        for top_k in (1, 3, 5, 10):
            for quantile in (0.90, 0.95, 0.98, 0.99):
                threshold = float(pred["score"].quantile(quantile))
                bt = rank_backtest(pred, top_k, threshold, max_pct_adv=0.03, min_adv=100_000, cost_bps=250)
                bt.update({"model": name, "sample": "validation", "quantile": quantile})
                search_rows.append(bt)
                if bt.get("trades", 0) >= 20 and (best is None or bt.get("total_return", -999) > best.get("total_return", -999)):
                    best = bt
                    best_model = name
    pd.DataFrame(search_rows).to_csv(out_dir / "validation_backtest_search.csv", index=False)

    nonoverlap_rows = []
    for sample_name in ("validation", "test"):
        for name in selected_models:
            pred = pd.read_csv(out_dir / f"{name}_{sample_name}_predictions.csv", parse_dates=["date"])
            for top_k in (1, 3, 5, 10):
                for quantile in (0.90, 0.95, 0.98, 0.99):
                    threshold = float(pred["score"].quantile(quantile))
                    bt = nonoverlap_20d_backtest(pred, top_k, threshold)
                    bt.update({"model": name, "sample": sample_name, "quantile": quantile})
                    nonoverlap_rows.append(bt)
    nonoverlap = pd.DataFrame(nonoverlap_rows)
    nonoverlap.to_csv(out_dir / "nonoverlap_20d_backtest_grid.csv", index=False)

    test_result = {}
    if best and best_model:
        test_pred = pd.read_csv(out_dir / f"{best_model}_test_predictions.csv", parse_dates=["date"])
        test_result = rank_backtest(
            test_pred,
            int(best["top_k"]),
            float(best["threshold"]),
            max_pct_adv=0.03,
            min_adv=float(best["min_adv"]),
            cost_bps=float(best["cost_bps"]),
        )
        test_result.update({"model": best_model, "sample": "test", "selected_from_validation": best})
        pd.DataFrame([test_result]).to_csv(out_dir / "selected_test_backtest.csv", index=False)
        joblib.dump(selected_models[best_model], out_dir / "selected_model.joblib")
    eligible = nonoverlap[(nonoverlap["sample"].eq("validation")) & (nonoverlap["trades"].ge(10))]
    best_nonoverlap = eligible.sort_values("total_return", ascending=False).head(1).to_dict("records")
    selected_nonoverlap_test = {}
    if best_nonoverlap:
        selected = best_nonoverlap[0]
        test_pred = pd.read_csv(out_dir / f"{selected['model']}_test_predictions.csv", parse_dates=["date"])
        selected_nonoverlap_test = nonoverlap_20d_backtest(
            test_pred,
            int(selected["top_k"]),
            float(selected["threshold"]),
            min_adv=float(selected["min_adv"]),
            cost_bps=float(selected["cost_bps"]),
        )
        selected_nonoverlap_test.update(
            {
                "model": selected["model"],
                "sample": "test",
                "selected_from_validation": selected,
            }
        )
        pd.DataFrame([selected_nonoverlap_test]).to_csv(out_dir / "selected_nonoverlap_test_backtest.csv", index=False)
    return {
        "classification_metrics": metrics.to_dict("records"),
        "best_validation": best,
        "selected_test": test_result,
        "best_nonoverlap_validation": best_nonoverlap[0] if best_nonoverlap else None,
        "selected_nonoverlap_test": selected_nonoverlap_test,
        "rows": int(len(panel)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_events": int(train[target].sum()),
        "validation_events": int(val[target].sum()),
        "test_events": int(test[target].sum()),
        "feature_count": len(features),
        "sec_feature_count": len(sec_feature_columns(panel)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-panel", default="reports/profitability_full_stockanalysis/market_feature_panel.csv")
    parser.add_argument("--cohort", default="data/processed/cohort/training_company_cohort.csv")
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion")
    parser.add_argument("--sec-cache", default="data/raw/sec")
    parser.add_argument("--user-agent", default="transformative-tx-research/0.1 davidgasper@example.com")
    parser.add_argument("--max-filings-per-company", type=int, default=12)
    parser.add_argument("--max-companies", type=int)
    parser.add_argument("--max-document-chars", type=int, default=1_000_000)
    parser.add_argument("--filing-selection", choices=["recent", "oldest", "even"], default="recent")
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--forms", nargs="*", default=DEFAULT_FORMS)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_market_panel(Path(args.market_panel))
    start = panel["date"].min() - pd.Timedelta(days=365)
    end = panel["date"].max() + pd.Timedelta(days=2)
    filing_features, sec_summary = collect_filing_texts(
        panel,
        Path(args.cohort),
        Path(args.sec_cache),
        out_dir,
        args.user_agent,
        args.forms,
        args.max_filings_per_company,
        start,
        end,
        args.max_companies,
        args.max_document_chars,
        args.filing_selection,
        args.checkpoint_every,
    )
    if filing_features.empty:
        raise ValueError("No SEC filings parsed.")
    filing_features.to_csv(out_dir / "sec_filing_phrase_features.csv", index=False)
    sec_features = rolling_sec_features(panel, filing_features)
    sec_features.to_csv(out_dir / "sec_rolling_features.csv", index=False)
    fused = panel.merge(sec_features, on=["ticker", "date"], how="left")
    sec_cols = sec_feature_columns(fused)
    fused[sec_cols] = fused[sec_cols].apply(pd.to_numeric, errors="coerce")
    for col in sec_cols:
        if col.startswith("sec_days_since"):
            continue
        fused[col] = fused[col].fillna(0)
    fused.to_csv(out_dir / "fused_market_sec_panel.csv", index=False)
    if args.skip_training:
        result = {
            "rows": int(len(fused)),
            "tickers": int(fused["ticker"].nunique()),
            "date_min": str(fused["date"].min().date()),
            "date_max": str(fused["date"].max().date()),
            "feature_count": int(len(MARKET_FEATURES) + len(sec_cols)),
            "sec_feature_count": int(len(sec_cols)),
            "sec_collection": sec_summary,
            "training_skipped": True,
        }
        (out_dir / "fusion_summary.json").write_text(json.dumps(result, indent=2, default=str))
        print(json.dumps(result, indent=2, default=str))
        return
    result = train_and_backtest_fusion(fused, out_dir)
    result["sec_collection"] = sec_summary
    (out_dir / "fusion_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
