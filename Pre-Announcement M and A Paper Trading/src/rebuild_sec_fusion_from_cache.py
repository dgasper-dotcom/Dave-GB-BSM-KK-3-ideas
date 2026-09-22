from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .nlp_features import PHRASE_GROUPS, phrase_hits
from .sec_ingestion import parse_filing_text
from .sec_market_fusion import load_market_panel, rolling_sec_features, sec_feature_columns, train_and_backtest_fusion


def filing_feature_row(row: dict[str, object], max_document_chars: int) -> dict[str, object]:
    raw_path = Path(str(row["raw_path"]))
    raw = raw_path.read_text(errors="replace")[:max_document_chars]
    hits = phrase_hits(parse_filing_text(raw))
    out = {
        "ticker": str(row["ticker"]).upper(),
        "cik": str(row["cik"]).zfill(10),
        "form_type": row.get("form_type"),
        "accession_number": row.get("accessionNumber") or row.get("accession_number"),
        "information_available_timestamp": row.get("information_available_timestamp"),
    }
    for group in PHRASE_GROUPS:
        group_hits = [hit for hit in hits if hit.group == group]
        out[f"{group}_hit"] = int(bool(group_hits))
        out[f"{group}_count"] = int(sum(hit.count for hit in group_hits))
    out["any_event_language_hit"] = int(any(hit.group != "financing" for hit in hits))
    return out


def rebuild_from_cache(
    market_panel: Path,
    filing_index_path: Path,
    out_dir: Path,
    max_document_chars: int,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_market_panel(market_panel)
    filing_index = pd.read_csv(filing_index_path, dtype={"cik": str})
    feature_rows: list[dict[str, object]] = []
    errors: list[dict[str, object]] = []

    for i, row in enumerate(filing_index.to_dict("records"), start=1):
        raw_path = Path(str(row.get("raw_path", "")))
        if not raw_path.exists():
            errors.append(
                {
                    "ticker": row.get("ticker"),
                    "cik": row.get("cik"),
                    "accession_number": row.get("accessionNumber") or row.get("accession_number"),
                    "error": "missing_raw_path",
                    "raw_path": str(raw_path),
                }
            )
            continue
        try:
            feature_rows.append(filing_feature_row(row, max_document_chars=max_document_chars))
        except Exception as exc:
            errors.append(
                {
                    "ticker": row.get("ticker"),
                    "cik": row.get("cik"),
                    "accession_number": row.get("accessionNumber") or row.get("accession_number"),
                    "error": f"{type(exc).__name__}:{exc}",
                    "raw_path": str(raw_path),
                }
            )
        if i % 500 == 0:
            print(f"reparsed_filings={i} features={len(feature_rows)} errors={len(errors)}", flush=True)

    filing_features = pd.DataFrame(feature_rows)
    if filing_features.empty:
        raise ValueError("No cached SEC filing features were rebuilt.")
    filing_features["information_available_timestamp"] = pd.to_datetime(
        filing_features["information_available_timestamp"], utc=True, errors="coerce"
    )

    filing_index.to_csv(out_dir / "sec_filings_index.csv", index=False)
    filing_features.to_csv(out_dir / "sec_filing_phrase_features.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "sec_reparse_errors.csv", index=False)

    sec_features = rolling_sec_features(panel, filing_features)
    sec_features.to_csv(out_dir / "sec_rolling_features.csv", index=False)
    fused = panel.merge(sec_features, on=["ticker", "date"], how="left")
    sec_cols = sec_feature_columns(fused)
    fused[sec_cols] = fused[sec_cols].apply(pd.to_numeric, errors="coerce")
    for col in sec_cols:
        if not col.startswith("sec_days_since"):
            fused[col] = fused[col].fillna(0)
    fused.to_csv(out_dir / "fused_market_sec_panel.csv", index=False)

    result = train_and_backtest_fusion(fused, out_dir)
    result["sec_reparse"] = {
        "source_filing_index": str(filing_index_path),
        "parsed_filings": int(len(filing_features)),
        "errors": int(len(errors)),
        "phrase_groups": sorted(PHRASE_GROUPS),
        "max_document_chars": int(max_document_chars),
    }
    (out_dir / "fusion_summary.json").write_text(json.dumps(result, indent=2, default=str))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--market-panel", default="reports/profitability_full_stockanalysis/market_feature_panel.csv")
    parser.add_argument("--filing-index", default="reports/profitability_sec_market_fusion_full/sec_filings_index.csv")
    parser.add_argument("--out-dir", default="reports/profitability_sec_market_fusion_full_strategy_review")
    parser.add_argument("--max-document-chars", type=int, default=1_000_000)
    args = parser.parse_args()
    result = rebuild_from_cache(
        market_panel=Path(args.market_panel),
        filing_index_path=Path(args.filing_index),
        out_dir=Path(args.out_dir),
        max_document_chars=args.max_document_chars,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
