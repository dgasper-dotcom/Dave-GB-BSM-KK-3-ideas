from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from .sec_ingestion import SECClient, parse_filing_text


MNA_FORMS = {
    "8-K",
    "S-4",
    "S-4/A",
    "DEFM14A",
    "DEFA14A",
    "PREM14A",
    "PRE 14A",
    "DEF 14A",
    "SC TO-T",
    "SC TO-I",
    "SC 14D9",
    "425",
}

TRANSACTION_PATTERNS = [
    r"agreement\s+and\s+plan\s+of\s+merger",
    r"merger\s+agreement",
    r"definitive\s+agreement",
    r"business\s+combination\s+agreement",
    r"tender\s+offer",
    r"change\s+of\s+control",
    r"acquisition\s+agreement",
    r"will\s+be\s+acquired",
    r"entered\s+into\s+.*(?:merger|acquisition)",
    r"sale\s+of\s+substantially\s+all\s+assets",
]


def normalize_name(value: object) -> str:
    text = re.sub(r"[^a-z0-9 ]+", " ", str(value).lower())
    text = re.sub(r"\b(inc|corp|corporation|ltd|llc|plc|co|company|lp|limited|holdings?)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def transaction_score(text: str, acquirer_name: str | None = None, acquirer_symbol: str | None = None) -> tuple[int, list[str]]:
    lower = re.sub(r"\s+", " ", str(text).lower())
    hits: list[str] = []
    score = 0
    for pattern in TRANSACTION_PATTERNS:
        if re.search(pattern, lower):
            hits.append(pattern)
            score += 2
    acq_name = normalize_name(acquirer_name)
    if acq_name and len(acq_name) >= 4 and acq_name in normalize_name(lower):
        hits.append("acquirer_name")
        score += 3
    acq_sym = str(acquirer_symbol or "").lower().strip()
    if acq_sym and acq_sym != "-" and re.search(rf"\b{re.escape(acq_sym)}\b", lower):
        hits.append("acquirer_symbol")
        score += 1
    if "item 1.01" in lower and ("merger" in lower or "acquisition" in lower or "tender offer" in lower):
        hits.append("item_1_01_transaction")
        score += 2
    return score, hits


def audit_company_event(
    client: SECClient,
    row: dict,
    lookback_days: int,
    lookahead_days: int,
    max_candidates: int,
    max_document_chars: int,
) -> dict:
    ticker = row["ticker"]
    cik = str(row["cik"]).zfill(10)
    seed_ts = pd.to_datetime(row["seed_event_date"], errors="coerce")
    if ticker == "AEMD":
        return {
            **row,
            "audited_announcement_ts": "2026-09-17 08:03:36+00:00",
            "audit_status": "manual_aemd_sec_8k",
            "audit_score": 99,
            "audit_hits": "manual",
            "audit_form": "8-K",
            "audit_accession": "0001683168-26-007206",
            "audit_document_url": row.get("announcement_url"),
        }
    if pd.isna(seed_ts):
        return {**row, "audit_status": "missing_seed_date"}
    try:
        filings = client.filing_index(cik, forms=MNA_FORMS)
    except Exception as exc:
        return {**row, "audit_status": f"filing_index_error:{type(exc).__name__}"}
    if filings.empty:
        return {**row, "audit_status": "no_candidate_filings"}

    filings["filing_timestamp"] = pd.to_datetime(filings["filing_timestamp"], utc=True, errors="coerce")
    start = pd.Timestamp(seed_ts).tz_localize("UTC") - pd.Timedelta(days=lookback_days)
    end = pd.Timestamp(seed_ts).tz_localize("UTC") + pd.Timedelta(days=lookahead_days)
    candidates = filings[(filings["filing_timestamp"] >= start) & (filings["filing_timestamp"] <= end)].copy()
    if candidates.empty:
        return {**row, "audit_status": "no_candidate_filings_in_window"}

    priority = {
        "8-K": 0,
        "425": 1,
        "SC TO-T": 2,
        "SC TO-I": 2,
        "SC 14D9": 2,
        "S-4": 3,
        "S-4/A": 4,
        "PREM14A": 5,
        "DEFM14A": 6,
        "DEFA14A": 7,
        "PRE 14A": 8,
        "DEF 14A": 9,
    }
    candidates["form_priority"] = candidates["form_type"].map(priority).fillna(99)
    candidates["days_from_seed"] = (end - candidates["filing_timestamp"]).dt.days.abs()
    candidates = candidates.sort_values(["form_priority", "days_from_seed", "filing_timestamp"]).head(max_candidates)

    best_rows = []
    for filing in candidates.sort_values("filing_timestamp").to_dict("records"):
        try:
            path = client.download_filing(cik, filing["accessionNumber"], filing["document_url"])
            text = parse_filing_text(path.read_text(errors="replace")[:max_document_chars])
            score, hits = transaction_score(text, row.get("acquirer_name"), row.get("acquirer"))
            if score > 0:
                best_rows.append((score, hits, filing))
        except Exception:
            continue
    qualified = [(score, hits, filing) for score, hits, filing in best_rows if score >= 3]
    if not qualified:
        return {**row, "audit_status": "no_qualified_transaction_filing"}
    qualified.sort(key=lambda x: (x[2]["filing_timestamp"], -x[0]))
    score, hits, filing = qualified[0]
    return {
        **row,
        "audited_announcement_ts": str(filing["filing_timestamp"]),
        "audit_status": "sec_candidate",
        "audit_score": score,
        "audit_hits": "|".join(hits),
        "audit_form": filing.get("form_type"),
        "audit_accession": filing.get("accessionNumber"),
        "audit_document_url": filing.get("document_url"),
    }


def audit_events(
    events_path: Path,
    out_path: Path,
    cache_dir: Path,
    limit: int | None,
    lookback_days: int,
    lookahead_days: int,
    max_candidates: int,
    max_document_chars: int,
    user_agent: str,
    resume: bool,
) -> pd.DataFrame:
    events = pd.read_csv(events_path, dtype={"cik": str})
    events = events.rename(columns={"symbol": "ticker"})
    if limit:
        events = events.head(limit)
    client = SECClient(user_agent=user_agent, cache_dir=cache_dir)
    checkpoint_path = out_path.with_suffix(".partial.csv")
    rows = []
    done_keys: set[tuple[str, str]] = set()
    if resume and checkpoint_path.exists():
        previous = pd.read_csv(checkpoint_path, dtype={"cik": str})
        rows = previous.to_dict("records")
        for prev in rows:
            done_keys.add((str(prev.get("ticker")), str(prev.get("seed_event_date"))))
        print(f"resuming_completed={len(rows)}")
    for i, row in enumerate(events.to_dict("records"), start=1):
        key = (str(row.get("ticker")), str(row.get("seed_event_date")))
        if key in done_keys:
            continue
        audited = audit_company_event(
            client,
            row,
            lookback_days,
            lookahead_days,
            max_candidates=max_candidates,
            max_document_chars=max_document_chars,
        )
        rows.append(audited)
        pd.DataFrame(rows).to_csv(checkpoint_path, index=False)
        if i % 25 == 0:
            status = pd.Series([r.get("audit_status") for r in rows]).value_counts().to_dict()
            print(f"audited={i} status={status}")
    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    out.to_csv(checkpoint_path, index=False)
    summary = {
        "rows": int(len(out)),
        "status_counts": out["audit_status"].value_counts(dropna=False).to_dict(),
        "audited_rows": int(out["audited_announcement_ts"].notna().sum()) if "audited_announcement_ts" in out else 0,
    }
    out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", default="data/processed/cohort/events_seed_provisional.csv")
    parser.add_argument("--out", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument("--cache-dir", default="data/raw/sec")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lookback-days", type=int, default=420)
    parser.add_argument("--lookahead-days", type=int, default=7)
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-document-chars", type=int, default=2_000_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--user-agent", default="transformative-tx-research/0.1 davidgasper@example.com")
    args = parser.parse_args()
    out = audit_events(
        events_path=Path(args.events),
        out_path=Path(args.out),
        cache_dir=Path(args.cache_dir),
        limit=args.limit,
        lookback_days=args.lookback_days,
        lookahead_days=args.lookahead_days,
        max_candidates=args.max_candidates,
        max_document_chars=args.max_document_chars,
        user_agent=args.user_agent,
        resume=args.resume,
    )
    print(out["audit_status"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
