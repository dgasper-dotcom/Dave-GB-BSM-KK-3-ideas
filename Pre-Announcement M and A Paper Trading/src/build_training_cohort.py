from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests


STOCKANALYSIS_ACQ_URL = "https://stockanalysis.com/actions/acquisitions/{year}/"
STOCKANALYSIS_RECENT_ACQ_URL = "https://stockanalysis.com/actions/acquisitions/"
STOCKANALYSIS_COMPANY_URL = "https://stockanalysis.com/stocks/{symbol}/company/"
SEC_TICKERS_EXCHANGE_URL = "https://www.sec.gov/files/company_tickers_exchange.json"


BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
}

SEC_HEADERS = {
    "User-Agent": "transformative-tx-research/0.1 davidgasper@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json,text/plain,*/*",
}


@dataclass(frozen=True)
class CohortConfig:
    start_year: int
    end_year: int
    target_positives: int
    controls_per_positive: int
    seed: int
    sleep_seconds: float


def get_text(url: str, cache_path: Path | None = None, sleep_seconds: float = 0.0) -> str:
    if cache_path and cache_path.exists():
        return cache_path.read_text()
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=30)
    response.raise_for_status()
    if sleep_seconds:
        time.sleep(sleep_seconds)
    text = response.text
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text)
    return text


def fetch_stockanalysis_acquisitions(
    years: Iterable[int],
    cache_dir: Path,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    frames = []
    urls = [STOCKANALYSIS_RECENT_ACQ_URL] + [STOCKANALYSIS_ACQ_URL.format(year=y) for y in years]
    for url in urls:
        cache_name = url.rstrip("/").replace("https://", "").replace("/", "_") + ".html"
        html = get_text(url, cache_dir / "stockanalysis" / cache_name, sleep_seconds=sleep_seconds)
        tables = pd.read_html(StringIO(html))
        if not tables:
            continue
        frame = tables[0].copy()
        frame["source_url"] = url
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    out["symbol"] = out["symbol"].astype(str).str.upper().str.strip()
    out["acquirer"] = out["acquirer"].astype(str).str.upper().str.strip()
    out["event_date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["event_date", "symbol"])
    out = out.sort_values("event_date", ascending=False)
    out = out.drop_duplicates(["symbol", "event_date"], keep="first")
    return out


def sec_ticker_map(cache_dir: Path, sleep_seconds: float = 0.0) -> pd.DataFrame:
    path = cache_dir / "sec" / "company_tickers_exchange.json"
    if path.exists():
        payload = json.loads(path.read_text())
    else:
        response = requests.get(SEC_TICKERS_EXCHANGE_URL, headers=SEC_HEADERS, timeout=30)
        response.raise_for_status()
        if sleep_seconds:
            time.sleep(sleep_seconds)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(response.text)
        payload = response.json()
    frame = pd.DataFrame(payload["data"], columns=payload["fields"])
    frame["ticker"] = frame["ticker"].astype(str).str.upper()
    frame["cik"] = frame["cik"].astype(str).str.zfill(10)
    return frame


def resolve_cik_from_stockanalysis(symbol: str, cache_dir: Path, sleep_seconds: float = 0.0) -> str | None:
    safe = re.sub(r"[^A-Z0-9.-]", "", symbol.upper())
    if not safe:
        return None
    url = STOCKANALYSIS_COMPANY_URL.format(symbol=safe.lower().replace(".", "-"))
    cache_path = cache_dir / "stockanalysis_company" / f"{safe}.html"
    try:
        html = get_text(url, cache_path, sleep_seconds=sleep_seconds)
    except requests.HTTPError:
        return None
    match = re.search(r"CIK Code</td><td[^>]*>(\d{1,10})</td>", html)
    if not match:
        match = re.search(r'cik:"(\d{1,10})"', html)
    if not match:
        return None
    return match.group(1).zfill(10)


def resolve_positive_ciks(events: pd.DataFrame, sec_map: pd.DataFrame, cache_dir: Path, sleep_seconds: float) -> pd.DataFrame:
    ticker_to_cik = dict(zip(sec_map["ticker"], sec_map["cik"]))
    rows = []
    for event in events.to_dict("records"):
        symbol = event["symbol"]
        cik = ticker_to_cik.get(symbol)
        source = "sec_company_tickers_exchange"
        if not cik:
            cik = resolve_cik_from_stockanalysis(symbol, cache_dir, sleep_seconds=sleep_seconds)
            source = "stockanalysis_company_page" if cik else None
        event["cik"] = cik
        event["cik_source"] = source
        rows.append(event)
    return pd.DataFrame(rows)


def add_aemd_seed(events: pd.DataFrame) -> pd.DataFrame:
    aemd = {
        "date": "Sep 17, 2026",
        "symbol": "AEMD",
        "company_name": "Aethlon Medical Inc",
        "acquirer": "NRTX",
        "acquirer_name": "North Immunology",
        "source_url": "https://www.sec.gov/Archives/edgar/data/882291/000168316826007206/0001683168-26-007206-index.htm",
        "event_date": pd.Timestamp("2026-09-17"),
        "cik": "0000882291",
        "cik_source": "manual_sec_8k",
    }
    out = pd.concat([pd.DataFrame([aemd]), events], ignore_index=True)
    return out.drop_duplicates(["symbol", "event_date"], keep="first")


def build_controls(sec_map: pd.DataFrame, positive_symbols: set[str], n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    eligible = sec_map[
        sec_map["exchange"].isin(["Nasdaq", "NYSE", "NYSE American"])
        & ~sec_map["ticker"].isin(positive_symbols)
    ].copy()
    eligible = eligible.drop_duplicates("ticker")
    records = eligible.to_dict("records")
    rng.shuffle(records)
    records = records[: min(n, len(records))]
    controls = pd.DataFrame(records).rename(columns={"ticker": "symbol", "name": "company_name"})
    controls["event_date"] = pd.NaT
    controls["event_type"] = "NONE"
    controls["label_quality"] = "control_no_seeded_acquisition_in_stockanalysis_pull"
    controls["source_url"] = SEC_TICKERS_EXCHANGE_URL
    controls["cik_source"] = "sec_company_tickers_exchange"
    return controls[["symbol", "cik", "company_name", "exchange", "event_type", "event_date", "label_quality", "source_url", "cik_source"]]


def build_cohort(config: CohortConfig, out_dir: Path, cache_dir: Path) -> dict[str, object]:
    years = range(config.start_year, config.end_year + 1)
    acquisitions = fetch_stockanalysis_acquisitions(years, cache_dir, sleep_seconds=config.sleep_seconds)
    sec_map = sec_ticker_map(cache_dir, sleep_seconds=config.sleep_seconds)
    acquisitions = acquisitions.head(config.target_positives * 3)
    positives = resolve_positive_ciks(acquisitions, sec_map, cache_dir, sleep_seconds=config.sleep_seconds)
    positives = positives[positives["cik"].notna()].head(config.target_positives)
    positives = add_aemd_seed(positives)
    positives["exchange"] = positives["symbol"].map(dict(zip(sec_map["ticker"], sec_map["exchange"])))
    positives["event_type"] = "ACQUISITION"
    positives.loc[positives["symbol"].eq("AEMD"), "event_type"] = "MERGER"
    positives["label_quality"] = "provisional_stockanalysis_action_date_not_first_announcement"
    positives.loc[positives["symbol"].eq("AEMD"), "label_quality"] = "audited_sec_8k_first_public_announcement"

    positive_symbols = set(positives["symbol"])
    control_count = len(positives) * config.controls_per_positive
    controls = build_controls(sec_map, positive_symbols, control_count, config.seed)

    positive_cols = ["symbol", "cik", "company_name", "exchange", "event_type", "event_date", "label_quality", "source_url", "cik_source", "acquirer", "acquirer_name"]
    for col in positive_cols:
        if col not in positives.columns:
            positives[col] = None
    positives_out = positives[positive_cols].copy()
    controls["acquirer"] = None
    controls["acquirer_name"] = None
    cohort = pd.concat([positives_out, controls[positive_cols]], ignore_index=True)
    cohort["event_date"] = pd.to_datetime(cohort["event_date"], errors="coerce").dt.date.astype("string")
    cohort["dataset_role"] = cohort["event_type"].where(cohort["event_type"].ne("NONE"), "CONTROL")

    events_seed = positives_out.copy()
    events_seed["announcement_ts"] = pd.to_datetime(events_seed["event_date"], errors="coerce").dt.strftime("%Y-%m-%d 12:00:00+00:00")
    events_seed = events_seed.rename(
        columns={
            "symbol": "ticker",
            "event_date": "seed_event_date",
            "source_url": "announcement_url",
        }
    )
    events_seed["announcement_source"] = events_seed["label_quality"]

    out_dir.mkdir(parents=True, exist_ok=True)
    positives_out.to_csv(out_dir / "positive_acquisition_seed_companies.csv", index=False)
    controls.to_csv(out_dir / "control_seed_companies.csv", index=False)
    cohort.to_csv(out_dir / "training_company_cohort.csv", index=False)
    events_seed.to_csv(out_dir / "events_seed_provisional.csv", index=False)

    summary = {
        "positive_companies": int(len(positives_out)),
        "control_companies": int(len(controls)),
        "total_companies": int(len(cohort)),
        "positive_date_min": str(pd.to_datetime(positives_out["event_date"]).min().date()),
        "positive_date_max": str(pd.to_datetime(positives_out["event_date"]).max().date()),
        "unresolved_positive_seed_rows": int(len(acquisitions) - len(positives[positives["symbol"].ne("AEMD")])),
        "stockanalysis_rows_pulled": int(len(acquisitions)),
        "sec_ticker_rows": int(len(sec_map)),
        "warning": "StockAnalysis acquisition dates are action/closing-style seeds, not audited first announcement timestamps.",
    }
    (out_dir / "training_company_cohort_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2021)
    parser.add_argument("--end-year", type=int, default=2026)
    parser.add_argument("--target-positives", type=int, default=250)
    parser.add_argument("--controls-per-positive", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument("--out-dir", default="data/processed/cohort")
    parser.add_argument("--cache-dir", default="data/raw/cohort_sources")
    args = parser.parse_args()
    summary = build_cohort(
        CohortConfig(
            start_year=args.start_year,
            end_year=args.end_year,
            target_positives=args.target_positives,
            controls_per_positive=args.controls_per_positive,
            seed=args.seed,
            sleep_seconds=args.sleep_seconds,
        ),
        out_dir=Path(args.out_dir),
        cache_dir=Path(args.cache_dir),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
