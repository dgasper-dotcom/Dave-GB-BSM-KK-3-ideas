from __future__ import annotations

import argparse
import json
import re
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4 import XMLParsedAsHTMLWarning


SEC_ARCHIVES = "https://www.sec.gov/Archives/edgar/data"
SEC_DATA = "https://data.sec.gov/submissions"


@dataclass
class SECClient:
    user_agent: str
    cache_dir: Path
    min_request_interval_seconds: float = 0.12

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )

    def _get(self, url: str) -> bytes:
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.min_request_interval_seconds:
            time.sleep(self.min_request_interval_seconds - elapsed)
        response = self.session.get(url, timeout=30)
        self._last_request = time.monotonic()
        response.raise_for_status()
        return response.content

    def submissions_json(self, cik: str) -> dict:
        cik10 = str(cik).zfill(10)
        path = self.cache_dir / "submissions" / f"CIK{cik10}.json"
        if path.exists():
            return json.loads(path.read_text())
        path.parent.mkdir(parents=True, exist_ok=True)
        url = f"{SEC_DATA}/CIK{cik10}.json"
        data = self._get(url)
        path.write_bytes(data)
        return json.loads(data)

    def filing_index(self, cik: str, forms: Iterable[str] | None = None) -> pd.DataFrame:
        payload = self.submissions_json(cik)
        recent = payload["filings"]["recent"]
        frame = pd.DataFrame(recent)
        frame["cik"] = str(cik).zfill(10)
        frame["ticker"] = (payload.get("tickers") or [None])[0]
        frame["company_name"] = payload.get("name")
        frame["acceptanceDateTime"] = pd.to_datetime(frame["acceptanceDateTime"], utc=True, errors="coerce")
        frame["filingDate"] = pd.to_datetime(frame["filingDate"], errors="coerce")
        frame["accession_clean"] = frame["accessionNumber"].str.replace("-", "", regex=False)
        frame["primary_document_url"] = frame.apply(
            lambda r: f"{SEC_ARCHIVES}/{int(r.cik)}/{r.accession_clean}/{r.primaryDocument}", axis=1
        )
        if forms:
            frame = frame[frame["form"].isin(set(forms))].copy()
        return frame[
            [
                "cik",
                "ticker",
                "company_name",
                "form",
                "filingDate",
                "acceptanceDateTime",
                "accessionNumber",
                "primaryDocument",
                "primary_document_url",
            ]
        ].rename(
            columns={
                "form": "form_type",
                "filingDate": "filing_date",
                "acceptanceDateTime": "filing_timestamp",
                "primary_document_url": "document_url",
            }
        )

    def download_filing(self, cik: str, accession_number: str, document_url: str) -> Path:
        cik10 = str(cik).zfill(10)
        out = self.cache_dir / "filings" / cik10 / f"{accession_number}.html"
        if out.exists():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(self._get(document_url))
        return out


def parse_filing_text(raw_html: str) -> str:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", XMLParsedAsHTMLWarning)
        soup = BeautifulSoup(raw_html, "lxml")
    for tag in soup(["script", "style", "ix:header", "header", "footer"]):
        tag.extract()
    text = soup.get_text(" ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ingest_cik(cik: str, out_dir: Path, forms: list[str], user_agent: str) -> pd.DataFrame:
    client = SECClient(user_agent=user_agent, cache_dir=out_dir)
    idx = client.filing_index(cik, forms=forms)
    rows = []
    for row in idx.to_dict("records"):
        path = client.download_filing(row["cik"], row["accessionNumber"], row["document_url"])
        raw = path.read_text(errors="replace")
        row["raw_path"] = str(path)
        row["parsed_text"] = parse_filing_text(raw)
        rows.append(row)
    result = pd.DataFrame(rows)
    if not result.empty:
        parquet_path = out_dir / f"filings_{str(cik).zfill(10)}.parquet"
        csv_path = out_dir / f"filings_{str(cik).zfill(10)}.csv"
        try:
            result.to_parquet(parquet_path, index=False)
        except Exception as exc:
            result.to_csv(csv_path, index=False)
            print(f"parquet_unavailable={type(exc).__name__}; wrote_csv={csv_path}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cik", required=True)
    parser.add_argument("--out", default="data/raw/sec")
    parser.add_argument("--forms", nargs="*", default=["8-K", "10-Q", "10-K"])
    parser.add_argument("--user-agent", default="transformative-tx-research/0.1 contact@example.com")
    args = parser.parse_args()
    frame = ingest_cik(args.cik, Path(args.out), args.forms, args.user_agent)
    print(f"downloaded_or_loaded_filings={len(frame)}")


if __name__ == "__main__":
    main()
