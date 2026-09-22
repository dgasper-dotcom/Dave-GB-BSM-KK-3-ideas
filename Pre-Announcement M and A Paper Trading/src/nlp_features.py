from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


PHRASE_GROUPS: dict[str, list[str]] = {
    "strategic_alternatives": [
        "strategic alternatives",
        "strategic opportunities",
        "strategic review",
        "exploring strategic alternatives",
        "evaluate strategic alternatives",
        "maximize shareholder value",
    ],
    "investment_bank": [
        "financial advisor",
        "investment bank",
        "investment banker",
        "exclusive financial advisor",
        "transaction advisor",
        "advisory agreement",
    ],
    "change_of_control": [
        "change in control",
        "change of control",
        "severance upon change of control",
        "termination following change of control",
        "acceleration of equity",
        "transaction bonus",
        "retention bonus",
    ],
    "committee": ["special committee", "transaction committee"],
    "unsolicited_interest": [
        "unsolicited proposal",
        "unsolicited indication of interest",
        "unsolicited expression of interest",
        "unsolicited offer",
        "inbound interest",
        "received an offer",
        "received a proposal",
    ],
    "multiple_party_interest": [
        "multiple parties",
        "several parties",
        "various parties",
        "other interested parties",
        "potential bidders",
        "indications of interest",
        "expressions of interest",
    ],
    "transaction": [
        "business combination",
        "merger",
        "acquisition",
        "potential transaction",
        "possible transaction",
        "strategic transaction",
        "sale of the company",
        "sale of substantially all assets",
        "definitive agreement",
        "letter of intent",
        "term sheet",
    ],
    "confidentiality": ["non-binding", "confidential", "confidentiality agreement"],
    "financing": [
        "registered direct offering",
        "private placement",
        "pipe financing",
        "at-the-market offering",
        "convertible note",
        "warrant",
    ],
}


@dataclass(frozen=True)
class PhraseHit:
    group: str
    phrase: str
    count: int
    first_char: int | None


def phrase_hits(text: str, phrase_groups: dict[str, list[str]] = PHRASE_GROUPS) -> list[PhraseHit]:
    normalized = re.sub(r"\s+", " ", str(text).lower())
    hits: list[PhraseHit] = []
    for group, phrases in phrase_groups.items():
        for phrase in phrases:
            pattern = r"\b" + re.escape(phrase.lower()).replace(r"\ ", r"\s+") + r"\b"
            matches = list(re.finditer(pattern, normalized))
            if matches:
                hits.append(PhraseHit(group, phrase, len(matches), matches[0].start()))
    return hits


def filing_phrase_features(filings: pd.DataFrame) -> pd.DataFrame:
    """Create one row per filing with timestamped NLP signals."""

    rows = []
    for row in filings.to_dict("records"):
        hits = phrase_hits(row.get("parsed_text", ""))
        out = {
            "cik": str(row.get("cik")).zfill(10),
            "ticker": row.get("ticker"),
            "accession_number": row.get("accessionNumber") or row.get("accession_number"),
            "form_type": row.get("form_type"),
            "information_available_timestamp": row.get("information_available_timestamp")
            or row.get("filing_timestamp"),
        }
        for group in PHRASE_GROUPS:
            group_hits = [h for h in hits if h.group == group]
            out[f"{group}_hit"] = int(bool(group_hits))
            out[f"{group}_count"] = int(sum(h.count for h in group_hits))
            out[f"{group}_first_char"] = min([h.first_char for h in group_hits if h.first_char is not None], default=np.nan)
        out["any_event_language_hit"] = int(any(h.group != "financing" for h in hits))
        rows.append(out)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["information_available_timestamp"] = pd.to_datetime(
            result["information_available_timestamp"], utc=True
        )
    return result


def rolling_nlp_features(
    filing_features: pd.DataFrame,
    observations: pd.DataFrame,
    windows_days: Iterable[int] = (30, 90, 180),
) -> pd.DataFrame:
    """Aggregate filing-level NLP features into prediction-date features."""

    obs = observations.copy()
    obs["prediction_ts"] = pd.to_datetime(obs["prediction_ts"], utc=True)
    ff = filing_features.copy()
    ff["information_available_timestamp"] = pd.to_datetime(ff["information_available_timestamp"], utc=True)
    feature_cols = [c for c in ff.columns if c.endswith("_hit") or c.endswith("_count")]
    rows = []
    for obs_row in obs.to_dict("records"):
        subset = ff[
            (ff["cik"].astype(str).str.zfill(10) == str(obs_row["cik"]).zfill(10))
            & (ff["information_available_timestamp"] <= obs_row["prediction_ts"])
        ]
        out = {
            "cik": str(obs_row["cik"]).zfill(10),
            "ticker": obs_row.get("ticker"),
            "date": obs_row["date"],
            "prediction_ts": obs_row["prediction_ts"],
            "feature_timestamp": subset["information_available_timestamp"].max()
            if not subset.empty
            else pd.NaT,
        }
        for col in feature_cols:
            if subset.empty:
                out[f"{col}_ever"] = 0
            else:
                out[f"{col}_ever"] = int(subset[col].fillna(0).max()) if col.endswith("_hit") else subset[col].sum()
        for days in windows_days:
            since = obs_row["prediction_ts"] - pd.Timedelta(days=days)
            recent = subset[subset["information_available_timestamp"] > since]
            for col in feature_cols:
                if recent.empty:
                    out[f"{col}_{days}d"] = 0
                else:
                    out[f"{col}_{days}d"] = (
                        int(recent[col].fillna(0).max()) if col.endswith("_hit") else recent[col].sum()
                    )
        for group in PHRASE_GROUPS:
            hit_col = f"{group}_hit"
            appeared = (
                subset[subset[hit_col].fillna(0).astype(int) > 0]
                if hit_col in subset.columns
                else pd.DataFrame()
            )
            out[f"{group}_first_seen_ts"] = (
                appeared["information_available_timestamp"].min() if not appeared.empty else pd.NaT
            )
            out[f"{group}_first_seen_within_90d"] = int(
                pd.notna(out[f"{group}_first_seen_ts"])
                and out["prediction_ts"] - out[f"{group}_first_seen_ts"] <= pd.Timedelta(days=90)
            )
        rows.append(out)
    return pd.DataFrame(rows)
