import pandas as pd

from src.nlp_features import filing_phrase_features, rolling_nlp_features
from src.point_in_time import prediction_timestamp


def test_phrase_features_detect_event_language():
    filings = pd.DataFrame(
        [
            {
                "cik": "882291",
                "ticker": "AEMD",
                "form_type": "8-K",
                "filing_timestamp": pd.Timestamp("2026-03-12 20:00:00Z"),
                "accessionNumber": "x",
                "parsed_text": "The company is exploring strategic alternatives and engaged a financial advisor.",
            }
        ]
    )
    features = filing_phrase_features(filings)
    assert features.loc[0, "strategic_alternatives_hit"] == 1
    assert features.loc[0, "investment_bank_hit"] == 1


def test_phrase_features_detect_review_interest_language():
    filings = pd.DataFrame(
        [
            {
                "cik": "882291",
                "ticker": "AEMD",
                "form_type": "8-K",
                "filing_timestamp": pd.Timestamp("2026-03-12 20:00:00Z"),
                "accessionNumber": "x",
                "parsed_text": (
                    "The board began a strategic review after receiving an unsolicited proposal "
                    "and expressions of interest from multiple parties."
                ),
            }
        ]
    )
    features = filing_phrase_features(filings)
    assert features.loc[0, "strategic_alternatives_hit"] == 1
    assert features.loc[0, "unsolicited_interest_hit"] == 1
    assert features.loc[0, "multiple_party_interest_hit"] == 1


def test_rolling_features_respect_prediction_timestamp():
    filing_features = pd.DataFrame(
        [
            {
                "cik": "0000882291",
                "ticker": "AEMD",
                "information_available_timestamp": pd.Timestamp("2026-03-12 19:00:00Z"),
                "strategic_alternatives_hit": 1,
                "strategic_alternatives_count": 1,
                "investment_bank_hit": 0,
                "investment_bank_count": 0,
                "change_of_control_hit": 0,
                "change_of_control_count": 0,
                "committee_hit": 0,
                "committee_count": 0,
                "transaction_hit": 0,
                "transaction_count": 0,
                "confidentiality_hit": 0,
                "confidentiality_count": 0,
                "financing_hit": 0,
                "financing_count": 0,
                "any_event_language_hit": 1,
            }
        ]
    )
    obs = pd.DataFrame(
        [
            {"cik": "0000882291", "ticker": "AEMD", "date": pd.Timestamp("2026-03-11"), "prediction_ts": prediction_timestamp("2026-03-11")},
            {"cik": "0000882291", "ticker": "AEMD", "date": pd.Timestamp("2026-03-12"), "prediction_ts": prediction_timestamp("2026-03-12")},
        ]
    )
    out = rolling_nlp_features(filing_features, obs)
    assert out.loc[0, "strategic_alternatives_hit_ever"] == 0
    assert out.loc[1, "strategic_alternatives_hit_ever"] == 1
