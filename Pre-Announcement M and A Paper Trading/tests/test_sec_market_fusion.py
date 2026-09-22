import pandas as pd

from src.sec_market_fusion import select_company_filings


def filing_frame(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filing_timestamp": pd.date_range("2018-01-01", periods=count, freq="QS", tz="UTC"),
            "accessionNumber": [f"000-{i:03d}" for i in range(count)],
        }
    )


def test_select_company_filings_recent_keeps_newest_rows():
    out = select_company_filings(filing_frame(8), 3, filing_selection="recent")
    assert out["accessionNumber"].tolist() == ["000-005", "000-006", "000-007"]


def test_select_company_filings_even_spreads_rows_across_history():
    out = select_company_filings(filing_frame(9), 5, filing_selection="even")
    assert out["accessionNumber"].tolist() == ["000-000", "000-002", "000-004", "000-006", "000-008"]
