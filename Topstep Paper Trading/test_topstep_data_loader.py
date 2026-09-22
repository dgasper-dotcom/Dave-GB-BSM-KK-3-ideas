import csv
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from topstep_data_loader import (
    FuturesBar,
    detect_csv_schema,
    iter_futures_bars,
    parse_timestamp,
    resample_bars,
    validate_ohlcv_csv,
    write_normalized_csv,
)
from topstep_rule_simulator import money


class TopstepDataLoaderTest(unittest.TestCase):
    def write_csv(self, directory, name, rows):
        path = Path(directory) / name
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(rows)
        return path

    def test_detects_kaggle_nq_schema_and_iterates_bars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(
                tmp,
                "nq.csv",
                [
                    [
                        "timestamp ET",
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "Vwap_RTH",
                        "Vwap_ETH",
                    ],
                    ["12/26/2022 18:01", "13759", "13794.75", "13759", "13788.5", "540", "0", "13780.75"],
                    ["12/26/2022 18:02", "13790.25", "13794", "13784", "13790.25", "304", "0", "13783.87164"],
                ],
            )

            schema = detect_csv_schema(path)
            bars = list(iter_futures_bars(str(path), symbol="NQ", schema=schema))

            self.assertEqual(schema.timestamp_col, "timestamp ET")
            self.assertEqual(len(bars), 2)
            self.assertEqual(bars[0].symbol, "NQ")
            self.assertEqual(bars[0].timestamp, datetime(2022, 12, 26, 18, 1))
            self.assertEqual(bars[0].open, money(13759))
            self.assertEqual(bars[0].high, money("13794.75"))
            self.assertEqual(bars[0].volume, money(540))
            self.assertEqual(bars[0].vwap_eth, money("13780.75"))

    def test_requires_symbol_when_csv_has_no_symbol_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(
                tmp,
                "nq.csv",
                [
                    ["timestamp ET", "open", "high", "low", "close", "volume"],
                    ["12/26/2022 18:01", "13759", "13760", "13758", "13759", "540"],
                ],
            )

            with self.assertRaises(ValueError):
                list(iter_futures_bars(str(path)))

    def test_detects_standard_normalized_schema_with_symbol_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(
                tmp,
                "normalized.csv",
                [
                    ["timestamp", "symbol", "open", "high", "low", "close", "volume"],
                    ["2024-01-02 09:30:00", "NQ", "17000", "17010", "16990", "17005", "100"],
                ],
            )

            bars = list(iter_futures_bars(str(path)))

            self.assertEqual(len(bars), 1)
            self.assertEqual(bars[0].symbol, "NQ")
            self.assertEqual(bars[0].timestamp, datetime(2024, 1, 2, 9, 30))

    def test_quality_report_flags_duplicates_non_monotonic_and_bad_ohlc(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(
                tmp,
                "bad.csv",
                [
                    ["timestamp ET", "open", "high", "low", "close", "volume"],
                    ["01/02/2024 09:30", "100", "101", "99", "100", "10"],
                    ["01/02/2024 09:31", "100", "99", "98", "100", "10"],
                    ["01/02/2024 09:31", "100", "101", "99", "100", "10"],
                    ["01/02/2024 09:30", "100", "101", "99", "100", "-1"],
                ],
            )

            report = validate_ohlcv_csv(str(path), symbol="NQ")

            self.assertEqual(report.rows, 4)
            self.assertEqual(report.bad_ohlc_rows, 1)
            self.assertEqual(report.duplicate_timestamps, 2)
            self.assertEqual(report.non_monotonic_rows, 2)
            self.assertEqual(report.negative_volume_rows, 1)
            self.assertFalse(report.is_clean)

    def test_quality_report_counts_gaps_and_years(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_csv(
                tmp,
                "gaps.csv",
                [
                    ["timestamp ET", "open", "high", "low", "close", "volume"],
                    ["12/31/2023 23:59", "100", "101", "99", "100", "10"],
                    ["01/01/2024 00:02", "100", "101", "99", "100", "10"],
                ],
            )

            report = validate_ohlcv_csv(str(path), symbol="NQ")

            self.assertTrue(report.is_clean)
            self.assertEqual(report.rows_by_year, {2023: 1, 2024: 1})
            self.assertEqual(report.gap_minutes, {3: 1})
            self.assertEqual(report.largest_gap_minutes, 3)

    def test_write_normalized_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = self.write_csv(
                tmp,
                "raw.csv",
                [
                    ["timestamp ET", "open", "high", "low", "close", "volume", "Vwap_ETH"],
                    ["12/26/2022 18:01", "13759", "13760", "13758", "13759.5", "540", "13759.25"],
                ],
            )
            normalized = Path(tmp) / "normalized.csv"

            rows = write_normalized_csv(str(raw), str(normalized), symbol="NQ")

            self.assertEqual(rows, 1)
            lines = normalized.read_text().splitlines()
            self.assertEqual(
                lines[0],
                "timestamp,symbol,open,high,low,close,volume,vwap_rth,vwap_eth",
            )
            self.assertEqual(
                lines[1],
                "2022-12-26 18:01:00,NQ,13759,13760,13758,13759.5,540,,13759.25",
            )

    def test_resample_bars_aggregates_ohlcv_by_bucket(self):
        bars = [
            FuturesBar(parse_timestamp("2024-01-02 09:31"), "NQ", money(100), money(101), money(99), money(100.5), money(10)),
            FuturesBar(parse_timestamp("2024-01-02 09:32"), "NQ", money(100.5), money(103), money(100), money(102), money(12)),
            FuturesBar(parse_timestamp("2024-01-02 09:33"), "NQ", money(102), money(102.5), money(98), money(99), money(8)),
            FuturesBar(parse_timestamp("2024-01-02 09:34"), "NQ", money(99), money(100), money(97), money(98), money(6)),
        ]

        resampled = list(resample_bars(bars, minutes=3))

        self.assertEqual(len(resampled), 2)
        self.assertEqual(resampled[0].timestamp, datetime(2024, 1, 2, 9, 30))
        self.assertEqual(resampled[0].open, money(100))
        self.assertEqual(resampled[0].high, money(103))
        self.assertEqual(resampled[0].low, money(99))
        self.assertEqual(resampled[0].close, money(102))
        self.assertEqual(resampled[0].volume, money(22))
        self.assertEqual(resampled[1].timestamp, datetime(2024, 1, 2, 9, 33))
        self.assertEqual(resampled[1].open, money(102))
        self.assertEqual(resampled[1].high, money(102.5))
        self.assertEqual(resampled[1].low, money(97))
        self.assertEqual(resampled[1].close, money(98))
        self.assertEqual(resampled[1].volume, money(14))


if __name__ == "__main__":
    unittest.main()
