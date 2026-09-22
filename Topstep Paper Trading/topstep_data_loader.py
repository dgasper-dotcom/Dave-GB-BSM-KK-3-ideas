"""Futures OHLCV data loading and validation utilities."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


TIMESTAMP_FORMATS = (
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%dT%H:%M:%S",
)


def parse_decimal(value: str) -> Decimal:
    return Decimal(str(value).strip())


def parse_timestamp(value: str) -> datetime:
    raw = str(value).strip()
    for fmt in TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass
    raise ValueError(f"unsupported timestamp format: {value!r}")


@dataclass(frozen=True)
class FuturesBar:
    timestamp: datetime
    symbol: str
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    vwap_rth: Optional[Decimal] = None
    vwap_eth: Optional[Decimal] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        if self.high < max(self.open, self.low, self.close):
            raise ValueError(f"invalid OHLC high at {self.timestamp}: {self}")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError(f"invalid OHLC low at {self.timestamp}: {self}")
        if self.volume < 0:
            raise ValueError(f"negative volume at {self.timestamp}: {self.volume}")


@dataclass(frozen=True)
class CSVBarSchema:
    timestamp_col: str
    open_col: str = "open"
    high_col: str = "high"
    low_col: str = "low"
    close_col: str = "close"
    volume_col: str = "volume"
    symbol_col: Optional[str] = None
    vwap_rth_col: Optional[str] = None
    vwap_eth_col: Optional[str] = None
    timezone_label: str = "America/New_York"


@dataclass(frozen=True)
class DataQualityReport:
    path: str
    symbol: str
    rows: int
    first_timestamp: Optional[datetime]
    last_timestamp: Optional[datetime]
    bad_parse_rows: int
    bad_ohlc_rows: int
    duplicate_timestamps: int
    non_monotonic_rows: int
    zero_volume_rows: int
    negative_volume_rows: int
    rows_by_year: Mapping[int, int] = field(default_factory=dict)
    gap_minutes: Mapping[int, int] = field(default_factory=dict)
    largest_gap_minutes: Optional[int] = None
    largest_gap_start: Optional[datetime] = None
    largest_gap_end: Optional[datetime] = None

    @property
    def is_clean(self) -> bool:
        return (
            self.bad_parse_rows == 0
            and self.bad_ohlc_rows == 0
            and self.duplicate_timestamps == 0
            and self.non_monotonic_rows == 0
            and self.negative_volume_rows == 0
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "rows": self.rows,
            "first_timestamp": self.first_timestamp.isoformat(sep=" ")
            if self.first_timestamp
            else None,
            "last_timestamp": self.last_timestamp.isoformat(sep=" ")
            if self.last_timestamp
            else None,
            "bad_parse_rows": self.bad_parse_rows,
            "bad_ohlc_rows": self.bad_ohlc_rows,
            "duplicate_timestamps": self.duplicate_timestamps,
            "non_monotonic_rows": self.non_monotonic_rows,
            "zero_volume_rows": self.zero_volume_rows,
            "negative_volume_rows": self.negative_volume_rows,
            "rows_by_year": dict(sorted(self.rows_by_year.items())),
            "gap_minutes": dict(sorted(self.gap_minutes.items())),
            "largest_gap_minutes": self.largest_gap_minutes,
            "largest_gap_start": self.largest_gap_start.isoformat(sep=" ")
            if self.largest_gap_start
            else None,
            "largest_gap_end": self.largest_gap_end.isoformat(sep=" ")
            if self.largest_gap_end
            else None,
            "is_clean": self.is_clean,
        }


def detect_csv_schema(path: Path) -> CSVBarSchema:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV file: {path}") from exc

    header_set = set(headers)
    if {
        "timestamp ET",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }.issubset(header_set):
        return CSVBarSchema(
            timestamp_col="timestamp ET",
            vwap_rth_col="Vwap_RTH" if "Vwap_RTH" in header_set else None,
            vwap_eth_col="Vwap_ETH" if "Vwap_ETH" in header_set else None,
            timezone_label="America/New_York",
        )

    if {"timestamp", "symbol", "open", "high", "low", "close", "volume"}.issubset(
        header_set
    ):
        return CSVBarSchema(
            timestamp_col="timestamp",
            symbol_col="symbol",
            timezone_label="America/New_York",
        )

    raise ValueError(f"unrecognized OHLCV CSV schema in {path}; headers={headers}")


def iter_futures_bars(
    path: str,
    symbol: Optional[str] = None,
    schema: Optional[CSVBarSchema] = None,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Iterator[FuturesBar]:
    csv_path = Path(path)
    schema = schema or detect_csv_schema(csv_path)
    fixed_symbol = symbol.upper() if symbol else None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_symbol = (
                row[schema.symbol_col].upper()
                if schema.symbol_col is not None
                else fixed_symbol
            )
            if not row_symbol:
                raise ValueError("symbol is required when CSV has no symbol column")

            ts = parse_timestamp(row[schema.timestamp_col])
            if start is not None and ts < start:
                continue
            if end is not None and ts > end:
                continue

            yield FuturesBar(
                timestamp=ts,
                symbol=row_symbol,
                open=parse_decimal(row[schema.open_col]),
                high=parse_decimal(row[schema.high_col]),
                low=parse_decimal(row[schema.low_col]),
                close=parse_decimal(row[schema.close_col]),
                volume=parse_decimal(row[schema.volume_col]),
                vwap_rth=_optional_decimal(row, schema.vwap_rth_col),
                vwap_eth=_optional_decimal(row, schema.vwap_eth_col),
            )


def validate_ohlcv_csv(
    path: str,
    symbol: Optional[str] = None,
    schema: Optional[CSVBarSchema] = None,
) -> DataQualityReport:
    csv_path = Path(path)
    schema = schema or detect_csv_schema(csv_path)
    fixed_symbol = (symbol or "").upper()
    rows = 0
    bad_parse_rows = 0
    bad_ohlc_rows = 0
    duplicate_timestamps = 0
    non_monotonic_rows = 0
    zero_volume_rows = 0
    negative_volume_rows = 0
    rows_by_year: Counter[int] = Counter()
    gap_minutes: Counter[int] = Counter()
    seen = set()
    first_timestamp = None
    last_timestamp = None
    prev_timestamp = None
    largest_gap: Optional[Tuple[int, datetime, datetime]] = None

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows += 1
            try:
                row_symbol = (
                    row[schema.symbol_col].upper()
                    if schema.symbol_col is not None
                    else fixed_symbol
                )
                if not row_symbol:
                    raise ValueError("missing symbol")
                ts = parse_timestamp(row[schema.timestamp_col])
                o = parse_decimal(row[schema.open_col])
                h = parse_decimal(row[schema.high_col])
                l = parse_decimal(row[schema.low_col])
                c = parse_decimal(row[schema.close_col])
                volume = parse_decimal(row[schema.volume_col])
            except Exception:
                bad_parse_rows += 1
                continue

            if first_timestamp is None:
                first_timestamp = ts
            last_timestamp = ts

            if ts in seen:
                duplicate_timestamps += 1
            seen.add(ts)

            if prev_timestamp is not None:
                delta_minutes = int((ts - prev_timestamp).total_seconds() // 60)
                if delta_minutes <= 0:
                    non_monotonic_rows += 1
                elif delta_minutes != 1:
                    gap_minutes[delta_minutes] += 1
                    if largest_gap is None or delta_minutes > largest_gap[0]:
                        largest_gap = (delta_minutes, prev_timestamp, ts)
            prev_timestamp = ts

            if h < max(o, l, c) or l > min(o, h, c):
                bad_ohlc_rows += 1
            if volume == 0:
                zero_volume_rows += 1
            if volume < 0:
                negative_volume_rows += 1

            rows_by_year[ts.year] += 1

    return DataQualityReport(
        path=str(csv_path),
        symbol=fixed_symbol or "CSV_SYMBOL",
        rows=rows,
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        bad_parse_rows=bad_parse_rows,
        bad_ohlc_rows=bad_ohlc_rows,
        duplicate_timestamps=duplicate_timestamps,
        non_monotonic_rows=non_monotonic_rows,
        zero_volume_rows=zero_volume_rows,
        negative_volume_rows=negative_volume_rows,
        rows_by_year=dict(rows_by_year),
        gap_minutes=dict(gap_minutes),
        largest_gap_minutes=largest_gap[0] if largest_gap else None,
        largest_gap_start=largest_gap[1] if largest_gap else None,
        largest_gap_end=largest_gap[2] if largest_gap else None,
    )


def write_normalized_csv(
    input_path: str,
    output_path: str,
    symbol: Optional[str] = None,
    schema: Optional[CSVBarSchema] = None,
) -> int:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap_rth",
                "vwap_eth",
            ]
        )
        for bar in iter_futures_bars(input_path, symbol=symbol, schema=schema):
            writer.writerow(
                [
                    bar.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    bar.symbol,
                    str(bar.open),
                    str(bar.high),
                    str(bar.low),
                    str(bar.close),
                    str(bar.volume),
                    "" if bar.vwap_rth is None else str(bar.vwap_rth),
                    "" if bar.vwap_eth is None else str(bar.vwap_eth),
                ]
            )
            rows += 1
    return rows


def resample_bars(bars: Iterable[FuturesBar], minutes: int) -> Iterator[FuturesBar]:
    if minutes <= 0:
        raise ValueError("minutes must be positive")

    current_key = None
    current_bar = None
    current_volume = Decimal("0")

    for bar in bars:
        bucket = _bucket_timestamp(bar.timestamp, minutes)
        key = (bar.symbol, bucket)
        if current_key is None:
            current_key = key
            current_bar = _bar_with_timestamp(bar, bucket)
            current_volume = bar.volume
            continue

        if key != current_key:
            assert current_bar is not None
            yield current_bar
            current_key = key
            current_bar = _bar_with_timestamp(bar, bucket)
            current_volume = bar.volume
            continue

        assert current_bar is not None
        current_volume += bar.volume
        current_bar = FuturesBar(
            timestamp=current_bar.timestamp,
            symbol=current_bar.symbol,
            open=current_bar.open,
            high=max(current_bar.high, bar.high),
            low=min(current_bar.low, bar.low),
            close=bar.close,
            volume=current_volume,
        )

    if current_bar is not None:
        yield current_bar


def _bucket_timestamp(ts: datetime, minutes: int) -> datetime:
    minute = (ts.minute // minutes) * minutes
    return ts.replace(minute=minute, second=0, microsecond=0)


def _bar_with_timestamp(bar: FuturesBar, timestamp: datetime) -> FuturesBar:
    return FuturesBar(
        timestamp=timestamp,
        symbol=bar.symbol,
        open=bar.open,
        high=bar.high,
        low=bar.low,
        close=bar.close,
        volume=bar.volume,
        vwap_rth=bar.vwap_rth,
        vwap_eth=bar.vwap_eth,
    )


def _optional_decimal(row: Mapping[str, str], col: Optional[str]) -> Optional[Decimal]:
    if col is None:
        return None
    raw = row.get(col, "")
    if raw == "":
        return None
    return parse_decimal(raw)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--path", required=True)
    validate_parser.add_argument("--symbol", required=True)

    normalize_parser = subparsers.add_parser("normalize")
    normalize_parser.add_argument("--input", required=True)
    normalize_parser.add_argument("--output", required=True)
    normalize_parser.add_argument("--symbol", required=True)

    args = parser.parse_args(argv)
    if args.command == "validate":
        report = validate_ohlcv_csv(args.path, symbol=args.symbol)
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        return 0 if report.is_clean else 1
    if args.command == "normalize":
        rows = write_normalized_csv(args.input, args.output, symbol=args.symbol)
        print(json.dumps({"output": args.output, "rows": rows}, indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(_main())
