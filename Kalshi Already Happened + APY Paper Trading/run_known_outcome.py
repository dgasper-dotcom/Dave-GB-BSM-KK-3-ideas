#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kalshi_mlb_backtest.known_outcome import (
    DEFAULT_KNOWN_OUTCOME_SERIES,
    KnownOutcomeConfig,
    scan_known_outcomes,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scan Kalshi markets for already-known outcomes and APY-adjusted paper opportunities."
    )
    parser.add_argument(
        "command",
        choices=["scan", "known-outcome-scan"],
        nargs="?",
        default="scan",
        help="Run one known-outcome scan.",
    )
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--timezone", default="America/New_York")
    parser.add_argument("--capture-date", help="YYYY-MM-DD local date. Defaults to today in --timezone.")
    parser.add_argument("--no-previous-date", action="store_true", help="Do not include the previous local date.")
    parser.add_argument("--contracts", type=float, default=100.0, help="Maximum contracts per paper entry.")
    parser.add_argument("--min-contracts", type=float, default=1.0, help="Minimum displayed contracts required.")
    parser.add_argument("--require-full-contracts", action="store_true", help="Require full --contracts size to be fillable.")
    parser.add_argument("--max-ask", type=float, default=0.99, help="Maximum executable ask eligible for entry.")
    parser.add_argument("--orderbook-depth", type=int, default=100)
    parser.add_argument("--max-markets", type=int, help="Development throttle: scan first N supported markets.")
    parser.add_argument("--max-market-pages", type=int, help="Development throttle: scan first N Kalshi pages per series.")
    parser.add_argument("--kalshi-api-key-id", help="Kalshi API key id. Defaults to KALSHI_ACCESS_KEY/KALSHI_API_KEY_ID.")
    parser.add_argument("--kalshi-private-key-file", help="Kalshi private key PEM path.")
    parser.add_argument("--known-outcome-series-ticker", default=DEFAULT_KNOWN_OUTCOME_SERIES)
    parser.add_argument("--known-outcome-lookback-days", type=int, default=3)
    parser.add_argument("--known-outcome-apy", type=float, default=0.0325)
    parser.add_argument("--known-outcome-fallback-settlement-hours", type=float, default=24.0)
    parser.add_argument("--known-outcome-min-net-profit-per-contract", type=float, default=0.001)
    parser.add_argument("--known-outcome-orderbook-workers", type=int, default=2)
    parser.add_argument("--known-outcome-disable-final-no", action="store_true")
    parser.add_argument("--known-outcome-disable-live-yes", action="store_true")
    parser.add_argument("--known-outcome-disable-final-yes", action="store_true")
    parser.add_argument("--known-outcome-disable-team-games", action="store_true")
    parser.add_argument("--known-outcome-disable-score-markets", action="store_true")
    parser.add_argument("--known-outcome-disable-weather", action="store_true")
    parser.add_argument("--known-outcome-disable-finance-indexes", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir.resolve()
    result = scan_known_outcomes(
        KnownOutcomeConfig(
            output_dir=output_dir,
            series_ticker=args.known_outcome_series_ticker,
            timezone=args.timezone,
            capture_date=args.capture_date,
            include_previous_date=not args.no_previous_date,
            lookback_days=args.known_outcome_lookback_days,
            contracts=args.contracts,
            min_contracts=args.min_contracts,
            require_full_contracts=args.require_full_contracts,
            max_ask=args.max_ask,
            annual_yield=args.known_outcome_apy,
            fallback_settlement_hours=args.known_outcome_fallback_settlement_hours,
            min_net_profit_per_contract=args.known_outcome_min_net_profit_per_contract,
            orderbook_depth=args.orderbook_depth,
            orderbook_workers=args.known_outcome_orderbook_workers,
            max_markets=args.max_markets,
            max_market_pages=args.max_market_pages,
            include_known_no_after_final=not args.known_outcome_disable_final_no,
            include_known_yes_in_game=not args.known_outcome_disable_live_yes,
            include_known_yes_after_final=not args.known_outcome_disable_final_yes,
            include_team_game_winners=not args.known_outcome_disable_team_games,
            include_score_markets=not args.known_outcome_disable_score_markets,
            include_weather_markets=not args.known_outcome_disable_weather,
            include_finance_index_markets=not args.known_outcome_disable_finance_indexes,
            api_key_id=args.kalshi_api_key_id,
            private_key_path=args.kalshi_private_key_file,
        )
    )
    print(
        f"Known-outcome scan complete: raw_markets={result.raw_markets} "
        f"matched={result.matched_markets} verified={result.verified_markets} "
        f"priced={result.priced_markets} opportunities={result.opportunities} "
        f"trade_ledger={result.trade_count} "
        f"no_liquidity={result.no_liquidity} orderbook_errors={result.orderbook_errors}",
        flush=True,
    )
    print(f"known_outcome_candidates: {result.candidates_csv}", flush=True)
    print(f"known_outcome_opportunities: {result.opportunities_csv}", flush=True)
    print(f"known_outcome_trades: {result.trade_ledger_csv}", flush=True)
    print(f"known_outcome_pnl: {result.pnl_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
