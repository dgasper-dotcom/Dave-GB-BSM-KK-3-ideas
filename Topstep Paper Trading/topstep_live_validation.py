"""Summarize live paper trades for validation checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

from topstep_rule_simulator import money


@dataclass(frozen=True)
class LiveTrade:
    symbol: str
    session_date: str
    realized_pnl: Decimal
    exit_reason: str
    rule_event: str
    mae_pnl: Decimal
    mfe_pnl: Decimal


def load_live_trades(output_dir: str | Path) -> list[LiveTrade]:
    root = Path(output_dir)
    paths = sorted(root.glob("*/trades.csv"))
    if (root / "trades.csv").exists():
        paths.append(root / "trades.csv")
    trades: list[LiveTrade] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                trades.append(
                    LiveTrade(
                        symbol=str(row.get("symbol") or path.parent.name).upper(),
                        session_date=str(row["session_date"]),
                        realized_pnl=money(row["realized_pnl"]),
                        exit_reason=str(row.get("exit_reason", "")),
                        rule_event=str(row.get("rule_event", "")),
                        mae_pnl=money(row.get("mae_pnl") or 0),
                        mfe_pnl=money(row.get("mfe_pnl") or 0),
                    )
                )
    return trades


def summarize_live_trades(trades: Sequence[LiveTrade]) -> dict[str, Any]:
    by_symbol: dict[str, list[LiveTrade]] = {}
    for trade in trades:
        by_symbol.setdefault(trade.symbol, []).append(trade)
    return {
        "total": summarize_group(trades),
        "symbols": {
            symbol: summarize_group(rows)
            for symbol, rows in sorted(by_symbol.items())
        },
    }


def summarize_group(trades: Sequence[LiveTrade]) -> dict[str, Any]:
    if not trades:
        return {
            "trades": 0,
            "active_days": 0,
            "win_rate": None,
            "realized_pnl": "0.00",
            "avg_trade_pnl": None,
            "avg_mae_pnl": None,
            "avg_mfe_pnl": None,
            "exit_reasons": {},
            "rule_events": {},
        }
    wins = [trade for trade in trades if trade.realized_pnl > 0]
    total = sum((trade.realized_pnl for trade in trades), Decimal("0"))
    return {
        "trades": len(trades),
        "active_days": len({trade.session_date for trade in trades}),
        "win_rate": str(rate(len(wins), len(trades))),
        "realized_pnl": str(money(total)),
        "avg_trade_pnl": str(money(total / Decimal(len(trades)))),
        "avg_mae_pnl": str(decimal_mean([trade.mae_pnl for trade in trades])),
        "avg_mfe_pnl": str(decimal_mean([trade.mfe_pnl for trade in trades])),
        "exit_reasons": dict(Counter(trade.exit_reason for trade in trades)),
        "rule_events": dict(Counter(trade.rule_event for trade in trades)),
    }


def decimal_mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return money(0)
    return money(Decimal(str(mean(values))))


def rate(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="topstep_prop_challenge_paper")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    summary = summarize_live_trades(load_live_trades(args.output_dir))
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        total = summary["total"]
        print(
            "trades={trades} active_days={active_days} win_rate={win_rate} "
            "realized_pnl={realized_pnl} avg_trade={avg_trade_pnl}".format(**total)
        )
        for symbol, row in summary["symbols"].items():
            print(
                "{symbol}: trades={trades} active_days={active_days} "
                "win_rate={win_rate} realized_pnl={realized_pnl} "
                "avg_mae={avg_mae_pnl} avg_mfe={avg_mfe_pnl}".format(
                    symbol=symbol,
                    **row,
                )
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
