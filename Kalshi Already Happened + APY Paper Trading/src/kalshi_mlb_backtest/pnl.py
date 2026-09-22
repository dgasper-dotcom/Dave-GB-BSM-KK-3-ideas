from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from typing import Any


THRESHOLDS = [0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92]


@dataclass(frozen=True)
class TradeResult:
    threshold: float
    mode: str
    theoretical: bool
    trades: int
    contracts_purchased: float
    total_capital_deployed: float | None
    gross_profit: float | None
    gross_return: float | None
    win_rate: float | None
    loss_rate: float | None
    average_profit_per_trade: float | None
    maximum_drawdown: float | None
    largest_losing_trade: float | None
    note: str


def kalshi_taker_fee(price: float, contracts: float, fee_multiplier: float | None = 1.0, fee_type: str | None = "quadratic") -> float:
    if contracts <= 0:
        return 0.0
    if fee_type not in {"quadratic", "quadratic_with_maker_fees"}:
        return 0.0
    multiplier = 1.0 if fee_multiplier is None else float(fee_multiplier)
    if multiplier <= 0:
        return 0.0
    price_cents = price * 100.0
    fee_cents = 0.07 * (price_cents * (100.0 - price_cents) / 100.0) * contracts * multiplier
    return math.ceil(fee_cents - 1e-12) / 100.0


def _drawdown(profits: list[float]) -> float:
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for profit in profits:
        equity += profit
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return max_dd


def _summarize_trades(threshold: float, mode: str, theoretical: bool, trade_rows: list[dict[str, Any]], note: str) -> TradeResult:
    if not trade_rows:
        return TradeResult(
            threshold=threshold,
            mode=mode,
            theoretical=theoretical,
            trades=0,
            contracts_purchased=0.0,
            total_capital_deployed=0.0 if not theoretical else None,
            gross_profit=0.0 if not theoretical else None,
            gross_return=None,
            win_rate=None,
            loss_rate=None,
            average_profit_per_trade=None,
            maximum_drawdown=None,
            largest_losing_trade=None,
            note=note,
        )
    profits = [float(row["profit"]) for row in trade_rows]
    deployed = [float(row["capital_deployed"]) for row in trade_rows]
    wins = sum(1 for row in trade_rows if row["no_won"])
    total_profit = sum(profits)
    total_deployed = sum(deployed)
    return TradeResult(
        threshold=threshold,
        mode=mode,
        theoretical=theoretical,
        trades=len(trade_rows),
        contracts_purchased=sum(float(row["contracts"]) for row in trade_rows),
        total_capital_deployed=total_deployed,
        gross_profit=total_profit,
        gross_return=(total_profit / total_deployed) if total_deployed else None,
        win_rate=wins / len(trade_rows),
        loss_rate=1.0 - wins / len(trade_rows),
        average_profit_per_trade=total_profit / len(trade_rows),
        maximum_drawdown=_drawdown(profits),
        largest_losing_trade=min(profits),
        note=note,
    )


def _first_threshold_snapshots(conn: sqlite3.Connection, threshold: float) -> list[sqlite3.Row]:
    return conn.execute(
        """
        WITH eligible AS (
            SELECT
                o.opportunity_id,
                o.no_won,
                s.snapshot_id,
                s.timestamp,
                s.best_no_ask,
                s.best_no_ask_size,
                s.exact_orderbook_available,
                m.fee_type,
                m.fee_multiplier,
                ROW_NUMBER() OVER (PARTITION BY o.opportunity_id ORDER BY s.timestamp, s.snapshot_id) AS rn
            FROM opportunities o
            JOIN orderbook_snapshots s ON s.market_id = o.market_id
            JOIN markets m ON m.market_id = o.market_id
            WHERE s.best_no_ask IS NOT NULL
              AND s.best_no_ask <= ?
              AND o.no_won IS NOT NULL
              AND COALESCE(s.source, '') <> 'kalshi_live_orderbook'
        )
        SELECT * FROM eligible WHERE rn = 1 ORDER BY timestamp, opportunity_id
        """,
        (threshold,),
    ).fetchall()


def _levels_available_at_threshold(conn: sqlite3.Connection, snapshot_id: int, threshold: float) -> float | None:
    rows = conn.execute(
        """
        SELECT SUM(quantity) AS quantity
        FROM orderbook_levels
        WHERE snapshot_id = ? AND side = 'no_ask' AND price <= ?
        """,
        (snapshot_id, threshold),
    ).fetchone()
    if rows is None or rows["quantity"] is None:
        return None
    return float(rows["quantity"])


def run_pnl(conn: sqlite3.Connection, thresholds: list[float] | None = None) -> list[TradeResult]:
    thresholds = thresholds or THRESHOLDS
    results: list[TradeResult] = []
    for threshold in thresholds:
        snapshots = _first_threshold_snapshots(conn, threshold)
        one_contract_rows: list[dict[str, Any]] = []
        max_available_rows: list[dict[str, Any]] = []
        any_exact_depth = False
        for row in snapshots:
            price = float(row["best_no_ask"])
            no_won = bool(row["no_won"])
            fee_type = row["fee_type"]
            fee_multiplier = row["fee_multiplier"]
            contracts = 1.0
            fee = kalshi_taker_fee(price, contracts, fee_multiplier, fee_type)
            cost = price * contracts
            payout = contracts if no_won else 0.0
            one_contract_rows.append(
                {
                    "contracts": contracts,
                    "capital_deployed": cost + fee,
                    "profit": payout - cost - fee,
                    "no_won": no_won,
                }
            )
            quantity = _levels_available_at_threshold(conn, int(row["snapshot_id"]), threshold)
            if quantity is not None and quantity > 0:
                any_exact_depth = True
                fee = kalshi_taker_fee(price, quantity, fee_multiplier, fee_type)
                cost = price * quantity
                payout = quantity if no_won else 0.0
                max_available_rows.append(
                    {
                        "contracts": quantity,
                        "capital_deployed": cost + fee,
                        "profit": payout - cost - fee,
                        "no_won": no_won,
                    }
                )
        results.append(
            _summarize_trades(
                threshold,
                "one_contract",
                theoretical=not any(bool(row["exact_orderbook_available"]) for row in snapshots),
                trade_rows=one_contract_rows,
                note=(
                    "One trade per unique opportunity at first threshold hit. "
                    "Theoretical if sourced from candlesticks because historical size/depth is unavailable."
                ),
            )
        )
        results.append(
            _summarize_trades(
                threshold,
                "max_available_depth",
                theoretical=not any_exact_depth,
                trade_rows=max_available_rows,
                note=(
                    "Uses exact no_ask orderbook levels at or below threshold when present. "
                    "Unavailable for Kalshi historical candlestick-only rows."
                ),
            )
        )
    return results


def trade_result_rows(results: list[TradeResult]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "threshold": result.threshold,
                "mode": result.mode,
                "theoretical": int(result.theoretical),
                "trades": result.trades,
                "contracts_purchased": result.contracts_purchased,
                "total_capital_deployed": result.total_capital_deployed,
                "gross_profit": result.gross_profit,
                "gross_return": result.gross_return,
                "win_rate": result.win_rate,
                "loss_rate": result.loss_rate,
                "average_profit_per_trade": result.average_profit_per_trade,
                "maximum_drawdown": result.maximum_drawdown,
                "largest_losing_trade": result.largest_losing_trade,
                "note": result.note,
            }
        )
    return rows
