#!/usr/bin/env python3
"""Replay a Topstep challenge strategy on QuantPad MBP-1 bid/ask quotes.

This is a fill-realism check for fast scalps. Signals are computed from
QuantPad 1-second OHLCV bars, while entries, stops, targets, time exits, and
mark-to-market rule checks use executable top-of-book quotes.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, time, timezone
from decimal import Decimal
from pathlib import Path
from statistics import mean
from typing import Any, Optional, Sequence

import quantpad_data as qpd

from topstep_execution_adapter import DEFAULT_CONTRACT_SPECS, TOPSTEP_ROUND_TURN_COSTS
from topstep_quantpad_backtest import (
    DayState,
    ET,
    StrategySpec,
    compact_decimal,
    fetch_quantpad_bars,
    iter_bars_from_frame,
    iter_chunks,
    parse_date_arg,
    round_to_tick,
)
from topstep_rule_simulator import (
    ChallengeStatus,
    RuleEventType,
    TopstepRuleSimulator,
    money,
)


UTC = timezone.utc


@dataclass
class Quote:
    timestamp: datetime
    bid: Decimal
    ask: Decimal
    bid_size: int = 0
    ask_size: int = 0


@dataclass
class ActiveTrade:
    attempt_id: int
    session_date: date
    side: int
    entry_time: datetime
    entry_price: Decimal
    quantity: int
    stop_price: Decimal
    target_price: Decimal
    entry_fee: Decimal
    mae_pnl: Decimal = Decimal("0")
    mfe_pnl: Decimal = Decimal("0")
    mae_points: Decimal = Decimal("0")
    mfe_points: Decimal = Decimal("0")
    min_mll_buffer: Decimal = Decimal("999999999")
    quotes_seen: int = 0


@dataclass
class TradeRecord:
    strategy: str
    symbol: str
    attempt_id: int
    session_date: date
    side: str
    quantity: int
    entry_time: datetime
    exit_time: datetime
    entry_price: Decimal
    exit_price: Decimal
    realized_pnl: Decimal
    mae_pnl: Decimal
    mfe_pnl: Decimal
    mae_points: Decimal
    mfe_points: Decimal
    min_mll_buffer: Decimal
    exit_reason: str
    rule_event: str

    def to_row(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "attempt_id": self.attempt_id,
            "session_date": self.session_date.isoformat(),
            "side": self.side,
            "quantity": self.quantity,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "realized_pnl": str(self.realized_pnl),
            "mae_pnl": str(self.mae_pnl),
            "mfe_pnl": str(self.mfe_pnl),
            "mae_points": str(self.mae_points),
            "mfe_points": str(self.mfe_points),
            "min_mll_buffer": str(self.min_mll_buffer),
            "exit_reason": self.exit_reason,
            "rule_event": self.rule_event,
        }


@dataclass
class AttemptRecord:
    strategy: str
    symbol: str
    attempt_id: int
    start_time: datetime
    end_time: datetime
    outcome: str
    days: int
    trades: int
    net_pnl: Decimal
    reached_mll_lock: bool
    failure_reason: str

    def to_row(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "symbol": self.symbol,
            "attempt_id": self.attempt_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "outcome": self.outcome,
            "days": self.days,
            "trades": self.trades,
            "net_pnl": str(self.net_pnl),
            "reached_mll_lock": int(self.reached_mll_lock),
            "failure_reason": self.failure_reason,
        }


@dataclass
class MBP1Replay:
    spec: StrategySpec
    account_tier: str
    commission_round_turn: Decimal
    slippage_ticks_per_side: Decimal
    quick_pass_days: int = 21
    sim: TopstepRuleSimulator = field(init=False)
    attempt_id: int = 1
    attempt_start: Optional[datetime] = None
    attempt_start_balance: Decimal = Decimal("0")
    last_closed_balance: Decimal = Decimal("0")
    current_day: Optional[DayState] = None
    current_quote: Optional[Quote] = None
    active_trade: Optional[ActiveTrade] = None
    trades: list[TradeRecord] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    bars_seen: int = 0
    quotes_seen: int = 0
    attempt_trades: int = 0
    reached_mll_lock: bool = False
    terminal: bool = False

    def __post_init__(self) -> None:
        self.sim = TopstepRuleSimulator.from_tier(self.account_tier)
        self.attempt_start_balance = self.sim.closed_balance
        self.last_closed_balance = self.sim.closed_balance

    @property
    def spec_info(self):
        return DEFAULT_CONTRACT_SPECS[self.spec.contract_symbol]

    @property
    def multiplier(self) -> Decimal:
        return self.spec_info.multiplier

    @property
    def tick_size(self) -> Decimal:
        return self.spec_info.tick_size

    @property
    def slippage_points(self) -> Decimal:
        return self.tick_size * self.slippage_ticks_per_side

    def process_quote(self, quote: Quote) -> None:
        if quote.bid <= 0 or quote.ask <= 0 or quote.bid > quote.ask:
            return
        if not (self.spec.rth_start <= quote.timestamp.time() <= self.spec.rth_end):
            return
        self.current_quote = quote
        self.quotes_seen += 1
        if self.active_trade is not None and not self.terminal:
            self._manage_trade_on_quote(quote)
        self.reached_mll_lock = self.reached_mll_lock or self.sim.mll_locked

    def process_bar(self, bar: dict[str, Any]) -> None:
        if self.terminal:
            return
        self.bars_seen += 1
        ts: datetime = bar["timestamp"]
        session = ts.date()
        if self.current_day is None or session != self.current_day.session_date:
            self._start_session(session, ts)
        if self.terminal:
            return
        if self.attempt_start is None:
            self.attempt_start = ts

        assert self.current_day is not None
        self._update_day_state(self.current_day, bar)

        if (
            self.active_trade is None
            and self.current_quote is not None
            and self.sim.status == ChallengeStatus.ACTIVE
            and not self.sim.day_locked
            and self.current_day.trades_taken < self.spec.max_trades_per_day
            and self._entry_window(ts)
        ):
            side = self._entry_signal(self.current_day, bar)
            if side is not None:
                self._enter_trade(side, ts)

        self.current_day.last_high = bar["high"]
        self.current_day.last_low = bar["low"]
        self.current_day.last_close = bar["close"]
        self.reached_mll_lock = self.reached_mll_lock or self.sim.mll_locked

    def finish(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        if self.active_trade is not None and self.current_quote is not None:
            self._exit_trade(timestamp, self._market_exit_price(self.active_trade, self.current_quote), "data_end")
        if self.sim.status == ChallengeStatus.ACTIVE and self.sim.session_active:
            try:
                self.sim.end_session()
            except Exception:
                pass
        if self.attempt_start is not None and not self.attempts:
            self._record_attempt(timestamp, "incomplete", "")

    def _start_session(self, session: date, timestamp: datetime) -> None:
        if self.active_trade is not None and self.current_quote is not None:
            self._exit_trade(
                timestamp,
                self._market_exit_price(self.active_trade, self.current_quote),
                "session_end",
            )
        if self.current_day is not None and self.sim.status == ChallengeStatus.ACTIVE:
            if self.sim.session_active:
                try:
                    self.sim.end_session()
                except Exception:
                    pass
            if self.sim.status == ChallengeStatus.ACTIVE:
                try:
                    self.sim.start_next_session()
                except Exception:
                    pass
        self.current_day = DayState(session_date=session)

    def _update_day_state(self, day: DayState, bar: dict[str, Any]) -> None:
        ts: datetime = bar["timestamp"]
        if self.spec.rth_start <= ts.time() < self.spec.opening_range_end_time:
            day.opening_high = (
                bar["high"] if day.opening_high is None else max(day.opening_high, bar["high"])
            )
            day.opening_low = (
                bar["low"] if day.opening_low is None else min(day.opening_low, bar["low"])
            )
        if self.spec.rth_start <= ts.time() <= self.spec.rth_end:
            volume = bar["volume"] if bar["volume"] > 0 else Decimal("1")
            typical = (bar["high"] + bar["low"] + bar["close"]) / Decimal("3")
            day.vwap_numerator += typical * volume
            day.vwap_denominator += volume
            if day.vwap_denominator > 0:
                day.vwap = day.vwap_numerator / day.vwap_denominator

    def _entry_window(self, ts: datetime) -> bool:
        return self.spec.first_entry_time <= ts.time() <= self.spec.last_entry_time

    def _entry_signal(self, day: DayState, bar: dict[str, Any]) -> Optional[int]:
        if self.spec.family != "vwap_reversion":
            raise ValueError("MBP-1 replay currently supports vwap_reversion only")
        if day.vwap is None:
            return None
        if self.spec.max_opening_range_points is not None:
            if day.opening_high is None or day.opening_low is None:
                return None
            if day.opening_high - day.opening_low > self.spec.max_opening_range_points:
                return None
        if self.spec.max_entry_bar_range_points is not None:
            if bar["high"] - bar["low"] > self.spec.max_entry_bar_range_points:
                return None
        distance = bar["close"] - day.vwap
        if distance >= self.spec.threshold_points:
            return -1
        if distance <= -self.spec.threshold_points:
            return 1
        return None

    def _enter_trade(self, side: int, timestamp: datetime) -> None:
        assert self.current_quote is not None
        fill = self._entry_price(side, self.current_quote)
        stop = fill - self.spec.stop_points if side == 1 else fill + self.spec.stop_points
        target = fill + self.spec.target_points if side == 1 else fill - self.spec.target_points
        entry_fee = money((self.commission_round_turn / Decimal("2")) * self.spec.quantity)
        self.active_trade = ActiveTrade(
            attempt_id=self.attempt_id,
            session_date=timestamp.date(),
            side=side,
            entry_time=timestamp,
            entry_price=fill,
            quantity=self.spec.quantity,
            stop_price=stop,
            target_price=target,
            entry_fee=entry_fee,
            min_mll_buffer=money(self.sim.valuation - self.sim.active_mll),
        )
        assert self.current_day is not None
        self.current_day.trades_taken += 1
        self.attempt_trades += 1

    def _manage_trade_on_quote(self, quote: Quote) -> None:
        trade = self.active_trade
        if trade is None:
            return
        trade.quotes_seen += 1
        mark_price = self._mark_price(trade, quote)
        self._mark_trade(trade, mark_price, quote.timestamp)
        if self.active_trade is None:
            return

        held_seconds = int((quote.timestamp - trade.entry_time).total_seconds())
        if held_seconds >= self.spec.max_hold_seconds or quote.timestamp.time() >= self.spec.force_exit_time:
            self._exit_trade(quote.timestamp, self._market_exit_price(trade, quote), "time_exit")
            return

        if trade.side == 1:
            if quote.bid <= trade.stop_price:
                self._exit_trade(quote.timestamp, self._market_exit_price(trade, quote), "stop")
            elif quote.bid >= trade.target_price:
                self._exit_trade(quote.timestamp, trade.target_price, "target")
        else:
            if quote.ask >= trade.stop_price:
                self._exit_trade(quote.timestamp, self._market_exit_price(trade, quote), "stop")
            elif quote.ask <= trade.target_price:
                self._exit_trade(quote.timestamp, trade.target_price, "target")

    def _mark_trade(self, trade: ActiveTrade, mark_price: Decimal, timestamp: datetime) -> None:
        favorable = (mark_price - trade.entry_price) * Decimal(trade.side)
        open_pnl = self._open_pnl(trade, mark_price)
        if favorable < 0:
            trade.mae_points = max(trade.mae_points, -favorable)
            trade.mae_pnl = min(trade.mae_pnl, open_pnl)
        else:
            trade.mfe_points = max(trade.mfe_points, favorable)
            trade.mfe_pnl = max(trade.mfe_pnl, open_pnl)
        trade.min_mll_buffer = min(
            trade.min_mll_buffer,
            money(self.sim.closed_balance + open_pnl - self.sim.active_mll),
        )
        event = self.sim.mark_to_market(open_pnl)
        if event.event_type in {RuleEventType.DLL_BREACH, RuleEventType.MLL_BREACH}:
            self._record_rule_exit(trade, timestamp, mark_price, event.event_type.value)

    def _entry_price(self, side: int, quote: Quote) -> Decimal:
        if side == 1:
            return round_to_tick(quote.ask + self.slippage_points, self.tick_size)
        return round_to_tick(quote.bid - self.slippage_points, self.tick_size)

    def _market_exit_price(self, trade: ActiveTrade, quote: Quote) -> Decimal:
        if trade.side == 1:
            return round_to_tick(quote.bid - self.slippage_points, self.tick_size)
        return round_to_tick(quote.ask + self.slippage_points, self.tick_size)

    def _mark_price(self, trade: ActiveTrade, quote: Quote) -> Decimal:
        return quote.bid if trade.side == 1 else quote.ask

    def _open_pnl(self, trade: ActiveTrade, mark_price: Decimal) -> Decimal:
        gross = (
            (mark_price - trade.entry_price)
            * Decimal(trade.side)
            * self.multiplier
            * Decimal(trade.quantity)
        )
        return money(gross - trade.entry_fee)

    def _realized_pnl(self, trade: ActiveTrade, exit_price: Decimal) -> Decimal:
        gross = (
            (exit_price - trade.entry_price)
            * Decimal(trade.side)
            * self.multiplier
            * Decimal(trade.quantity)
        )
        return money(gross - self.commission_round_turn * Decimal(trade.quantity))

    def _exit_trade(self, timestamp: datetime, exit_price: Decimal, reason: str) -> None:
        trade = self.active_trade
        if trade is None:
            return
        realized = self._realized_pnl(trade, exit_price)
        event = self.sim.close_position(realized)
        self.last_closed_balance = self.sim.closed_balance
        self.trades.append(self._trade_record(trade, timestamp, exit_price, realized, reason, event.event_type.value))
        self.active_trade = None
        if event.event_type == RuleEventType.PASSED:
            self._record_attempt(timestamp, ChallengeStatus.PASSED.value, "")
            self.terminal = True
        elif event.event_type == RuleEventType.MLL_BREACH:
            self._record_attempt(timestamp, ChallengeStatus.FAILED_MLL.value, "mll_breach")
            self.terminal = True

    def _record_rule_exit(
        self,
        trade: ActiveTrade,
        timestamp: datetime,
        price: Decimal,
        event_type: str,
    ) -> None:
        realized = money(self.sim.closed_balance - self.last_closed_balance)
        self.last_closed_balance = self.sim.closed_balance
        reason = "dll_breach" if event_type == RuleEventType.DLL_BREACH.value else "mll_breach"
        self.trades.append(self._trade_record(trade, timestamp, price, realized, reason, event_type))
        self.active_trade = None
        if event_type == RuleEventType.MLL_BREACH.value:
            self._record_attempt(timestamp, ChallengeStatus.FAILED_MLL.value, "unrealized_mll_breach")
            self.terminal = True

    def _trade_record(
        self,
        trade: ActiveTrade,
        timestamp: datetime,
        exit_price: Decimal,
        realized: Decimal,
        reason: str,
        event_type: str,
    ) -> TradeRecord:
        return TradeRecord(
            strategy=self.spec.name,
            symbol=self.spec.contract_symbol,
            attempt_id=trade.attempt_id,
            session_date=trade.session_date,
            side="long" if trade.side == 1 else "short",
            quantity=trade.quantity,
            entry_time=trade.entry_time,
            exit_time=timestamp,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            realized_pnl=realized,
            mae_pnl=trade.mae_pnl,
            mfe_pnl=trade.mfe_pnl,
            mae_points=trade.mae_points,
            mfe_points=trade.mfe_points,
            min_mll_buffer=trade.min_mll_buffer,
            exit_reason=reason,
            rule_event=event_type,
        )

    def _record_attempt(self, timestamp: datetime, outcome: str, failure_reason: str) -> None:
        if self.attempts:
            return
        start = self.attempt_start or timestamp
        self.attempts.append(
            AttemptRecord(
                strategy=self.spec.name,
                symbol=self.spec.contract_symbol,
                attempt_id=self.attempt_id,
                start_time=start,
                end_time=timestamp,
                outcome=outcome,
                days=max(1, self.sim.day_number),
                trades=self.attempt_trades,
                net_pnl=money(self.sim.closed_balance - self.attempt_start_balance),
                reached_mll_lock=self.reached_mll_lock or self.sim.mll_locked,
                failure_reason=failure_reason,
            )
        )

    def summary(self) -> dict[str, Any]:
        completed = [
            attempt
            for attempt in self.attempts
            if attempt.outcome in {ChallengeStatus.PASSED.value, ChallengeStatus.FAILED_MLL.value}
        ]
        passes = [attempt for attempt in completed if attempt.outcome == ChallengeStatus.PASSED.value]
        failures = [attempt for attempt in completed if attempt.outcome == ChallengeStatus.FAILED_MLL.value]
        wins = [trade for trade in self.trades if trade.realized_pnl > 0]
        total_pnl = sum((trade.realized_pnl for trade in self.trades), Decimal("0"))
        return {
            "strategy": self.spec.name,
            "symbol": self.spec.contract_symbol,
            "quantpad_symbol": self.spec.quantpad_symbol,
            "data": "ohlcv-1s signals + mbp-1 executable quotes",
            "quantity": self.spec.quantity,
            "stop_points": str(self.spec.stop_points),
            "target_points": str(self.spec.target_points),
            "threshold_points": str(self.spec.threshold_points),
            "bars_seen": self.bars_seen,
            "quotes_seen": self.quotes_seen,
            "completed_attempts": len(completed),
            "passes": len(passes),
            "failures": len(failures),
            "outcome": self.attempts[0].outcome if self.attempts else "unrecorded",
            "days": self.attempts[0].days if self.attempts else self.sim.day_number,
            "trade_count": len(self.trades),
            "wins": len(wins),
            "losses": len([trade for trade in self.trades if trade.realized_pnl < 0]),
            "win_rate": str(decimal_ratio(len(wins), len(self.trades))),
            "total_trade_pnl": str(money(total_pnl)),
            "ending_balance": str(self.sim.closed_balance),
            "target_balance": str(self.sim.config.target_balance),
            "avg_trade_pnl": str(decimal_mean([trade.realized_pnl for trade in self.trades])),
            "avg_mae_pnl": str(decimal_mean([trade.mae_pnl for trade in self.trades])),
            "avg_mfe_pnl": str(decimal_mean([trade.mfe_pnl for trade in self.trades])),
            "mll_locked": self.sim.mll_locked,
            "active_mll": str(self.sim.active_mll),
            "dll_breach_count": self.sim.dll_breach_count,
            "status": self.sim.status.value,
        }


def decimal_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def decimal_mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return money(0)
    return money(Decimal(str(mean(values))))


def build_es_vwap_spec(args: argparse.Namespace) -> StrategySpec:
    return StrategySpec(
        name=(
            "es_vwap_reversion"
            f"_q{args.quantity}"
            f"_s{compact_decimal(Decimal(args.stop_points))}"
            f"_t{compact_decimal(Decimal(args.target_points))}"
            f"_thr{compact_decimal(Decimal(args.threshold_points))}"
            "_mbp1"
        ),
        quantpad_symbol="ES.FUT",
        contract_symbol="ES",
        family="vwap_reversion",
        quantity=args.quantity,
        stop_points=Decimal(args.stop_points),
        target_points=Decimal(args.target_points),
        threshold_points=Decimal(args.threshold_points),
        max_hold_seconds=args.max_hold_minutes * 60,
        max_trades_per_day=args.max_trades_per_day,
        first_entry_time=time(9, 35),
        last_entry_time=datetime.strptime(args.last_entry_time, "%H:%M").time(),
    )


def iter_quotes_from_frame(frame) -> list[Quote]:
    quotes: list[Quote] = []
    if frame is None or len(frame) == 0:
        return quotes
    for row in frame.itertuples(index=False):
        raw_t = getattr(row, "t")
        timestamp = datetime.fromtimestamp(int(raw_t) / 1_000_000_000, UTC).astimezone(ET)
        bid = Decimal(str(getattr(row, "bid_px")))
        ask = Decimal(str(getattr(row, "ask_px")))
        bid_size = int(getattr(row, "bid_sz", 0) or 0)
        ask_size = int(getattr(row, "ask_sz", 0) or 0)
        quotes.append(Quote(timestamp, bid, ask, bid_size, ask_size))
    return quotes


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_replay(args: argparse.Namespace) -> MBP1Replay:
    if args.api_key_stdin:
        os.environ["QUANTPAD_API_KEY"] = sys.stdin.readline().strip()
    if not os.environ.get("QUANTPAD_API_KEY"):
        raise ValueError("set QUANTPAD_API_KEY or pass --api-key-stdin")

    spec = build_es_vwap_spec(args)
    round_turn = (
        money(args.round_turn_cost)
        if args.round_turn_cost
        else TOPSTEP_ROUND_TURN_COSTS[spec.contract_symbol]
    )
    replay = MBP1Replay(
        spec=spec,
        account_tier=args.account,
        commission_round_turn=round_turn,
        slippage_ticks_per_side=Decimal(args.slippage_ticks_per_side),
        quick_pass_days=args.quick_pass_days,
    )

    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    last_timestamp: Optional[datetime] = None

    for chunk_start, chunk_end in iter_chunks(start, end, args.chunk_days):
        print(
            json.dumps(
                {
                    "event": "fetch",
                    "symbol": spec.quantpad_symbol,
                    "start": chunk_start.isoformat(),
                    "end": chunk_end.isoformat(),
                }
            ),
            flush=True,
        )
        bars = list(iter_bars_from_frame(fetch_quantpad_bars(spec.quantpad_symbol, chunk_start, chunk_end, "1s")))
        bar_index = 0
        start_ms = int(chunk_start.timestamp() * 1000)
        end_ms = int(chunk_end.timestamp() * 1000)
        for frame in qpd.get_mbp1(
            spec.quantpad_symbol,
            start_ms,
            end_ms,
            columns=["t", "bid_px", "ask_px", "bid_sz", "ask_sz"],
            chunk_rows=args.quote_chunk_rows,
        ):
            for quote in iter_quotes_from_frame(frame):
                last_timestamp = quote.timestamp
                replay.process_quote(quote)
                while bar_index < len(bars) and bars[bar_index]["timestamp"] <= quote.timestamp:
                    replay.process_bar(bars[bar_index])
                    bar_index += 1
                if replay.terminal:
                    break
            if replay.terminal:
                break
        if replay.terminal:
            break

    replay.finish(last_timestamp)
    return replay


def write_outputs(args: argparse.Namespace, replay: MBP1Replay) -> None:
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "summary": replay.summary(),
        "metadata": {
            "start": args.start,
            "end": args.end,
            "account": args.account,
            "slippage_ticks_per_side": args.slippage_ticks_per_side,
            "round_turn_cost": args.round_turn_cost,
            "quote_chunk_rows": args.quote_chunk_rows,
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_csv(out / "summary.csv", [summary["summary"]])
    write_csv(out / "trades.csv", [trade.to_row() for trade in replay.trades])
    write_csv(out / "attempts.csv", [attempt.to_row() for attempt in replay.attempts])


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--output-dir", default="topstep_quantpad_mbp1_replay")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--chunk-days", type=int, default=1)
    parser.add_argument("--quote-chunk-rows", type=int, default=250_000)
    parser.add_argument("--account", default="150K")
    parser.add_argument("--quantity", type=int, default=6)
    parser.add_argument("--stop-points", default="8")
    parser.add_argument("--target-points", default="3")
    parser.add_argument("--threshold-points", default="3")
    parser.add_argument("--max-trades-per-day", type=int, default=3)
    parser.add_argument("--max-hold-minutes", type=int, default=10)
    parser.add_argument("--last-entry-time", default="11:30")
    parser.add_argument("--slippage-ticks-per-side", default="0.5")
    parser.add_argument("--round-turn-cost", default="3.80")
    parser.add_argument("--quick-pass-days", type=int, default=21)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    replay = run_replay(args)
    write_outputs(args, replay)
    print(json.dumps({"output_dir": args.output_dir, "summary": replay.summary()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
