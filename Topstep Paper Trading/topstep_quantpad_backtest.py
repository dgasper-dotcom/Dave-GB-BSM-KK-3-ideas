"""Backtest Topstep strategies using QuantPad historical futures data.

The script pulls bounded 1-second OHLCV windows from QuantPad, runs strategy
fills through the Topstep challenge rule simulator, then bootstraps the same
trade-day distribution through the XFA payout simulator.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Iterator, Optional, Sequence
from zoneinfo import ZoneInfo

from topstep_execution_adapter import (
    DEFAULT_CONTRACT_SPECS,
    TOPSTEP_ROUND_TURN_COSTS,
)
from topstep_rule_simulator import (
    ChallengeStatus,
    RuleEventType,
    TopstepRuleSimulator,
    get_account_config,
    money,
)
from topstep_xfa_simulator import (
    TradePath,
    TradingDay,
    XFAAccountConfig,
    XFAPayoutPath,
    XFAPayoutPolicy,
    XFARiskPolicy,
    run_xfa_bootstrap,
    summarize_xfa_trials,
)


ET = ZoneInfo("America/New_York")
UTC = timezone.utc


DEFAULT_LIBRARY_NAMES = [
    "nq_vwap_scalp_20_12",
    "nq_or_fade_15_40",
    "es_vwap_scalp_8_5",
    "es_failed_continuation_8_5",
    "nq_breakout_pullback_15_60",
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    quantpad_symbol: str
    contract_symbol: str
    family: str
    quantity: int
    stop_points: Decimal
    target_points: Decimal
    threshold_points: Decimal = Decimal("0")
    opening_range_minutes: int = 5
    warmup_minutes: int = 5
    confirm_points: Decimal = Decimal("0")
    max_opening_range_points: Optional[Decimal] = None
    max_entry_bar_range_points: Optional[Decimal] = None
    max_hold_seconds: int = 600
    max_trades_per_day: int = 3
    first_entry_time: time = time(9, 35)
    last_entry_time: time = time(11, 30)
    force_exit_time: time = time(16, 0)
    rth_start: time = time(9, 30)
    rth_end: time = time(16, 0)
    opening_range_end_time: time = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "contract_symbol", self.contract_symbol.upper())
        object.__setattr__(self, "stop_points", Decimal(str(self.stop_points)))
        object.__setattr__(self, "target_points", Decimal(str(self.target_points)))
        object.__setattr__(self, "threshold_points", Decimal(str(self.threshold_points)))
        object.__setattr__(self, "confirm_points", Decimal(str(self.confirm_points)))
        if self.max_opening_range_points is not None:
            object.__setattr__(
                self,
                "max_opening_range_points",
                Decimal(str(self.max_opening_range_points)),
            )
        if self.max_entry_bar_range_points is not None:
            object.__setattr__(
                self,
                "max_entry_bar_range_points",
                Decimal(str(self.max_entry_bar_range_points)),
            )
        if self.contract_symbol not in DEFAULT_CONTRACT_SPECS:
            raise ValueError(f"unsupported contract symbol: {self.contract_symbol}")
        start_dt = datetime.combine(date(2000, 1, 1), self.rth_start)
        object.__setattr__(
            self,
            "opening_range_end_time",
            (start_dt + timedelta(minutes=self.opening_range_minutes)).time(),
        )
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.stop_points <= 0 or self.target_points <= 0:
            raise ValueError("stop_points and target_points must be positive")
        if (
            self.max_opening_range_points is not None
            and self.max_opening_range_points <= 0
        ):
            raise ValueError("max_opening_range_points must be positive")
        if (
            self.max_entry_bar_range_points is not None
            and self.max_entry_bar_range_points <= 0
        ):
            raise ValueError("max_entry_bar_range_points must be positive")
        if self.family not in {
            "vwap_reversion",
            "vwap_failed_continuation",
            "opening_range_fade",
            "opening_range_breakout_pullback",
        }:
            raise ValueError(f"unsupported family: {self.family}")


@dataclass
class DayState:
    session_date: date
    trades_taken: int = 0
    opening_high: Optional[Decimal] = None
    opening_low: Optional[Decimal] = None
    vwap_numerator: Decimal = Decimal("0")
    vwap_denominator: Decimal = Decimal("0")
    vwap: Optional[Decimal] = None
    pending_side: Optional[int] = None
    pending_extreme: Optional[Decimal] = None
    pending_started_at: Optional[datetime] = None
    last_high: Optional[Decimal] = None
    last_low: Optional[Decimal] = None
    last_close: Optional[Decimal] = None


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
    bars_held: int = 0


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
class StrategyBacktest:
    spec: StrategySpec
    account_tier: str
    commission_round_turn: Decimal
    slippage_ticks_per_side: Decimal
    quick_pass_days: int = 20
    sim: TopstepRuleSimulator = field(init=False)
    attempt_id: int = 1
    attempt_start: Optional[datetime] = None
    attempt_start_balance: Decimal = Decimal("0")
    last_closed_balance: Decimal = Decimal("0")
    attempt_trades: int = 0
    reached_mll_lock: bool = False
    current_day: Optional[DayState] = None
    active_trade: Optional[ActiveTrade] = None
    trades: list[TradeRecord] = field(default_factory=list)
    attempts: list[AttemptRecord] = field(default_factory=list)
    waiting_for_next_session: bool = False
    rows_seen: int = 0

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

    def process_bar(self, bar: dict[str, Any]) -> None:
        self.rows_seen += 1
        ts: datetime = bar["timestamp"]
        session = ts.date()
        if self.current_day is None or session != self.current_day.session_date:
            self._start_session(session, ts)

        if self.waiting_for_next_session:
            return

        if self.attempt_start is None:
            self.attempt_start = ts

        assert self.current_day is not None
        self._update_day_state(self.current_day, bar)

        if self.active_trade is not None:
            self._manage_trade(bar)

        if (
            self.active_trade is None
            and self.sim.status == ChallengeStatus.ACTIVE
            and not self.sim.day_locked
            and self.current_day.trades_taken < self.spec.max_trades_per_day
            and self._entry_window(ts)
        ):
            side = self._entry_signal(self.current_day, bar)
            if side is not None:
                self._enter_trade(side, bar)

        self.current_day.last_high = bar["high"]
        self.current_day.last_low = bar["low"]
        self.current_day.last_close = bar["close"]
        self.reached_mll_lock = self.reached_mll_lock or self.sim.mll_locked

    def finish(self, timestamp: Optional[datetime]) -> None:
        if timestamp is None:
            return
        if self.active_trade is not None:
            self._exit_trade(timestamp, self.active_trade.entry_price, "data_end")
        if self.sim.status == ChallengeStatus.ACTIVE and self.sim.session_active:
            try:
                self.sim.end_session()
            except Exception:
                pass
        if self.attempt_start is not None and not self._attempt_recorded(self.attempt_id):
            self._record_attempt(timestamp, "incomplete", "")

    def _start_session(self, session: date, timestamp: datetime) -> None:
        if self.current_day is not None and self.active_trade is not None:
            self._exit_trade(timestamp, self.active_trade.entry_price, "session_end")

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

        if self.waiting_for_next_session:
            self._new_attempt(timestamp)

        self.current_day = DayState(session_date=session)

    def _new_attempt(self, timestamp: datetime) -> None:
        self.attempt_id += 1
        self.sim = TopstepRuleSimulator.from_tier(self.account_tier)
        self.attempt_start = timestamp
        self.attempt_start_balance = self.sim.closed_balance
        self.last_closed_balance = self.sim.closed_balance
        self.attempt_trades = 0
        self.reached_mll_lock = False
        self.waiting_for_next_session = False
        self.active_trade = None

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
        if not self._volatility_filters_pass(day, bar):
            return None

        if self.spec.family == "vwap_reversion":
            if day.vwap is None:
                return None
            distance = bar["close"] - day.vwap
            if distance >= self.spec.threshold_points:
                return -1
            if distance <= -self.spec.threshold_points:
                return 1
            return None

        if self.spec.family == "vwap_failed_continuation":
            return self._failed_continuation_signal(day, bar)

        if self.spec.family == "opening_range_fade":
            if day.opening_high is None or day.opening_low is None:
                return None
            if bar["close"] >= day.opening_high + self.spec.threshold_points:
                return -1
            if bar["close"] <= day.opening_low - self.spec.threshold_points:
                return 1
            return None

        if self.spec.family == "opening_range_breakout_pullback":
            if day.opening_high is None or day.opening_low is None or day.vwap is None:
                return None
            if (
                day.last_high is not None
                and day.last_low is not None
                and bar["close"] > day.last_high
                and day.last_low >= day.opening_high
                and bar["close"] > day.vwap
            ):
                return 1
            if (
                day.last_high is not None
                and day.last_low is not None
                and bar["close"] < day.last_low
                and day.last_high <= day.opening_low
                and bar["close"] < day.vwap
            ):
                return -1
            return None

        return None

    def _volatility_filters_pass(self, day: DayState, bar: dict[str, Any]) -> bool:
        if self.spec.max_opening_range_points is not None:
            if day.opening_high is None or day.opening_low is None:
                return False
            if day.opening_high - day.opening_low > self.spec.max_opening_range_points:
                return False
        if self.spec.max_entry_bar_range_points is not None:
            if bar["high"] - bar["low"] > self.spec.max_entry_bar_range_points:
                return False
        return True

    def _failed_continuation_signal(self, day: DayState, bar: dict[str, Any]) -> Optional[int]:
        if day.vwap is None:
            return None
        distance = bar["close"] - day.vwap
        if distance >= self.spec.threshold_points:
            if day.pending_side != -1:
                day.pending_side = -1
                day.pending_extreme = bar["high"]
                day.pending_started_at = bar["timestamp"]
            else:
                day.pending_extreme = max(day.pending_extreme or bar["high"], bar["high"])
        elif distance <= -self.spec.threshold_points:
            if day.pending_side != 1:
                day.pending_side = 1
                day.pending_extreme = bar["low"]
                day.pending_started_at = bar["timestamp"]
            else:
                day.pending_extreme = min(day.pending_extreme or bar["low"], bar["low"])

        if day.pending_side is None:
            return None
        if day.pending_started_at is not None:
            age = (bar["timestamp"] - day.pending_started_at).total_seconds()
            if age > self.spec.max_hold_seconds:
                day.pending_side = None
                day.pending_extreme = None
                day.pending_started_at = None
                return None

        if (
            day.pending_side == -1
            and day.last_low is not None
            and day.pending_extreme is not None
            and bar["close"] < day.last_low
            and bar["close"] <= day.pending_extreme - self.spec.confirm_points
        ):
            day.pending_side = None
            return -1
        if (
            day.pending_side == 1
            and day.last_high is not None
            and day.pending_extreme is not None
            and bar["close"] > day.last_high
            and bar["close"] >= day.pending_extreme + self.spec.confirm_points
        ):
            day.pending_side = None
            return 1
        return None

    def _enter_trade(self, side: int, bar: dict[str, Any]) -> None:
        fill = self._apply_entry_slippage(bar["close"], side)
        stop = fill - self.spec.stop_points if side == 1 else fill + self.spec.stop_points
        target = fill + self.spec.target_points if side == 1 else fill - self.spec.target_points
        entry_fee = money((self.commission_round_turn / Decimal("2")) * self.spec.quantity)
        self.active_trade = ActiveTrade(
            attempt_id=self.attempt_id,
            session_date=bar["timestamp"].date(),
            side=side,
            entry_time=bar["timestamp"],
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

    def _manage_trade(self, bar: dict[str, Any]) -> None:
        assert self.active_trade is not None
        trade = self.active_trade
        ts: datetime = bar["timestamp"]
        held_seconds = int((ts - trade.entry_time).total_seconds())
        if held_seconds >= self.spec.max_hold_seconds or ts.time() >= self.spec.force_exit_time:
            self._mark_trade_path(trade, self._adverse_first_path(trade, bar), ts)
            if self.active_trade is not None:
                self._exit_trade(ts, self._apply_exit_slippage(bar["close"], trade.side), "time_exit")
            return

        exit_price = None
        exit_reason = None
        if trade.side == 1:
            if bar["low"] <= trade.stop_price:
                exit_price = trade.stop_price
                exit_reason = "stop"
            elif bar["high"] >= trade.target_price:
                exit_price = trade.target_price
                exit_reason = "target"
        else:
            if bar["high"] >= trade.stop_price:
                exit_price = trade.stop_price
                exit_reason = "stop"
            elif bar["low"] <= trade.target_price:
                exit_price = trade.target_price
                exit_reason = "target"

        if exit_price is not None and exit_reason is not None:
            self._mark_trade_path(
                trade,
                self._path_until_exit(trade, bar, exit_price, exit_reason),
                ts,
            )
            if self.active_trade is not None:
                self._exit_trade(ts, exit_price, exit_reason)
            return

        self._mark_trade_path(trade, self._adverse_first_path(trade, bar), ts)

    def _adverse_first_path(self, trade: ActiveTrade, bar: dict[str, Any]) -> tuple[Decimal, ...]:
        if trade.side == 1:
            return (bar["open"], bar["low"], bar["high"], bar["close"])
        return (bar["open"], bar["high"], bar["low"], bar["close"])

    def _path_until_exit(
        self,
        trade: ActiveTrade,
        bar: dict[str, Any],
        exit_price: Decimal,
        exit_reason: str,
    ) -> tuple[Decimal, ...]:
        if exit_reason == "stop":
            return (bar["open"], exit_price)
        path = self._adverse_first_path(trade, bar)
        return (path[0], path[1], exit_price)

    def _mark_trade_path(
        self,
        trade: ActiveTrade,
        prices: Iterable[Decimal],
        timestamp: datetime,
    ) -> None:
        for px in prices:
            open_pnl = self._open_pnl(trade, px)
            favorable = (px - trade.entry_price) * Decimal(trade.side)
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
                self._record_rule_exit(trade, timestamp, px, event.event_type.value)
                return
        trade.bars_held += 1

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
        self.trades.append(
            TradeRecord(
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
                exit_reason=reason,
                rule_event=event.event_type.value,
            )
        )
        self.active_trade = None
        if event.event_type == RuleEventType.PASSED:
            self._record_attempt(timestamp, ChallengeStatus.PASSED.value, "")
            self.waiting_for_next_session = True
        elif event.event_type == RuleEventType.MLL_BREACH:
            self._record_attempt(timestamp, ChallengeStatus.FAILED_MLL.value, "mll_breach")
            self.waiting_for_next_session = True

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
        self.trades.append(
            TradeRecord(
                strategy=self.spec.name,
                symbol=self.spec.contract_symbol,
                attempt_id=trade.attempt_id,
                session_date=trade.session_date,
                side="long" if trade.side == 1 else "short",
                quantity=trade.quantity,
                entry_time=trade.entry_time,
                exit_time=timestamp,
                entry_price=trade.entry_price,
                exit_price=price,
                realized_pnl=realized,
                mae_pnl=trade.mae_pnl,
                mfe_pnl=trade.mfe_pnl,
                mae_points=trade.mae_points,
                mfe_points=trade.mfe_points,
                exit_reason=reason,
                rule_event=event_type,
            )
        )
        self.active_trade = None
        if event_type == RuleEventType.MLL_BREACH.value:
            self._record_attempt(trade.entry_time, ChallengeStatus.FAILED_MLL.value, "unrealized_mll_breach")
            self.waiting_for_next_session = True

    def _record_attempt(self, timestamp: datetime, outcome: str, failure_reason: str) -> None:
        if self._attempt_recorded(self.attempt_id):
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

    def _attempt_recorded(self, attempt_id: int) -> bool:
        return any(attempt.attempt_id == attempt_id for attempt in self.attempts)

    def _apply_entry_slippage(self, price: Decimal, side: int) -> Decimal:
        return round_to_tick(price + self.slippage_points * Decimal(side), self.tick_size)

    def _apply_exit_slippage(self, price: Decimal, side: int) -> Decimal:
        return round_to_tick(price - self.slippage_points * Decimal(side), self.tick_size)

    def summary(self, xfa_summary: Optional[dict[str, Any]]) -> dict[str, Any]:
        completed = [
            attempt
            for attempt in self.attempts
            if attempt.outcome in {ChallengeStatus.PASSED.value, ChallengeStatus.FAILED_MLL.value}
        ]
        passes = [attempt for attempt in completed if attempt.outcome == ChallengeStatus.PASSED.value]
        failures = [attempt for attempt in completed if attempt.outcome == ChallengeStatus.FAILED_MLL.value]
        quick_passes = [
            attempt for attempt in passes if attempt.days <= self.quick_pass_days
        ]
        wins = [trade for trade in self.trades if trade.realized_pnl > 0]
        total_trade_pnl = sum((trade.realized_pnl for trade in self.trades), Decimal("0"))
        pass_rate = decimal_ratio(len(passes), len(completed))
        avg_xfa_payout = money(xfa_summary["avg_trader_payout"]) if xfa_summary else money(0)
        challenge_cost = money("85")
        ev_per_challenge_attempt = money(pass_rate * avg_xfa_payout - challenge_cost)
        return {
            "strategy": self.spec.name,
            "symbol": self.spec.contract_symbol,
            "quantpad_symbol": self.spec.quantpad_symbol,
            "family": self.spec.family,
            "quantity": self.spec.quantity,
            "stop_points": str(self.spec.stop_points),
            "target_points": str(self.spec.target_points),
            "reward_risk_ratio": str((self.spec.target_points / self.spec.stop_points).quantize(Decimal("0.0001"))),
            "threshold_points": str(self.spec.threshold_points),
            "max_opening_range_points": (
                str(self.spec.max_opening_range_points)
                if self.spec.max_opening_range_points is not None
                else ""
            ),
            "max_entry_bar_range_points": (
                str(self.spec.max_entry_bar_range_points)
                if self.spec.max_entry_bar_range_points is not None
                else ""
            ),
            "rows_seen": self.rows_seen,
            "completed_attempts": len(completed),
            "passes": len(passes),
            "quick_passes": len(quick_passes),
            "failures": len(failures),
            "pass_rate": str(pass_rate),
            "quick_pass_days": self.quick_pass_days,
            "quick_pass_rate": str(decimal_ratio(len(quick_passes), len(completed))),
            "trade_count": len(self.trades),
            "win_rate": str(decimal_ratio(len(wins), len(self.trades))),
            "avg_trade_pnl": str(decimal_mean([trade.realized_pnl for trade in self.trades])),
            "total_trade_pnl": str(money(total_trade_pnl)),
            "avg_mae_pnl": str(decimal_mean([trade.mae_pnl for trade in self.trades])),
            "avg_mfe_pnl": str(decimal_mean([trade.mfe_pnl for trade in self.trades])),
            "avg_days_to_pass": str(decimal_mean([Decimal(a.days) for a in passes])),
            "avg_days_to_fail": str(decimal_mean([Decimal(a.days) for a in failures])),
            "mll_lock_reached": sum(1 for attempt in self.attempts if attempt.reached_mll_lock),
            "failure_reasons": failure_counts(failures),
            "xfa": xfa_summary or {},
            "payout_rate": xfa_summary.get("probability_of_any_payout", "0.0000") if xfa_summary else "0.0000",
            "avg_payout": xfa_summary.get("avg_trader_payout", "0.00") if xfa_summary else "0.00",
            "ev_per_challenge_attempt": str(ev_per_challenge_attempt),
        }


def default_strategies(names: Sequence[str]) -> list[StrategySpec]:
    library = {
        "nq_vwap_scalp_20_12": StrategySpec(
            name="nq_vwap_scalp_20_12",
            quantpad_symbol="NQ.FUT",
            contract_symbol="NQ",
            family="vwap_reversion",
            quantity=1,
            stop_points=Decimal("20"),
            target_points=Decimal("12"),
            threshold_points=Decimal("10"),
            max_hold_seconds=600,
            max_trades_per_day=3,
        ),
        "nq_or_fade_15_40": StrategySpec(
            name="nq_or_fade_15_40",
            quantpad_symbol="NQ.FUT",
            contract_symbol="NQ",
            family="opening_range_fade",
            quantity=1,
            stop_points=Decimal("15"),
            target_points=Decimal("40"),
            threshold_points=Decimal("10"),
            opening_range_minutes=5,
            max_hold_seconds=10_800,
            max_trades_per_day=1,
            last_entry_time=time(15, 0),
        ),
        "es_vwap_scalp_8_5": StrategySpec(
            name="es_vwap_scalp_8_5",
            quantpad_symbol="ES.FUT",
            contract_symbol="ES",
            family="vwap_reversion",
            quantity=1,
            stop_points=Decimal("8"),
            target_points=Decimal("5"),
            threshold_points=Decimal("4"),
            max_hold_seconds=600,
            max_trades_per_day=3,
        ),
        "es_failed_continuation_8_5": StrategySpec(
            name="es_failed_continuation_8_5",
            quantpad_symbol="ES.FUT",
            contract_symbol="ES",
            family="vwap_failed_continuation",
            quantity=1,
            stop_points=Decimal("8"),
            target_points=Decimal("5"),
            threshold_points=Decimal("4"),
            confirm_points=Decimal("1"),
            max_hold_seconds=600,
            max_trades_per_day=3,
        ),
        "nq_breakout_pullback_15_60": StrategySpec(
            name="nq_breakout_pullback_15_60",
            quantpad_symbol="NQ.FUT",
            contract_symbol="NQ",
            family="opening_range_breakout_pullback",
            quantity=1,
            stop_points=Decimal("15"),
            target_points=Decimal("60"),
            opening_range_minutes=15,
            max_hold_seconds=10_800,
            max_trades_per_day=1,
            last_entry_time=time(11, 30),
        ),
    }
    if not names:
        names = DEFAULT_LIBRARY_NAMES
    strategies = []
    for name in names:
        if name not in library:
            raise ValueError(f"unknown strategy {name!r}; choices={sorted(library)}")
        strategies.append(library[name])
    return strategies


def build_strategies(args: argparse.Namespace) -> list[StrategySpec]:
    names = parse_csv_list(args.strategies)
    if args.sweep:
        if names:
            raise ValueError("--strategies and --sweep are mutually exclusive")
        return sweep_strategies(args)
    return default_strategies(names)


def sweep_strategies(args: argparse.Namespace) -> list[StrategySpec]:
    specs: list[StrategySpec] = []
    families = parse_csv_list(args.sweep_families)
    symbols = parse_csv_list(args.sweep_symbols)
    quantities = parse_int_grid(args.sweep_quantities)
    max_trades = parse_int_grid(args.sweep_max_trades_per_day)
    max_hold_seconds = [minutes * 60 for minutes in parse_int_grid(args.sweep_max_hold_minutes)]
    opening_range_minutes = parse_int_grid(args.sweep_opening_range_minutes)
    last_entry_times = parse_time_grid(args.sweep_last_entry_times)
    max_opening_ranges = parse_optional_decimal_grid(args.sweep_max_opening_range_points)
    max_entry_bar_ranges = parse_optional_decimal_grid(args.sweep_max_entry_bar_range_points)

    for symbol in symbols:
        symbol = symbol.upper()
        if symbol == "NQ":
            quantpad_symbol = "NQ.FUT"
            stops = parse_decimal_grid(args.sweep_nq_stop_points)
            targets = parse_decimal_grid(args.sweep_nq_target_points)
            thresholds = parse_decimal_grid(args.sweep_nq_threshold_points)
            confirms = parse_decimal_grid(args.sweep_nq_confirm_points)
        elif symbol == "ES":
            quantpad_symbol = "ES.FUT"
            stops = parse_decimal_grid(args.sweep_es_stop_points)
            targets = parse_decimal_grid(args.sweep_es_target_points)
            thresholds = parse_decimal_grid(args.sweep_es_threshold_points)
            confirms = parse_decimal_grid(args.sweep_es_confirm_points)
        else:
            raise ValueError(f"unsupported sweep symbol: {symbol}")

        for family in families:
            for quantity in quantities:
                for stop_points in stops:
                    for target_points in targets:
                        for threshold_points in thresholds:
                            for confirm_points in confirms:
                                for or_minutes in opening_range_minutes:
                                    for hold_seconds in max_hold_seconds:
                                        for max_trades_per_day in max_trades:
                                            for last_entry_time in last_entry_times:
                                                for max_opening_range in max_opening_ranges:
                                                    for max_entry_bar_range in max_entry_bar_ranges:
                                                        if not candidate_is_reasonable(
                                                            family,
                                                            stop_points,
                                                            target_points,
                                                            quantity,
                                                        ):
                                                            continue
                                                        name_parts = [
                                                            symbol.lower(),
                                                            family.replace("opening_range_", "or_"),
                                                            f"q{quantity}",
                                                            f"s{compact_decimal(stop_points)}",
                                                            f"t{compact_decimal(target_points)}",
                                                            f"thr{compact_decimal(threshold_points)}",
                                                            f"or{or_minutes}",
                                                            f"h{hold_seconds // 60}",
                                                            f"m{max_trades_per_day}",
                                                        ]
                                                        if max_opening_range is not None:
                                                            name_parts.append(
                                                                f"orcap{compact_decimal(max_opening_range)}"
                                                            )
                                                        if max_entry_bar_range is not None:
                                                            name_parts.append(
                                                                f"barcap{compact_decimal(max_entry_bar_range)}"
                                                            )
                                                        name_parts.append(last_entry_time.strftime("%H%M"))
                                                        specs.append(
                                                            StrategySpec(
                                                                name="_".join(name_parts),
                                                                quantpad_symbol=quantpad_symbol,
                                                                contract_symbol=symbol,
                                                                family=family,
                                                                quantity=quantity,
                                                                stop_points=stop_points,
                                                                target_points=target_points,
                                                                threshold_points=threshold_points,
                                                                opening_range_minutes=or_minutes,
                                                                confirm_points=confirm_points,
                                                                max_opening_range_points=max_opening_range,
                                                                max_entry_bar_range_points=max_entry_bar_range,
                                                                max_hold_seconds=hold_seconds,
                                                                max_trades_per_day=max_trades_per_day,
                                                                first_entry_time=time(9, 35),
                                                                last_entry_time=last_entry_time,
                                                            )
                                                        )
    if not specs:
        raise ValueError("sweep grid produced no strategy candidates")
    return specs


def candidate_is_reasonable(
    family: str,
    stop_points: Decimal,
    target_points: Decimal,
    quantity: int,
) -> bool:
    if family in {"vwap_reversion", "vwap_failed_continuation"}:
        return target_points <= stop_points
    if family in {"opening_range_fade", "opening_range_breakout_pullback"}:
        return target_points >= stop_points
    return quantity > 0


def fetch_quantpad_bars(symbol: str, start: datetime, end: datetime, timeframe: str):
    import quantpad_data as qpd

    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    return qpd.get_bars(symbol, timeframe, start_ms, end_ms, as_df=True)


def iter_bars_from_frame(frame) -> Iterator[dict[str, Any]]:
    if frame is None or len(frame) == 0:
        return
    for row in frame.itertuples():
        ts = row.Index
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        local_ts = ts.astimezone(ET)
        if not (time(9, 30) <= local_ts.time() <= time(16, 0)):
            continue
        yield {
            "timestamp": local_ts,
            "open": Decimal(str(row.open)),
            "high": Decimal(str(row.high)),
            "low": Decimal(str(row.low)),
            "close": Decimal(str(row.close)),
            "volume": Decimal(str(row.volume)),
        }


def iter_chunks(start: datetime, end: datetime, days: int) -> Iterator[tuple[datetime, datetime]]:
    cursor = start
    while cursor < end:
        chunk_end = min(end, cursor + timedelta(days=days))
        yield cursor, chunk_end
        cursor = chunk_end


def run_backtest(args: argparse.Namespace) -> tuple[list[StrategyBacktest], list[dict[str, Any]]]:
    if args.api_key_stdin:
        os.environ["QUANTPAD_API_KEY"] = sys.stdin.readline().strip()

    start = parse_date_arg(args.start)
    end = parse_date_arg(args.end)
    if end <= start:
        raise ValueError("--end must be after --start")

    strategies = build_strategies(args)
    grouped: dict[str, list[StrategyBacktest]] = defaultdict(list)
    for spec in strategies:
        round_turn = (
            money(args.round_turn_cost)
            if args.round_turn_cost
            else TOPSTEP_ROUND_TURN_COSTS[spec.contract_symbol]
        )
        grouped[spec.quantpad_symbol].append(
            StrategyBacktest(
                spec=spec,
                account_tier=args.account,
                commission_round_turn=round_turn,
                slippage_ticks_per_side=Decimal(args.slippage_ticks_per_side),
                quick_pass_days=args.quick_pass_days,
            )
        )

    last_timestamp: Optional[datetime] = None
    for symbol, engines in grouped.items():
        for chunk_start, chunk_end in iter_chunks(start, end, args.chunk_days):
            print(
                json.dumps(
                    {
                        "event": "fetch",
                        "symbol": symbol,
                        "start": chunk_start.isoformat(),
                        "end": chunk_end.isoformat(),
                    }
                ),
                flush=True,
            )
            frame = fetch_quantpad_bars(symbol, chunk_start, chunk_end, args.timeframe)
            for bar in iter_bars_from_frame(frame):
                last_timestamp = bar["timestamp"]
                for engine in engines:
                    engine.process_bar(bar)

    for engines in grouped.values():
        for engine in engines:
            engine.finish(last_timestamp)

    all_engines = [engine for engines in grouped.values() for engine in engines]
    summaries = []
    for engine in all_engines:
        xfa_summary = build_xfa_summary(engine, args)
        summaries.append(engine.summary(xfa_summary))
    summaries.sort(
        key=lambda row: (
            Decimal(row["quick_pass_rate"]),
            Decimal(row["pass_rate"]),
            -Decimal(row["avg_days_to_pass"]) if Decimal(row["avg_days_to_pass"]) > 0 else Decimal("-999999"),
            Decimal(row["ev_per_challenge_attempt"]),
            Decimal(row["avg_trade_pnl"]),
        ),
        reverse=True,
    )
    return all_engines, summaries


def build_xfa_summary(engine: StrategyBacktest, args: argparse.Namespace) -> Optional[dict[str, Any]]:
    if args.skip_xfa:
        return None
    days = trade_days_from_records(engine.trades)
    if not days:
        return None
    account_config = xfa_account_config_for_tier(args.account, args.xfa_daily_loss_limit)
    payout_policy = XFAPayoutPolicy(path=XFAPayoutPath(args.payout_path))
    risk_policy = XFARiskPolicy(
        danger_buffer=Decimal(args.xfa_danger_buffer)
        if args.xfa_danger_buffer
        else None,
        danger_scale=Decimal(args.xfa_danger_scale),
        protected_profit=Decimal(args.xfa_protected_profit)
        if args.xfa_protected_profit
        else None,
        protected_scale=Decimal(args.xfa_protected_scale),
    )
    completed = [
        attempt
        for attempt in engine.attempts
        if attempt.outcome in {ChallengeStatus.PASSED.value, ChallengeStatus.FAILED_MLL.value}
    ]
    passes = [attempt for attempt in completed if attempt.outcome == ChallengeStatus.PASSED.value]
    challenge_pass_rate = decimal_ratio(len(passes), len(completed))
    if challenge_pass_rate <= 0:
        challenge_pass_rate = Decimal("0.0001")
    results = run_xfa_bootstrap(
        days=days,
        trials=args.xfa_trials,
        account_config=account_config,
        payout_policy=payout_policy,
        risk_policy=risk_policy,
        max_days=args.xfa_max_days,
        seed=args.seed,
        pnl_haircut_per_trade=args.xfa_extra_haircut_per_trade,
        liquidation_slippage=args.xfa_liquidation_slippage,
    )
    return summarize_xfa_trials(
        results,
        challenge_cost=args.challenge_cost,
        challenge_pass_rate=challenge_pass_rate,
        activation_fee=args.activation_fee,
    )


def xfa_account_config_for_tier(
    tier: str,
    daily_loss_limit: Optional[str],
) -> XFAAccountConfig:
    challenge_config = get_account_config(tier)
    payout_caps = {
        "50K": {
            "standard_payout_cap": "2000",
            "consistency_payout_cap": "3000",
            "standard_payout_cap_with_dll": "4000",
            "consistency_payout_cap_with_dll": "6000",
        },
        "100K": {
            "standard_payout_cap": "3000",
            "consistency_payout_cap": "4000",
            "standard_payout_cap_with_dll": "6000",
            "consistency_payout_cap_with_dll": "8000",
        },
        "150K": {
            "standard_payout_cap": "5000",
            "consistency_payout_cap": "6000",
            "standard_payout_cap_with_dll": "10000",
            "consistency_payout_cap_with_dll": "12000",
        },
    }[challenge_config.name]
    return XFAAccountConfig(
        name=challenge_config.name,
        starting_balance=0,
        max_loss_limit=challenge_config.mll_distance,
        mll_lock_profit=challenge_config.mll_distance,
        locked_mll=0,
        daily_loss_limit=Decimal(daily_loss_limit) if daily_loss_limit else None,
        **payout_caps,
    )


def trade_days_from_records(records: Sequence[TradeRecord]) -> list[TradingDay]:
    grouped: dict[date, list[TradePath]] = defaultdict(list)
    for trade in records:
        grouped[trade.session_date].append(
            TradePath(
                session_date=trade.session_date,
                realized_pnl=trade.realized_pnl,
                mae_pnl=trade.mae_pnl,
                mfe_pnl=trade.mfe_pnl,
                exit_reason=trade.exit_reason,
                risk_state="",
            )
        )
    return [
        TradingDay(session_date=session, trades=tuple(trades))
        for session, trades in sorted(grouped.items())
    ]


def write_outputs(
    args: argparse.Namespace,
    engines: Sequence[StrategyBacktest],
    summaries: Sequence[dict[str, Any]],
) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "summary.csv", list(summaries))
    _write_json(out / "summary.json", {"summaries": summaries, "metadata": metadata(args)})

    trade_rows = []
    attempt_rows = []
    for engine in engines:
        trade_rows.extend(trade.to_row() for trade in engine.trades)
        attempt_rows.extend(attempt.to_row() for attempt in engine.attempts)
    _write_csv(out / "trades.csv", trade_rows)
    _write_csv(out / "attempts.csv", attempt_rows)


def metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "start": args.start,
        "end": args.end,
        "timeframe": args.timeframe,
        "account": args.account,
        "slippage_ticks_per_side": args.slippage_ticks_per_side,
        "quick_pass_days": args.quick_pass_days,
        "sweep": args.sweep,
        "strategy_count": len(build_strategies(args)),
        "challenge_cost": args.challenge_cost,
        "activation_fee": args.activation_fee,
        "payout_path": args.payout_path,
        "xfa_trials": args.xfa_trials,
        "xfa_max_days": args.xfa_max_days,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    ticks = (value / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return ticks * tick


def decimal_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def decimal_mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return money(0)
    return money(Decimal(str(mean(values))))


def failure_counts(attempts: Sequence[AttemptRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for attempt in attempts:
        reason = attempt.failure_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_decimal_grid(value: str) -> list[Decimal]:
    items = [Decimal(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("decimal grid cannot be empty")
    return items


def parse_optional_decimal_grid(value: str) -> list[Optional[Decimal]]:
    items: list[Optional[Decimal]] = []
    for item in value.split(","):
        normalized = item.strip().lower()
        if not normalized:
            continue
        if normalized in {"none", "off", "unfiltered"}:
            items.append(None)
        else:
            items.append(Decimal(normalized))
    if not items:
        raise ValueError("optional decimal grid cannot be empty")
    return items


def parse_int_grid(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("integer grid cannot be empty")
    return items


def parse_time_grid(value: str) -> list[time]:
    items = [
        datetime.strptime(item.strip(), "%H:%M").time()
        for item in value.split(",")
        if item.strip()
    ]
    if not items:
        raise ValueError("time grid cannot be empty")
    return items


def compact_decimal(value: Decimal) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def parse_date_arg(value: str) -> datetime:
    if len(value) == 10:
        return datetime.fromisoformat(value).replace(tzinfo=ET).astimezone(UTC)
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=ET)
    return parsed.astimezone(UTC)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--output-dir", default="topstep_quantpad_backtest")
    parser.add_argument("--start", required=True, help="Inclusive local ET date or datetime.")
    parser.add_argument("--end", required=True, help="Exclusive local ET date or datetime.")
    parser.add_argument("--timeframe", default="1s", choices=["1s", "1m"])
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--account", default="50K")
    parser.add_argument("--strategies", default="")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument(
        "--sweep-symbols",
        default="NQ,ES",
        help="Comma grid. Supported: NQ,ES.",
    )
    parser.add_argument(
        "--sweep-families",
        default="vwap_reversion,opening_range_fade",
        help=(
            "Comma grid: vwap_reversion,vwap_failed_continuation,"
            "opening_range_fade,opening_range_breakout_pullback"
        ),
    )
    parser.add_argument("--sweep-quantities", default="1,2")
    parser.add_argument("--sweep-max-trades-per-day", default="1,3")
    parser.add_argument("--sweep-max-hold-minutes", default="10")
    parser.add_argument("--sweep-opening-range-minutes", default="5")
    parser.add_argument("--sweep-last-entry-times", default="11:30")
    parser.add_argument(
        "--sweep-max-opening-range-points",
        default="none",
        help="Comma grid of opening range size caps in points. Use none to disable.",
    )
    parser.add_argument(
        "--sweep-max-entry-bar-range-points",
        default="none",
        help="Comma grid of current-bar range caps in points. Use none to disable.",
    )
    parser.add_argument("--sweep-nq-stop-points", default="10,15,20")
    parser.add_argument("--sweep-nq-target-points", default="8,10,12,40")
    parser.add_argument("--sweep-nq-threshold-points", default="10,15")
    parser.add_argument("--sweep-nq-confirm-points", default="0")
    parser.add_argument("--sweep-es-stop-points", default="5,8")
    parser.add_argument("--sweep-es-target-points", default="3,5,20")
    parser.add_argument("--sweep-es-threshold-points", default="3,4")
    parser.add_argument("--sweep-es-confirm-points", default="0")
    parser.add_argument("--slippage-ticks-per-side", default="0.5")
    parser.add_argument("--quick-pass-days", type=int, default=20)
    parser.add_argument("--round-turn-cost")
    parser.add_argument("--challenge-cost", default="85")
    parser.add_argument("--activation-fee", default="0")
    parser.add_argument("--payout-path", default="consistency", choices=["standard", "consistency"])
    parser.add_argument("--xfa-trials", type=int, default=1000)
    parser.add_argument("--xfa-max-days", type=int, default=120)
    parser.add_argument("--xfa-daily-loss-limit")
    parser.add_argument("--xfa-danger-buffer", default="1000")
    parser.add_argument("--xfa-danger-scale", default="0.5")
    parser.add_argument("--xfa-protected-profit")
    parser.add_argument("--xfa-protected-scale", default="1")
    parser.add_argument("--xfa-extra-haircut-per-trade", default="0")
    parser.add_argument("--xfa-liquidation-slippage", default="0")
    parser.add_argument("--skip-xfa", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    engines, summaries = run_backtest(args)
    write_outputs(args, engines, summaries)
    print(json.dumps({"output_dir": args.output_dir, "summaries": summaries}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
