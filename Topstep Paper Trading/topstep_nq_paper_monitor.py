#!/usr/bin/env python3
"""Local paper monitor for the NQ Topstep OR-fade strategy.

The monitor processes completed 1-minute OHLC bars incrementally, applies the
same Topstep rule simulator and futures execution adapter used by the research
runner, and exposes account/strategy state through a small Flask dashboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time as time_module
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from flask import Flask, Response, jsonify, request

from topstep_data_loader import (
    CSVBarSchema,
    FuturesBar,
    detect_csv_schema,
    iter_futures_bars,
    parse_decimal,
    parse_timestamp,
)
from topstep_execution_adapter import (
    DEFAULT_CONTRACT_SPECS,
    ContractSpec,
    ExecutionError,
    Fill,
    FillSide,
    FuturesExecutionAdapter,
    topstep_commission_per_side,
)
from topstep_nq_strategy_runner import (
    ActiveTrade,
    OpeningRangeBreakoutConfig,
    SessionState,
    TradeRecord,
    session_date_for_timestamp,
)
from topstep_rule_simulator import ChallengeStatus, RuleEvent, RuleEventType, TopstepRuleSimulator, money


DEFAULT_OUTPUT_DIR = "topstep_nq_paper_monitor"
SIX_SYMBOLS = ("NQ", "ES", "RTY", "CL", "GC", "6E")
EIGHT_SYMBOLS = ("NQ", "ES", "RTY", "CL", "GC", "6E", "MNQ", "MES")
MARKET_DEFAULTS = {
    "NQ": {
        "yahoo_symbol": "NQ=F",
        "stop": "20",
        "target": "40",
        "buffer": "10",
        "max_or": "55",
        "max_gap": "125",
    },
    "MNQ": {
        "yahoo_symbol": "NQ=F",
        "stop": "20",
        "target": "40",
        "buffer": "10",
        "max_or": "55",
        "max_gap": "125",
    },
    "ES": {
        "yahoo_symbol": "ES=F",
        "stop": "8",
        "target": "16",
        "buffer": "4",
        "max_or": "22",
        "max_gap": "50",
    },
    "MES": {
        "yahoo_symbol": "ES=F",
        "stop": "8",
        "target": "16",
        "buffer": "4",
        "max_or": "22",
        "max_gap": "50",
    },
    "RTY": {
        "yahoo_symbol": "RTY=F",
        "stop": "8",
        "target": "16",
        "buffer": "4",
        "max_or": "22",
        "max_gap": "50",
    },
    "CL": {
        "yahoo_symbol": "CL=F",
        "stop": "0.40",
        "target": "0.80",
        "buffer": "0.20",
        "max_or": "1.10",
        "max_gap": "2.50",
    },
    "GC": {
        "yahoo_symbol": "GC=F",
        "stop": "4",
        "target": "8",
        "buffer": "2",
        "max_or": "11",
        "max_gap": "25",
    },
    "6E": {
        "yahoo_symbol": "6E=F",
        "stop": "0.00320",
        "target": "0.00640",
        "buffer": "0.00160",
        "max_or": "0.00880",
        "max_gap": "0.02000",
    },
}


def default_strategy_config(symbol: str = "NQ") -> OpeningRangeBreakoutConfig:
    symbol = symbol.upper()
    if symbol not in MARKET_DEFAULTS:
        known = ", ".join(sorted(MARKET_DEFAULTS))
        raise ValueError(f"unsupported monitor symbol {symbol!r}; expected one of {known}")
    defaults = MARKET_DEFAULTS[symbol]
    return OpeningRangeBreakoutConfig(
        strategy_family="orb_fade",
        account_tier="50K",
        data_symbol=symbol,
        contract_symbol=symbol,
        quantity=1,
        opening_range_minutes=5,
        breakout_buffer_points=Decimal(defaults["buffer"]),
        stop_points=Decimal(defaults["stop"]),
        target_points=Decimal(defaults["target"]),
        max_hold_minutes=180,
        max_trades_per_session=1,
        last_entry_time=datetime.strptime("11:30", "%H:%M").time(),
        force_exit_time=datetime.strptime("16:00", "%H:%M").time(),
        commission_per_contract=topstep_commission_per_side(symbol),
        max_opening_range_points=Decimal(defaults["max_or"]),
        max_opening_gap_points=Decimal(defaults["max_gap"]),
    )


@dataclass
class SourceStatus:
    mode: str = "manual"
    message: str = "waiting for bars"
    updated_at: str = ""
    last_error: str = ""
    running: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "message": self.message,
            "updated_at": self.updated_at,
            "last_error": self.last_error,
            "running": self.running,
        }


class NQPaperEngine:
    def __init__(
        self,
        config: Optional[OpeningRangeBreakoutConfig] = None,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.config = config or default_strategy_config()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._source_status = SourceStatus(updated_at=utc_now())
        self._reset_unlocked()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_unlocked()
            self._record_event("reset", "paper account reset", None)
            self._persist_state_unlocked()
            return self.snapshot()

    def _reset_unlocked(self) -> None:
        self.contract_specs = {
            symbol: ContractSpec(
                spec.symbol,
                tick_size=spec.tick_size,
                tick_value=spec.tick_value,
                commission_per_contract=self.config.commission_per_contract,
                is_micro=spec.is_micro,
            )
            for symbol, spec in DEFAULT_CONTRACT_SPECS.items()
        }
        self.sim = TopstepRuleSimulator.from_tier(self.config.account_tier)
        self.adapter = FuturesExecutionAdapter(self.sim, contract_specs=self.contract_specs)
        self.session_state: Optional[SessionState] = None
        self.active_trade: Optional[ActiveTrade] = None
        self.previous_bar: Optional[FuturesBar] = None
        self.previous_session_close: Optional[Decimal] = None
        self.start_time: Optional[datetime] = None
        self.last_closed_balance = self.sim.closed_balance
        self.last_rule_event_type = ""
        self.last_rule_event_detail = ""
        self.last_exit_reason = ""
        self.bars_processed = 0
        self.duplicate_or_stale_bars = 0
        self.trades: list[TradeRecord] = []
        self.events: list[dict[str, Any]] = []
        self.risk_filter_skips: dict[str, int] = {}
        self.equity_points: list[dict[str, Any]] = []
        self.last_bar_time: Optional[datetime] = None

    def set_source_status(
        self,
        mode: str,
        message: str,
        *,
        running: bool,
        last_error: str = "",
    ) -> None:
        with self._lock:
            self._source_status = SourceStatus(
                mode=mode,
                message=message,
                updated_at=utc_now(),
                last_error=last_error,
                running=running,
            )

    def process_bar(self, bar: FuturesBar, allow_entries: bool = True) -> dict[str, Any]:
        with self._lock:
            if self.last_bar_time is not None and bar.timestamp <= self.last_bar_time:
                self.duplicate_or_stale_bars += 1
                return {"accepted": False, "reason": "stale_or_duplicate_bar"}
            self._process_bar_unlocked(bar, allow_entries=allow_entries)
            self._persist_state_unlocked()
            return {"accepted": True, "snapshot": self.snapshot_unlocked()}

    def process_bars(self, bars: Iterable[tuple[FuturesBar, bool]]) -> dict[str, int]:
        accepted = 0
        stale = 0
        tradable = 0
        with self._lock:
            for bar, allow_entries in bars:
                if self.last_bar_time is not None and bar.timestamp <= self.last_bar_time:
                    self.duplicate_or_stale_bars += 1
                    stale += 1
                    continue
                self._process_bar_unlocked(bar, allow_entries=allow_entries)
                accepted += 1
                if allow_entries:
                    tradable += 1
            if accepted:
                self._persist_state_unlocked()
        return {"accepted": accepted, "stale": stale, "tradable": tradable}

    def _process_bar_unlocked(self, bar: FuturesBar, allow_entries: bool = True) -> None:
        self.bars_processed += 1
        self.last_bar_time = bar.timestamp
        if self.start_time is None:
            self.start_time = bar.timestamp

        bar_session_date = session_date_for_timestamp(bar.timestamp)
        if self.session_state is None:
            self.session_state = self._new_session_state(bar_session_date)
        elif bar_session_date != self.session_state.session_date:
            self._roll_session_unlocked(bar_session_date)

        if self.sim.status != ChallengeStatus.ACTIVE:
            self.previous_bar = bar
            self._append_equity_point(bar)
            return

        if self.active_trade is not None:
            terminal_now = self._manage_open_trade_unlocked(bar)
            self.previous_bar = bar
            self._append_equity_point(bar)
            if terminal_now:
                return

        if (
            self.active_trade is None
            and self.sim.status == ChallengeStatus.ACTIVE
            and not self.sim.day_locked
            and allow_entries
        ):
            assert self.session_state is not None
            self._update_opening_range_unlocked(self.session_state, bar)
            self._maybe_enter_unlocked(self.session_state, bar)
        elif self.active_trade is None:
            assert self.session_state is not None
            self._update_opening_range_unlocked(self.session_state, bar)

        self.previous_bar = bar
        self._append_equity_point(bar)

    def _roll_session_unlocked(self, new_session_date) -> None:
        if self.previous_bar is not None and self.active_trade is not None:
            self._force_exit_unlocked(
                self.previous_bar.timestamp,
                self.previous_bar.close,
                "session_end",
            )

        if self.sim.status == ChallengeStatus.ACTIVE and self.sim.session_active:
            try:
                event = self.sim.end_session()
                self._remember_rule_event(event, "session_end")
                if self.sim.status == ChallengeStatus.ACTIVE:
                    event = self.sim.start_next_session()
                    self._remember_rule_event(event, "session_start")
            except Exception as exc:
                self._record_event("session_error", str(exc), self.previous_bar)

        if self.previous_bar is not None:
            self.previous_session_close = self.previous_bar.close
        self.session_state = self._new_session_state(new_session_date)

    def _new_session_state(self, session_date) -> SessionState:
        rth_start_dt = datetime.combine(session_date, self.config.rth_start)
        return SessionState(
            session_date=session_date,
            opening_range_end=rth_start_dt
            + __import__("datetime").timedelta(minutes=self.config.opening_range_minutes),
            previous_session_close=self.previous_session_close,
        )

    def _update_opening_range_unlocked(self, session_state: SessionState, bar: FuturesBar) -> None:
        if (
            session_state.opening_gap_points is None
            and session_state.previous_session_close is not None
        ):
            session_state.opening_gap_points = abs(bar.open - session_state.previous_session_close)

        if not (
            bar.timestamp.time() >= self.config.rth_start
            and bar.timestamp < session_state.opening_range_end
        ):
            return

        session_state.opening_high = (
            bar.high
            if session_state.opening_high is None
            else max(session_state.opening_high, bar.high)
        )
        session_state.opening_low = (
            bar.low
            if session_state.opening_low is None
            else min(session_state.opening_low, bar.low)
        )

    def _maybe_enter_unlocked(self, session_state: SessionState, bar: FuturesBar) -> None:
        if session_state.trades_taken >= self.config.max_trades_per_session:
            return
        if not (
            session_state.opening_range_end <= bar.timestamp
            and bar.timestamp.time() <= self.config.last_entry_time
        ):
            return

        side = self._entry_side(session_state, bar)
        if side is None:
            return

        session_skip_reason = self._session_filter_skip_reason(session_state)
        if session_skip_reason is not None:
            self._record_risk_filter_skip(session_state, session_skip_reason)
            return

        entry_price = round_to_tick(bar.close, self._contract_spec().tick_size)
        stop_price, target_price = self._stop_target(entry_price, side)
        quantity = self.config.quantity

        try:
            result = self.adapter.process_fill(
                Fill(
                    self.config.contract_symbol,
                    side,
                    quantity,
                    entry_price,
                    timestamp=bar.timestamp.isoformat(sep=" "),
                )
            )
        except ExecutionError as exc:
            self._record_event("entry_rejected", str(exc), bar)
            return

        self._remember_rule_event(result.rule_event, "entry")
        session_state.trades_taken += 1
        entry_mll_buffer = money(self.sim.valuation - self.sim.active_mll)
        self.active_trade = ActiveTrade(
            attempt_id=1,
            session_date=session_state.session_date,
            risk_state="post_lock" if self.sim.mll_locked else "pre_lock",
            side=side,
            entry_time=bar.timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            entry_mll_buffer=entry_mll_buffer,
            min_mll_buffer=entry_mll_buffer,
        )
        self._record_event(
            "entry",
            f"{side.value.upper()} {quantity} {self.config.contract_symbol} @ {entry_price}",
            bar,
        )

    def _entry_side(self, session_state: SessionState, bar: FuturesBar) -> Optional[FillSide]:
        if not session_state.opening_range_ready:
            return None
        assert session_state.opening_high is not None
        assert session_state.opening_low is not None

        long_trigger = session_state.opening_high + self.config.breakout_buffer_points
        short_trigger = session_state.opening_low - self.config.breakout_buffer_points
        if self.config.strategy_family == "orb_fade":
            if bar.close > long_trigger:
                return FillSide.SELL
            if bar.close < short_trigger:
                return FillSide.BUY
        elif self.config.strategy_family == "orb_breakout":
            if bar.close > long_trigger:
                return FillSide.BUY
            if bar.close < short_trigger:
                return FillSide.SELL
        return None

    def _session_filter_skip_reason(self, session_state: SessionState) -> Optional[str]:
        if self.config.max_opening_range_points is not None:
            opening_range_size = session_state.opening_range_size
            if opening_range_size is not None and opening_range_size > self.config.max_opening_range_points:
                return "opening_range"
        if (
            self.config.max_opening_gap_points is not None
            and session_state.opening_gap_points is not None
            and session_state.opening_gap_points > self.config.max_opening_gap_points
        ):
            return "opening_gap"
        return None

    def _record_risk_filter_skip(self, session_state: SessionState, reason: str) -> None:
        if reason in session_state.recorded_skip_reasons:
            return
        session_state.recorded_skip_reasons.add(reason)
        self.risk_filter_skips[reason] = self.risk_filter_skips.get(reason, 0) + 1
        self._record_event("skip", f"session skipped by {reason} filter", None)

    def _manage_open_trade_unlocked(self, bar: FuturesBar) -> bool:
        assert self.active_trade is not None
        trade = self.active_trade
        forced_time_exit = (
            bar.timestamp.time() >= self.config.force_exit_time
            or int((bar.timestamp - trade.entry_time).total_seconds() // 60)
            >= self.config.max_hold_minutes
        )
        if forced_time_exit:
            self._update_trade_excursions(trade, self._trade_bar_path(trade, bar))
            self._force_exit_unlocked(bar.timestamp, bar.close, "time_exit")
            return self.sim.status != ChallengeStatus.ACTIVE

        exit_price = None
        exit_reason = None
        if trade.side == FillSide.BUY:
            if bar.low <= trade.stop_price:
                exit_price = trade.stop_price
                exit_reason = "stop"
            elif bar.high >= trade.target_price:
                exit_price = trade.target_price
                exit_reason = "target"
        else:
            if bar.high >= trade.stop_price:
                exit_price = trade.stop_price
                exit_reason = "stop"
            elif bar.low <= trade.target_price:
                exit_price = trade.target_price
                exit_reason = "target"

        if exit_price is not None and exit_reason is not None:
            self._update_trade_excursions(
                trade,
                self._trade_bar_path_until_exit(trade, bar, exit_price, exit_reason),
            )
            self._force_exit_unlocked(bar.timestamp, exit_price, exit_reason)
            return self.sim.status != ChallengeStatus.ACTIVE

        terminal_price = self._mark_bar_path_until_terminal(trade, bar)
        if terminal_price is not None:
            self._record_forced_rule_exit_unlocked(bar.timestamp, terminal_price)
            return True

        return False

    def _trade_bar_path(self, trade: ActiveTrade, bar: FuturesBar) -> tuple[Decimal, ...]:
        if trade.side == FillSide.BUY:
            return (bar.open, bar.low, bar.high, bar.close)
        return (bar.open, bar.high, bar.low, bar.close)

    def _trade_bar_path_until_exit(
        self,
        trade: ActiveTrade,
        bar: FuturesBar,
        exit_price: Decimal,
        exit_reason: str,
    ) -> tuple[Decimal, ...]:
        path = self._trade_bar_path(trade, bar)
        if exit_reason == "stop":
            return (path[0], exit_price)
        if exit_reason == "target":
            return (path[0], path[1], exit_price)
        return path

    def _update_trade_excursions(self, trade: ActiveTrade, price_path: Iterable[Decimal]) -> None:
        spec = self.contract_specs[self.config.contract_symbol]
        underwater = False
        for mark in price_path:
            favorable_points = (mark - trade.entry_price) * Decimal(trade.side.sign)
            open_pnl = money(
                favorable_points * spec.multiplier * Decimal(trade.quantity)
                - self.config.commission_per_contract * Decimal(trade.quantity)
            )
            mll_buffer = money(self.sim.closed_balance + open_pnl - self.sim.active_mll)
            trade.min_mll_buffer = min(trade.min_mll_buffer, mll_buffer)
            if favorable_points < 0:
                trade.mae_points = max(trade.mae_points, -favorable_points)
                trade.mae_pnl = min(trade.mae_pnl, open_pnl)
                underwater = True
            else:
                trade.mfe_points = max(trade.mfe_points, favorable_points)
                trade.mfe_pnl = max(trade.mfe_pnl, open_pnl)
        trade.bars_held += 1
        if underwater:
            trade.time_underwater_minutes += 1

    def _mark_bar_path_until_terminal(self, trade: ActiveTrade, bar: FuturesBar) -> Optional[Decimal]:
        observed_path = []
        for mark in self._trade_bar_path(trade, bar):
            observed_path.append(mark)
            event = self.adapter.mark_price(self.config.contract_symbol, mark).rule_event
            self._remember_rule_event(event, "mark")
            if event.event_type in {RuleEventType.DLL_BREACH, RuleEventType.MLL_BREACH}:
                self._update_trade_excursions(trade, tuple(observed_path))
                return mark
        self._update_trade_excursions(trade, tuple(observed_path))
        return None

    def _force_exit_unlocked(self, timestamp: datetime, exit_price: Decimal, exit_reason: str) -> None:
        if self.active_trade is None or self.sim.status != ChallengeStatus.ACTIVE:
            self.active_trade = None
            return

        trade = self.active_trade
        exit_side = FillSide.SELL if trade.side == FillSide.BUY else FillSide.BUY
        event = self.adapter.process_fill(
            Fill(
                self.config.contract_symbol,
                exit_side,
                trade.quantity,
                round_to_tick(exit_price, self._contract_spec().tick_size),
                timestamp=timestamp.isoformat(sep=" "),
            )
        ).rule_event
        self._remember_rule_event(event, exit_reason)
        realized = money(self.sim.closed_balance - self.last_closed_balance)
        self.last_closed_balance = self.sim.closed_balance
        record = self._closed_trade_record(trade, timestamp, exit_price, realized, exit_reason, event)
        self.trades.append(record)
        self._append_trade_row(record)
        self._record_event(
            "exit",
            f"{exit_reason} @ {round_to_tick(exit_price, self._contract_spec().tick_size)}; pnl {realized}",
            None,
        )
        self.active_trade = None

    def _record_forced_rule_exit_unlocked(self, timestamp: datetime, exit_price: Decimal) -> None:
        if self.active_trade is None:
            return
        trade = self.active_trade
        realized = money(self.sim.closed_balance - self.last_closed_balance)
        self.last_closed_balance = self.sim.closed_balance
        event_type = self.last_rule_event_type or (
            RuleEventType.MLL_BREACH.value
            if self.sim.status == ChallengeStatus.FAILED_MLL
            else RuleEventType.DLL_BREACH.value
        )
        event = RuleEvent(
            event_type=RuleEventType(event_type),
            detail=self.last_rule_event_detail,
            day_number=self.sim.day_number,
            closed_balance=self.sim.closed_balance,
            open_pnl=self.sim.open_pnl,
            valuation=self.sim.valuation,
            active_mll=self.sim.active_mll,
            active_dll_threshold=self.sim.active_dll_threshold,
            status=self.sim.status,
            day_locked=self.sim.day_locked,
            mll_locked=self.sim.mll_locked,
        )
        exit_reason = self.last_exit_reason or event.event_type.value
        record = self._closed_trade_record(trade, timestamp, exit_price, realized, exit_reason, event)
        self.trades.append(record)
        self._append_trade_row(record)
        self._record_event("rule_exit", f"{event.event_type.value} @ {exit_price}; pnl {realized}", None)
        self.active_trade = None

    def _closed_trade_record(
        self,
        trade: ActiveTrade,
        timestamp: datetime,
        exit_price: Decimal,
        realized: Decimal,
        exit_reason: str,
        event: RuleEvent,
    ) -> TradeRecord:
        return TradeRecord(
            attempt_id=1,
            session_date=trade.session_date.isoformat(),
            risk_state=trade.risk_state,
            side=trade.side.value,
            quantity=trade.quantity,
            entry_time=trade.entry_time.isoformat(sep=" "),
            exit_time=timestamp.isoformat(sep=" "),
            entry_price=trade.entry_price,
            exit_price=round_to_tick(exit_price, self._contract_spec().tick_size),
            realized_pnl=money(realized),
            exit_reason=exit_reason,
            rule_event=event.event_type.value,
            mae_points=trade.mae_points,
            mfe_points=trade.mfe_points,
            mae_pnl=trade.mae_pnl,
            mfe_pnl=trade.mfe_pnl,
            time_to_target_minutes=time_to_target_minutes(trade, timestamp, exit_reason),
            time_underwater_minutes=trade.time_underwater_minutes,
            bars_held=trade.bars_held,
            entry_mll_buffer=trade.entry_mll_buffer,
            min_mll_buffer=trade.min_mll_buffer,
        )

    def _stop_target(self, entry_price: Decimal, side: FillSide) -> tuple[Decimal, Decimal]:
        if side == FillSide.BUY:
            return entry_price - self.config.stop_points, entry_price + self.config.target_points
        return entry_price + self.config.stop_points, entry_price - self.config.target_points

    def _contract_spec(self) -> ContractSpec:
        return self.contract_specs[self.config.contract_symbol]

    def _remember_rule_event(self, event: RuleEvent, exit_reason: str) -> None:
        self.last_rule_event_type = event.event_type.value
        self.last_rule_event_detail = event.detail
        self.last_exit_reason = exit_reason

    def _append_equity_point(self, bar: FuturesBar) -> None:
        self.equity_points.append(
            {
                "ts": bar.timestamp.isoformat(sep=" "),
                "closed_balance": decimal_to_float(self.sim.closed_balance),
                "open_pnl": decimal_to_float(self.sim.open_pnl),
                "valuation": decimal_to_float(self.sim.valuation),
                "active_mll": decimal_to_float(self.sim.active_mll),
                "dll_floor": decimal_to_float(self.sim.active_dll_threshold),
            }
        )
        if len(self.equity_points) > 2000:
            self.equity_points = self.equity_points[-2000:]

    def _record_event(self, kind: str, message: str, bar: Optional[FuturesBar]) -> None:
        row = {
            "ts": (bar.timestamp.isoformat(sep=" ") if bar else utc_now()),
            "kind": kind,
            "message": message,
            "status": self.sim.status.value,
            "closed_balance": str(self.sim.closed_balance),
            "valuation": str(self.sim.valuation),
            "mll_buffer": str(money(self.sim.valuation - self.sim.active_mll)),
        }
        self.events.append(row)
        if len(self.events) > 500:
            self.events = self.events[-500:]
        self._append_csv_row(self.output_dir / "events.csv", row)

    def _append_trade_row(self, trade: TradeRecord) -> None:
        self._append_csv_row(self.output_dir / "trades.csv", trade.to_row())

    def _append_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _persist_state_unlocked(self) -> None:
        state_path = self.output_dir / "state.json"
        state_path.write_text(json.dumps(self.snapshot_unlocked(), indent=2) + "\n", encoding="utf-8")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict[str, Any]:
        snap = self.sim.snapshot()
        session = self._session_snapshot()
        active = self._active_trade_snapshot()
        stats = self._stats_snapshot()
        return {
            "ts": utc_now(),
            "output_dir": str(self.output_dir),
            "source": self._source_status.to_dict(),
            "config": config_to_dict(self.config),
            "bars_processed": self.bars_processed,
            "duplicate_or_stale_bars": self.duplicate_or_stale_bars,
            "last_bar_time": self.last_bar_time.isoformat(sep=" ") if self.last_bar_time else "",
            "account": {
                "status": snap.status.value,
                "day_number": snap.day_number,
                "closed_balance": decimal_to_float(snap.closed_balance),
                "open_pnl": decimal_to_float(snap.open_pnl),
                "valuation": decimal_to_float(snap.valuation),
                "active_mll": decimal_to_float(snap.active_mll),
                "active_dll_threshold": decimal_to_float(snap.active_dll_threshold),
                "target_balance": decimal_to_float(snap.target_balance),
                "mll_buffer": decimal_to_float(snap.valuation - snap.active_mll),
                "dll_buffer": decimal_to_float(
                    None
                    if snap.active_dll_threshold is None
                    else snap.valuation - snap.active_dll_threshold
                ),
                "mll_locked": snap.mll_locked,
                "day_locked": snap.day_locked,
                "dll_breach_count": snap.dll_breach_count,
            },
            "session": session,
            "active_trade": active,
            "stats": stats,
            "risk_filter_skips": dict(self.risk_filter_skips),
            "recent_trades": [trade_to_dict(t) for t in self.trades[-30:]],
            "recent_events": self.events[-80:],
            "equity_curve": self.equity_points,
        }

    def _session_snapshot(self) -> dict[str, Any]:
        if self.session_state is None:
            return {}
        high = self.session_state.opening_high
        low = self.session_state.opening_low
        long_trigger = high + self.config.breakout_buffer_points if high is not None else None
        short_trigger = low - self.config.breakout_buffer_points if low is not None else None
        skip_reason = self._session_filter_skip_reason(self.session_state)
        return {
            "session_date": self.session_state.session_date.isoformat(),
            "opening_range_end": self.session_state.opening_range_end.isoformat(sep=" "),
            "opening_high": decimal_to_float(high),
            "opening_low": decimal_to_float(low),
            "opening_range_size": decimal_to_float(self.session_state.opening_range_size),
            "opening_gap_points": decimal_to_float(self.session_state.opening_gap_points),
            "long_trigger": decimal_to_float(long_trigger),
            "short_trigger": decimal_to_float(short_trigger),
            "trades_taken": self.session_state.trades_taken,
            "skip_reason": skip_reason or "",
            "ready": self.session_state.opening_range_ready,
        }

    def _active_trade_snapshot(self) -> dict[str, Any]:
        trade = self.active_trade
        if trade is None:
            return {}
        pos = self.adapter.position(self.config.contract_symbol)
        return {
            "side": trade.side.value,
            "quantity": trade.quantity,
            "entry_time": trade.entry_time.isoformat(sep=" "),
            "entry_price": decimal_to_float(trade.entry_price),
            "stop_price": decimal_to_float(trade.stop_price),
            "target_price": decimal_to_float(trade.target_price),
            "mae_points": decimal_to_float(trade.mae_points),
            "mfe_points": decimal_to_float(trade.mfe_points),
            "mae_pnl": decimal_to_float(trade.mae_pnl),
            "mfe_pnl": decimal_to_float(trade.mfe_pnl),
            "entry_mll_buffer": decimal_to_float(trade.entry_mll_buffer),
            "min_mll_buffer": decimal_to_float(trade.min_mll_buffer),
            "bars_held": trade.bars_held,
            "time_underwater_minutes": trade.time_underwater_minutes,
            "position_qty": 0 if pos is None else pos.quantity,
        }

    def _stats_snapshot(self) -> dict[str, Any]:
        wins = [t for t in self.trades if t.realized_pnl > 0]
        losses = [t for t in self.trades if t.realized_pnl < 0]
        total = sum((t.realized_pnl for t in self.trades), Decimal("0"))
        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": decimal_to_float(safe_ratio(len(wins), len(self.trades))),
            "realized_pnl": decimal_to_float(money(total)),
            "avg_trade_pnl": decimal_to_float(safe_money_ratio(total, len(self.trades))),
            "targets": sum(1 for t in self.trades if t.exit_reason == "target"),
            "stops": sum(1 for t in self.trades if t.exit_reason == "stop"),
            "rule_exits": sum(
                1
                for t in self.trades
                if t.rule_event in {RuleEventType.DLL_BREACH.value, RuleEventType.MLL_BREACH.value}
            ),
        }


class MultiSymbolPaperMonitor:
    def __init__(
        self,
        symbols: Iterable[str],
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        unknown = [symbol for symbol in self.symbols if symbol not in MARKET_DEFAULTS]
        if unknown:
            known = ", ".join(sorted(MARKET_DEFAULTS))
            raise ValueError(f"unsupported monitor symbols {unknown}; expected one of {known}")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.engines = {
            symbol: NQPaperEngine(
                config=default_strategy_config(symbol),
                output_dir=self.output_dir / symbol,
            )
            for symbol in self.symbols
        }
        self._lock = threading.RLock()
        self._source_status = SourceStatus(updated_at=utc_now())

    def reset(self) -> dict[str, Any]:
        for engine in self.engines.values():
            engine.reset()
        return self.snapshot()

    def set_source_status(
        self,
        mode: str,
        message: str,
        *,
        running: bool,
        last_error: str = "",
    ) -> None:
        with self._lock:
            self._source_status = SourceStatus(
                mode=mode,
                message=message,
                updated_at=utc_now(),
                last_error=last_error,
                running=running,
            )

    def process_bar(self, bar: FuturesBar, allow_entries: bool = True) -> dict[str, Any]:
        symbol = bar.symbol.upper()
        if symbol not in self.engines:
            return {"accepted": False, "reason": f"unmonitored_symbol:{symbol}"}
        return self.engines[symbol].process_bar(bar, allow_entries=allow_entries)

    def snapshot(self) -> dict[str, Any]:
        engine_snaps = {symbol: engine.snapshot() for symbol, engine in self.engines.items()}
        stats = self._aggregate_stats(engine_snaps)
        account = self._aggregate_account(engine_snaps)
        recent_trades = []
        recent_events = []
        for symbol, snap in engine_snaps.items():
            for trade in snap.get("recent_trades", [])[-10:]:
                row = dict(trade)
                row["contract_symbol"] = symbol
                recent_trades.append(row)
            for event in snap.get("recent_events", [])[-10:]:
                row = dict(event)
                row["contract_symbol"] = symbol
                recent_events.append(row)

        recent_trades.sort(key=lambda row: row.get("exit_time") or row.get("entry_time") or "")
        recent_events.sort(key=lambda row: row.get("ts") or "")
        return {
            "ts": utc_now(),
            "multi_symbol": True,
            "symbols": list(self.symbols),
            "output_dir": str(self.output_dir),
            "source": self._source_status.to_dict(),
            "config": {
                "symbols": list(self.symbols),
                "symbol_set": "eight" if self.symbols == EIGHT_SYMBOLS else "six"
                if self.symbols == SIX_SYMBOLS
                else "custom",
            },
            "account": account,
            "stats": stats,
            "engines": engine_snaps,
            "recent_trades": recent_trades[-50:],
            "recent_events": recent_events[-80:],
            "equity_curve": [],
        }

    def _aggregate_stats(self, engine_snaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
        trades = sum(int(snap["stats"]["trades"]) for snap in engine_snaps.values())
        wins = sum(int(snap["stats"]["wins"]) for snap in engine_snaps.values())
        losses = sum(int(snap["stats"]["losses"]) for snap in engine_snaps.values())
        realized = sum(Decimal(str(snap["stats"]["realized_pnl"])) for snap in engine_snaps.values())
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": decimal_to_float(safe_ratio(wins, trades)),
            "realized_pnl": decimal_to_float(money(realized)),
            "avg_trade_pnl": decimal_to_float(safe_money_ratio(realized, trades)),
            "targets": sum(int(snap["stats"]["targets"]) for snap in engine_snaps.values()),
            "stops": sum(int(snap["stats"]["stops"]) for snap in engine_snaps.values()),
            "rule_exits": sum(int(snap["stats"]["rule_exits"]) for snap in engine_snaps.values()),
        }

    def _aggregate_account(self, engine_snaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
        active_count = sum(1 for snap in engine_snaps.values() if snap["account"]["status"] == "active")
        valuation = sum(Decimal(str(snap["account"]["valuation"])) for snap in engine_snaps.values())
        closed_balance = sum(Decimal(str(snap["account"]["closed_balance"])) for snap in engine_snaps.values())
        open_pnl = sum(Decimal(str(snap["account"]["open_pnl"])) for snap in engine_snaps.values())
        return {
            "status": f"{active_count}/{len(engine_snaps)} active sleeves",
            "closed_balance": decimal_to_float(money(closed_balance)),
            "open_pnl": decimal_to_float(money(open_pnl)),
            "valuation": decimal_to_float(money(valuation)),
            "mll_buffer": None,
            "dll_buffer": None,
        }


def parse_bar_payload(payload: dict[str, Any], symbol: str = "NQ") -> FuturesBar:
    return FuturesBar(
        timestamp=parse_timestamp(str(payload["timestamp"])),
        symbol=str(payload.get("symbol") or symbol),
        open=parse_decimal(str(payload["open"])),
        high=parse_decimal(str(payload["high"])),
        low=parse_decimal(str(payload["low"])),
        close=parse_decimal(str(payload["close"])),
        volume=parse_decimal(str(payload.get("volume", "0"))),
    )


def run_csv_replay(
    engine: NQPaperEngine,
    data_path: str,
    start: Optional[datetime],
    end: Optional[datetime],
    delay_seconds: float,
) -> None:
    engine.set_source_status("replay", f"replaying {data_path}", running=True)
    try:
        for bar in iter_futures_bars(data_path, symbol=engine.config.data_symbol, start=start, end=end):
            engine.process_bar(bar)
            if delay_seconds > 0:
                time_module.sleep(delay_seconds)
        engine.set_source_status("replay", "replay complete", running=False)
    except Exception as exc:
        engine.set_source_status("replay", "replay stopped", running=False, last_error=str(exc))


def run_csv_tail(
    engine: NQPaperEngine,
    data_path: str,
    poll_seconds: float,
    from_start: bool,
) -> None:
    path = Path(data_path)
    engine.set_source_status("csv-tail", f"tailing {path}", running=True)
    while not path.exists():
        engine.set_source_status("csv-tail", f"waiting for {path}", running=True)
        time_module.sleep(max(0.25, poll_seconds))

    try:
        schema = detect_csv_schema(path)
        with path.open(newline="") as f:
            header_line = f.readline()
            if not header_line:
                raise ValueError(f"empty CSV file: {path}")
            headers = next(csv.reader([header_line]))
            if not from_start:
                f.seek(0, 2)

            while True:
                line = f.readline()
                if not line:
                    time_module.sleep(max(0.25, poll_seconds))
                    continue
                try:
                    values = next(csv.reader([line]))
                    row = dict(zip(headers, values))
                    bar = bar_from_csv_row(row, schema, engine.config.data_symbol)
                    accepted = engine.process_bar(bar)["accepted"]
                    message = "accepted appended bar" if accepted else "ignored appended stale bar"
                    engine.set_source_status("csv-tail", f"{message}: {bar.timestamp}", running=True)
                except Exception as exc:
                    engine.set_source_status(
                        "csv-tail",
                        "tail parse error; still watching",
                        running=True,
                        last_error=str(exc),
                    )
    except Exception as exc:
        engine.set_source_status("csv-tail", "tail stopped", running=False, last_error=str(exc))


def bar_from_csv_row(row: dict[str, str], schema: CSVBarSchema, fixed_symbol: str) -> FuturesBar:
    row_symbol = row[schema.symbol_col].upper() if schema.symbol_col is not None else fixed_symbol
    return FuturesBar(
        timestamp=parse_timestamp(row[schema.timestamp_col]),
        symbol=row_symbol,
        open=parse_decimal(row[schema.open_col]),
        high=parse_decimal(row[schema.high_col]),
        low=parse_decimal(row[schema.low_col]),
        close=parse_decimal(row[schema.close_col]),
        volume=parse_decimal(row[schema.volume_col]),
    )


def fetch_yahoo_chart_bars(
    yahoo_symbol: str,
    data_symbol: str,
    *,
    range_: str = "2d",
    interval: str = "1m",
) -> list[FuturesBar]:
    query = urllib.parse.urlencode(
        {
            "range": range_,
            "interval": interval,
            "includePrePost": "true",
        }
    )
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=20) as response:
        payload = json.load(response)
    return parse_yahoo_chart_payload(payload, data_symbol=data_symbol)


def parse_yahoo_chart_payload(payload: dict[str, Any], data_symbol: str) -> list[FuturesBar]:
    chart = payload.get("chart") or {}
    error = chart.get("error")
    if error:
        raise ValueError(f"Yahoo chart error: {error}")
    results = chart.get("result") or []
    if not results:
        return []

    result = results[0]
    meta = result.get("meta") or {}
    timezone_name = meta.get("exchangeTimezoneName") or "America/New_York"
    exchange_tz = ZoneInfo(timezone_name)
    timestamps = result.get("timestamp") or []
    indicators = result.get("indicators") or {}
    quote_blocks = indicators.get("quote") or []
    if not quote_blocks:
        return []
    quote = quote_blocks[0]

    bars: list[FuturesBar] = []
    now_utc = datetime.now(timezone.utc)
    for index, raw_ts in enumerate(timestamps):
        if raw_ts is None:
            continue
        ts_utc = datetime.fromtimestamp(int(raw_ts), timezone.utc)
        age_seconds = (now_utc - ts_utc).total_seconds()
        if ts_utc.second != 0 or age_seconds < 65:
            continue

        try:
            open_ = quote["open"][index]
            high = quote["high"][index]
            low = quote["low"][index]
            close = quote["close"][index]
            volume = quote.get("volume", [0] * len(timestamps))[index]
        except (IndexError, KeyError):
            continue
        if any(value is None for value in (open_, high, low, close)):
            continue

        ts_local = ts_utc.astimezone(exchange_tz).replace(tzinfo=None)
        bars.append(
            FuturesBar(
                timestamp=ts_local,
                symbol=data_symbol,
                open=Decimal(str(open_)),
                high=Decimal(str(high)),
                low=Decimal(str(low)),
                close=Decimal(str(close)),
                volume=Decimal(str(volume or 0)),
            )
        )
    return bars


def run_yahoo_chart_poll(engine: NQPaperEngine, yahoo_symbol: str, poll_seconds: int) -> None:
    engine.set_source_status("yahoo-chart", f"polling {yahoo_symbol}", running=True)
    source_started_at = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    ).replace(tzinfo=None, second=0, microsecond=0)
    while True:
        try:
            bars = fetch_yahoo_chart_bars(yahoo_symbol, engine.config.data_symbol)
            counts = engine.process_bars(
                (bar, bar.timestamp >= source_started_at) for bar in bars
            )
            engine.set_source_status(
                "yahoo-chart",
                f"polling {yahoo_symbol}; accepted {counts['accepted']} completed bars, "
                f"{counts['tradable']} tradable",
                running=True,
            )
        except Exception as exc:
            engine.set_source_status(
                "yahoo-chart",
                f"polling {yahoo_symbol}; last poll failed",
                running=True,
                last_error=str(exc),
            )
        time_module.sleep(max(5, poll_seconds))


def run_multi_yahoo_chart_poll(
    monitor: MultiSymbolPaperMonitor,
    poll_seconds: int,
) -> None:
    monitor.set_source_status("yahoo-chart", "polling multi-symbol Yahoo chart feed", running=True)
    source_started_at = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    ).replace(tzinfo=None, second=0, microsecond=0)
    while True:
        accepted_total = 0
        tradable_total = 0
        errors: list[str] = []
        for symbol, engine in monitor.engines.items():
            yahoo_symbol = MARKET_DEFAULTS[symbol]["yahoo_symbol"]
            try:
                bars = fetch_yahoo_chart_bars(yahoo_symbol, engine.config.data_symbol)
                counts = engine.process_bars(
                    (bar, bar.timestamp >= source_started_at) for bar in bars
                )
                accepted_total += counts["accepted"]
                tradable_total += counts["tradable"]
                engine.set_source_status(
                    "yahoo-chart",
                    f"polling {yahoo_symbol}; accepted {counts['accepted']} completed bars, "
                    f"{counts['tradable']} tradable",
                    running=True,
                )
            except Exception as exc:
                message = f"{symbol}:{exc}"
                errors.append(message)
                engine.set_source_status(
                    "yahoo-chart",
                    f"polling {yahoo_symbol}; last poll failed",
                    running=True,
                    last_error=str(exc),
                )

        error_text = "; ".join(errors[:3])
        monitor.set_source_status(
            "yahoo-chart",
            f"polled {len(monitor.engines)} symbols; accepted {accepted_total} bars, "
            f"{tradable_total} tradable",
            running=True,
            last_error=error_text,
        )
        time_module.sleep(max(5, poll_seconds))


def run_yfinance_poll(engine: NQPaperEngine, ticker_symbol: str, poll_seconds: int) -> None:
    engine.set_source_status("yfinance", f"polling {ticker_symbol}", running=True)
    try:
        import yfinance as yf
    except Exception as exc:
        engine.set_source_status("yfinance", "yfinance import failed", running=False, last_error=str(exc))
        return

    ticker = yf.Ticker(ticker_symbol)
    while True:
        try:
            history = ticker.history(period="2d", interval="1m", prepost=True)
            processed = 0
            if not history.empty:
                for ts, row in history.iterrows():
                    try:
                        if getattr(ts, "tzinfo", None) is not None:
                            ts = ts.tz_convert("America/New_York").to_pydatetime().replace(tzinfo=None)
                        else:
                            ts = ts.to_pydatetime()
                        bar = FuturesBar(
                            timestamp=ts,
                            symbol=engine.config.data_symbol,
                            open=Decimal(str(row["Open"])),
                            high=Decimal(str(row["High"])),
                            low=Decimal(str(row["Low"])),
                            close=Decimal(str(row["Close"])),
                            volume=Decimal(str(row.get("Volume", 0))),
                        )
                        if engine.process_bar(bar)["accepted"]:
                            processed += 1
                    except Exception:
                        continue
            engine.set_source_status(
                "yfinance",
                f"polling {ticker_symbol}; accepted {processed} new bars last poll",
                running=True,
            )
        except Exception as exc:
            engine.set_source_status(
                "yfinance",
                f"polling {ticker_symbol}; last poll failed",
                running=True,
                last_error=str(exc),
            )
        time_module.sleep(max(5, poll_seconds))


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>NQ OR Fade Paper Monitor</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #657085;
      --line: #d8dee8;
      --accent: #1957c2;
      --good: #0b7a43;
      --warn: #ad5b00;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      position: sticky;
      top: 0;
      z-index: 3;
    }
    h1 { font-size: 20px; margin: 0; }
    h2 { font-size: 15px; margin: 0 0 12px; }
    main { max-width: 1440px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; }
    .muted { color: var(--muted); }
    .status { display: flex; align-items: center; gap: 8px; white-space: nowrap; color: var(--muted); }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--warn); }
    .dot.ok { background: var(--good); }
    .dot.bad { background: var(--bad); }
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 12px; }
    .card, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .card { padding: 14px; min-height: 88px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .value { font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }
    .grid-2 { display: grid; grid-template-columns: 1.1fr .9fr; gap: 16px; }
    .grid-3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
    .panel { padding: 16px; overflow: hidden; }
    .chart-wrap { height: 260px; }
    canvas { width: 100%; height: 100%; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td {
      padding: 8px 6px;
      border-bottom: 1px solid #edf0f4;
      text-align: right;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    .kv { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
    .kv div { border-bottom: 1px solid #edf0f4; padding: 5px 0; display: flex; justify-content: space-between; gap: 8px; }
    code { background: #eef2ff; color: #1d3b8b; padding: 1px 4px; border-radius: 4px; }
    @media (max-width: 1050px) {
      .metrics { grid-template-columns: repeat(3, 1fr); }
      .grid-2, .grid-3 { grid-template-columns: 1fr; }
      header { flex-direction: column; }
    }
    @media (max-width: 640px) {
      main { padding: 14px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .value { font-size: 18px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>NQ OR Fade Paper Monitor</h1>
      <div class="muted" id="subtitle"></div>
    </div>
    <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">Loading</span></div>
  </header>

  <main>
    <section class="metrics">
      <div class="card"><div class="label">Valuation</div><div class="value" id="valuation">-</div></div>
      <div class="card"><div class="label">Realized PnL</div><div class="value" id="realized">-</div></div>
      <div class="card"><div class="label">Open PnL</div><div class="value" id="openPnl">-</div></div>
      <div class="card"><div class="label">MLL Buffer</div><div class="value" id="mllBuffer">-</div></div>
      <div class="card"><div class="label">DLL Buffer</div><div class="value" id="dllBuffer">-</div></div>
      <div class="card"><div class="label">Status</div><div class="value" id="accountStatus">-</div></div>
    </section>

    <section class="grid-2">
      <div class="panel">
        <h2>Equity Path</h2>
        <div class="chart-wrap"><canvas id="equityChart"></canvas></div>
      </div>
      <div class="panel">
        <h2>Strategy State</h2>
        <div class="kv" id="strategyState"></div>
      </div>
    </section>

    <section class="grid-3">
      <div class="panel">
        <h2>Active Trade</h2>
        <div class="kv" id="activeTrade"></div>
      </div>
      <div class="panel">
        <h2>Recent Trades</h2>
        <table>
          <thead><tr><th>Exit</th><th>Side</th><th>Entry</th><th>PnL</th></tr></thead>
          <tbody id="tradesTable"></tbody>
        </table>
      </div>
      <div class="panel">
        <h2>Events</h2>
        <table>
          <thead><tr><th>Time</th><th>Kind</th><th>Message</th><th>Buffer</th></tr></thead>
          <tbody id="eventsTable"></tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Run Assumptions</h2>
      <div class="kv" id="assumptions"></div>
    </section>
  </main>

  <script>
    const money = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', maximumFractionDigits: 0}) : '-';
    const money2 = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2}) : '-';
    const num = v => Number.isFinite(Number(v)) ? Number(v) : NaN;
    const pct = v => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + '%' : '-';
    const fixed = (v, d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '-';
    const set = (id, text) => document.getElementById(id).textContent = text;
    const clsPnl = v => num(v) >= 0 ? 'good' : 'bad';

    function kvHtml(rows) {
      return rows.map(([k, v]) => `<div><span class="muted">${k}</span><strong>${v}</strong></div>`).join('');
    }

    function rowHtml(cells) {
      return '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    }

    function renderTable(id, rows, mapper, empty) {
      const body = document.getElementById(id);
      body.innerHTML = rows.length ? rows.map(mapper).join('') : `<tr><td colspan="4" class="muted">${empty}</td></tr>`;
    }

    function drawChart(points) {
      const canvas = document.getElementById('equityChart');
      const rect = canvas.getBoundingClientRect();
      const scale = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.floor(rect.width * scale));
      canvas.height = Math.max(1, Math.floor(rect.height * scale));
      const ctx = canvas.getContext('2d');
      ctx.scale(scale, scale);
      ctx.clearRect(0, 0, rect.width, rect.height);
      const values = points.map(p => num(p.valuation)).filter(Number.isFinite);
      if (values.length < 2) {
        ctx.fillStyle = '#657085';
        ctx.fillText('Waiting for bars.', 12, 24);
        return;
      }
      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = Math.max(max - min, 1);
      const pad = 18;
      ctx.strokeStyle = '#d8dee8';
      ctx.beginPath();
      ctx.moveTo(0, rect.height - pad);
      ctx.lineTo(rect.width, rect.height - pad);
      ctx.stroke();
      ctx.strokeStyle = '#1957c2';
      ctx.lineWidth = 2;
      ctx.beginPath();
      values.forEach((value, i) => {
        const x = pad + i * ((rect.width - 2 * pad) / Math.max(values.length - 1, 1));
        const y = pad + (max - value) * ((rect.height - 2 * pad) / range);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    }

    async function refresh() {
      const response = await fetch('/api/snapshot', {cache: 'no-store'});
      const data = await response.json();
      const account = data.account || {};
      const session = data.session || {};
      const active = data.active_trade || {};
      const stats = data.stats || {};
      const config = data.config || {};
      const source = data.source || {};

      set('subtitle', `${config.strategy_family || ''} ${config.contract_symbol || ''}; ${data.output_dir || ''}`);
      set('valuation', money(account.valuation));
      set('realized', money2(stats.realized_pnl));
      set('openPnl', money2(account.open_pnl));
      set('mllBuffer', money2(account.mll_buffer));
      set('dllBuffer', money2(account.dll_buffer));
      set('accountStatus', account.status || '-');

      const staleMs = data.last_bar_time ? Date.now() - Date.parse(data.last_bar_time.replace(' ', 'T')) : Infinity;
      const dot = document.getElementById('statusDot');
      dot.className = 'dot ' + (account.status === 'failed_mll' ? 'bad' : staleMs < 180000 ? 'ok' : '');
      set('statusText', `${source.mode || 'manual'}: ${source.message || 'waiting'}; last bar ${data.last_bar_time || 'none'}`);

      document.getElementById('strategyState').innerHTML = kvHtml([
        ['Session', session.session_date || '-'],
        ['Opening Range', `${fixed(session.opening_low)} / ${fixed(session.opening_high)}`],
        ['OR Size', fixed(session.opening_range_size)],
        ['Gap', fixed(session.opening_gap_points)],
        ['Triggers', `${fixed(session.short_trigger)} / ${fixed(session.long_trigger)}`],
        ['Trades Today', session.trades_taken ?? '-'],
        ['Skip Reason', session.skip_reason || '-'],
        ['Bars Processed', data.bars_processed ?? 0],
      ]);

      document.getElementById('activeTrade').innerHTML = Object.keys(active).length ? kvHtml([
        ['Side', active.side],
        ['Qty', active.quantity],
        ['Entry', money2(active.entry_price)],
        ['Stop', money2(active.stop_price)],
        ['Target', money2(active.target_price)],
        ['MAE / MFE', `${fixed(active.mae_points)} / ${fixed(active.mfe_points)}`],
        ['Min MLL Buffer', money2(active.min_mll_buffer)],
        ['Bars Held', active.bars_held],
      ]) : '<div class="muted">No open paper trade.</div>';

      renderTable('tradesTable', (data.recent_trades || []).slice().reverse().slice(0, 14), row => rowHtml([
        row.exit_reason,
        row.side,
        money2(row.entry_price),
        `<span class="${clsPnl(row.realized_pnl)}">${money2(row.realized_pnl)}</span>`
      ]), 'No closed trades yet.');

      renderTable('eventsTable', (data.recent_events || []).slice().reverse().slice(0, 14), row => rowHtml([
        String(row.ts || '').slice(5, 16),
        row.kind,
        row.message,
        money2(row.mll_buffer)
      ]), 'No events yet.');

      document.getElementById('assumptions').innerHTML = kvHtml([
        ['Strategy', config.strategy_family],
        ['Risk Geometry', `${config.stop_points} stop / ${config.target_points} target`],
        ['Opening Range', `${config.opening_range_minutes} min, ${config.breakout_buffer_points} point buffer`],
        ['Filters', `OR <= ${config.max_opening_range_points}, gap <= ${config.max_opening_gap_points}`],
        ['Last Entry', config.last_entry_time],
        ['Cost Model', `$${(Number(config.commission_per_contract || 0) * 2).toFixed(2)} round-trip cost`],
        ['Win Rate', pct(stats.win_rate)],
        ['Avg Trade', money2(stats.avg_trade_pnl)],
      ]);

      drawChart(data.equity_curve || []);
    }

    refresh().catch(err => {
      document.getElementById('statusText').textContent = err.message;
      document.getElementById('statusDot').className = 'dot bad';
    });
    setInterval(() => refresh().catch(console.error), 3000);
    window.addEventListener('resize', () => refresh().catch(console.error));
  </script>
</body>
</html>
"""


MULTI_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Multi-Futures Paper Monitor</title>
  <style>
    :root {
      --bg: #f5f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #657085;
      --line: #d8dee8;
      --good: #0b7a43;
      --warn: #ad5b00;
      --bad: #b42318;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { background: var(--panel); border-bottom: 1px solid var(--line); padding: 18px 24px; display: flex; justify-content: space-between; gap: 16px; position: sticky; top: 0; z-index: 3; }
    h1 { font-size: 20px; margin: 0; }
    h2 { font-size: 15px; margin: 0 0 12px; }
    main { max-width: 1500px; margin: 0 auto; padding: 20px; display: grid; gap: 16px; }
    .muted { color: var(--muted); }
    .status { display: flex; align-items: center; gap: 8px; white-space: nowrap; color: var(--muted); }
    .dot { width: 10px; height: 10px; border-radius: 50%; background: var(--warn); }
    .dot.ok { background: var(--good); }
    .dot.bad { background: var(--bad); }
    .metrics { display: grid; grid-template-columns: repeat(5, minmax(150px, 1fr)); gap: 12px; }
    .card, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
    .card { padding: 14px; min-height: 82px; }
    .label { color: var(--muted); font-size: 12px; margin-bottom: 8px; }
    .value { font-size: 22px; font-weight: 700; overflow-wrap: anywhere; }
    .panel { padding: 16px; overflow: auto; }
    table { width: 100%; border-collapse: collapse; table-layout: fixed; }
    th, td { padding: 8px 6px; border-bottom: 1px solid #edf0f4; text-align: right; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-size: 12px; font-weight: 600; }
    .good { color: var(--good); }
    .bad { color: var(--bad); }
    .warn { color: var(--warn); }
    @media (max-width: 900px) {
      header { flex-direction: column; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      table { min-width: 980px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Multi-Futures Paper Monitor</h1>
      <div class="muted" id="subtitle"></div>
    </div>
    <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">Loading</span></div>
  </header>

  <main>
    <section class="metrics">
      <div class="card"><div class="label">Sleeves</div><div class="value" id="sleeves">-</div></div>
      <div class="card"><div class="label">Combined Valuation</div><div class="value" id="valuation">-</div></div>
      <div class="card"><div class="label">Realized PnL</div><div class="value" id="realized">-</div></div>
      <div class="card"><div class="label">Trades</div><div class="value" id="trades">-</div></div>
      <div class="card"><div class="label">Win Rate</div><div class="value" id="winRate">-</div></div>
    </section>

    <section class="panel">
      <h2>Symbols</h2>
      <table>
        <thead>
          <tr>
            <th>Symbol</th><th>Last Bar</th><th>Status</th><th>Valuation</th><th>PnL</th><th>OR Size</th><th>Gap</th><th>Skip</th><th>Active</th><th>Trades</th><th>Win Rate</th>
          </tr>
        </thead>
        <tbody id="symbolsTable"></tbody>
      </table>
    </section>

    <section class="panel">
      <h2>Recent Trades</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Exit</th><th>Side</th><th>Entry</th><th>Exit Price</th><th>PnL</th><th>MAE</th><th>MFE</th></tr></thead>
        <tbody id="tradesTable"></tbody>
      </table>
    </section>
  </main>

  <script>
    const money = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', maximumFractionDigits: 0}) : '-';
    const money2 = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2}) : '-';
    const fixed = (v, d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '-';
    const pct = v => Number.isFinite(Number(v)) ? (Number(v) * 100).toFixed(1) + '%' : '-';
    const pnlClass = v => Number(v) >= 0 ? 'good' : 'bad';
    const set = (id, value) => document.getElementById(id).textContent = value;
    const rowHtml = cells => '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    function renderTable(id, rows, mapper, empty, cols) {
      document.getElementById(id).innerHTML = rows.length ? rows.map(mapper).join('') : `<tr><td colspan="${cols}" class="muted">${empty}</td></tr>`;
    }
    async function refresh() {
      const response = await fetch('/api/snapshot', {cache: 'no-store'});
      const data = await response.json();
      const stats = data.stats || {};
      const account = data.account || {};
      const engines = data.engines || {};
      const source = data.source || {};
      const symbols = data.symbols || Object.keys(engines);

      set('subtitle', `${symbols.join(', ')}; ${data.output_dir || ''}`);
      set('sleeves', account.status || String(symbols.length));
      set('valuation', money(account.valuation));
      set('realized', money2(stats.realized_pnl));
      set('trades', String(stats.trades ?? 0));
      set('winRate', pct(stats.win_rate));

      const dot = document.getElementById('statusDot');
      dot.className = 'dot ' + (source.last_error ? 'bad' : 'ok');
      set('statusText', `${source.mode || '-'}: ${source.message || '-'}${source.last_error ? '; ' + source.last_error : ''}`);

      renderTable('symbolsTable', symbols, symbol => {
        const snap = engines[symbol] || {};
        const session = snap.session || {};
        const s = snap.stats || {};
        const a = snap.account || {};
        const active = snap.active_trade && Object.keys(snap.active_trade).length ? `${snap.active_trade.side} ${snap.active_trade.quantity}` : '-';
        return rowHtml([
          symbol,
          snap.last_bar_time || '-',
          a.status || '-',
          money(a.valuation),
          `<span class="${pnlClass(s.realized_pnl)}">${money2(s.realized_pnl)}</span>`,
          fixed(session.opening_range_size),
          fixed(session.opening_gap_points),
          session.skip_reason || '-',
          active,
          String(s.trades ?? 0),
          pct(s.win_rate),
        ]);
      }, 'No symbol state yet.', 11);

      renderTable('tradesTable', (data.recent_trades || []).slice().reverse().slice(0, 20), row => rowHtml([
        row.contract_symbol,
        row.exit_reason,
        row.side,
        money2(row.entry_price),
        money2(row.exit_price),
        `<span class="${pnlClass(row.realized_pnl)}">${money2(row.realized_pnl)}</span>`,
        money2(row.mae_pnl),
        money2(row.mfe_pnl),
      ]), 'No closed trades yet.', 8);
    }
    refresh().catch(console.error);
    setInterval(() => refresh().catch(console.error), 3000);
  </script>
</body>
</html>
"""


def create_app(engine: NQPaperEngine | MultiSymbolPaperMonitor) -> Flask:
    app = Flask(__name__)
    html = MULTI_HTML if isinstance(engine, MultiSymbolPaperMonitor) else HTML

    @app.get("/")
    def index() -> Response:
        return Response(html, mimetype="text/html")

    @app.get("/api/snapshot")
    def api_snapshot() -> Any:
        return jsonify(engine.snapshot())

    @app.post("/api/bar")
    def api_bar() -> Any:
        payload = request.get_json(force=True)
        default_symbol = "NQ"
        if isinstance(engine, NQPaperEngine):
            default_symbol = engine.config.data_symbol
        bar = parse_bar_payload(payload, symbol=default_symbol)
        return jsonify(engine.process_bar(bar))

    @app.post("/api/reset")
    def api_reset() -> Any:
        return jsonify(engine.reset())

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "ts": utc_now(), "snapshot": engine.snapshot()})

    return app


def config_to_dict(config: OpeningRangeBreakoutConfig) -> dict[str, Any]:
    return {
        "strategy_family": config.strategy_family,
        "account_tier": config.account_tier,
        "data_symbol": config.data_symbol,
        "contract_symbol": config.contract_symbol,
        "quantity": config.quantity,
        "opening_range_minutes": config.opening_range_minutes,
        "breakout_buffer_points": str(config.breakout_buffer_points),
        "stop_points": str(config.stop_points),
        "target_points": str(config.target_points),
        "max_hold_minutes": config.max_hold_minutes,
        "max_trades_per_session": config.max_trades_per_session,
        "last_entry_time": config.last_entry_time.strftime("%H:%M"),
        "force_exit_time": config.force_exit_time.strftime("%H:%M"),
        "commission_per_contract": str(config.commission_per_contract),
        "max_opening_range_points": none_or_str(config.max_opening_range_points),
        "max_opening_gap_points": none_or_str(config.max_opening_gap_points),
    }


def trade_to_dict(trade: TradeRecord) -> dict[str, Any]:
    row = trade.to_row()
    for key in (
        "entry_price",
        "exit_price",
        "realized_pnl",
        "mae_points",
        "mfe_points",
        "mae_pnl",
        "mfe_pnl",
        "entry_mll_buffer",
        "min_mll_buffer",
    ):
        row[key] = decimal_to_float(Decimal(str(row[key])))
    return row


def decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def none_or_str(value: Optional[Decimal]) -> Optional[str]:
    return None if value is None else str(value)


def safe_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def safe_money_ratio(numerator: Decimal, denominator: int) -> Decimal:
    if denominator == 0:
        return money(0)
    return money(numerator / Decimal(denominator))


def round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    ticks = (Decimal(value) / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return ticks * tick


def time_to_target_minutes(
    trade: ActiveTrade,
    timestamp: datetime,
    exit_reason: str,
) -> Optional[int]:
    if exit_reason != "target":
        return None
    return max(0, int((timestamp - trade.entry_time).total_seconds() // 60))


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str):
    return datetime.strptime(value, "%H:%M").time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--source",
        choices=["manual", "replay", "csv-tail", "yahoo-chart", "yfinance"],
        default="manual",
    )
    parser.add_argument("--symbol-set", choices=["single", "six", "eight"], default="single")
    parser.add_argument("--symbols", help="Comma-separated custom monitor symbols, e.g. NQ,ES,RTY")
    parser.add_argument("--data", default="/Users/davidgasper/Downloads/Dataset_NQ_1min_2022_2025.csv")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--replay-delay", type=float, default=0.0)
    parser.add_argument("--yfinance-symbol", default="NQ=F")
    parser.add_argument("--yahoo-symbol", default="NQ=F")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--csv-tail-from-start", action="store_true")
    parser.add_argument("--contract", choices=sorted(MARKET_DEFAULTS), default="NQ")
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--stop-points")
    parser.add_argument("--target-points")
    parser.add_argument("--breakout-buffer-points")
    parser.add_argument("--opening-range-minutes", type=int, default=5)
    parser.add_argument("--max-opening-range-points")
    parser.add_argument("--max-opening-gap-points")
    parser.add_argument("--last-entry-time", default="11:30")
    parser.add_argument(
        "--commission-per-contract",
        help="Optional per-side cost override. Omit to use Topstep published product costs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> OpeningRangeBreakoutConfig:
    base = default_strategy_config(args.contract)
    return OpeningRangeBreakoutConfig(
        strategy_family="orb_fade",
        account_tier="50K",
        data_symbol=args.contract,
        contract_symbol=args.contract,
        quantity=args.quantity,
        opening_range_minutes=args.opening_range_minutes,
        breakout_buffer_points=Decimal(args.breakout_buffer_points)
        if args.breakout_buffer_points
        else base.breakout_buffer_points,
        stop_points=Decimal(args.stop_points) if args.stop_points else base.stop_points,
        target_points=Decimal(args.target_points)
        if args.target_points
        else base.target_points,
        max_hold_minutes=180,
        max_trades_per_session=1,
        last_entry_time=parse_time(args.last_entry_time),
        force_exit_time=parse_time("16:00"),
        commission_per_contract=Decimal(args.commission_per_contract)
        if args.commission_per_contract
        else topstep_commission_per_side(args.contract),
        max_opening_range_points=Decimal(args.max_opening_range_points)
        if args.max_opening_range_points
        else base.max_opening_range_points,
        max_opening_gap_points=Decimal(args.max_opening_gap_points)
        if args.max_opening_gap_points
        else base.max_opening_gap_points,
    )


def resolve_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols:
        return tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    if args.symbol_set == "six":
        return SIX_SYMBOLS
    if args.symbol_set == "eight":
        return EIGHT_SYMBOLS
    return (args.contract.upper(),)


def maybe_start_source_thread(
    engine: NQPaperEngine | MultiSymbolPaperMonitor,
    args: argparse.Namespace,
) -> None:
    if args.source == "manual":
        engine.set_source_status("manual", "waiting for POST /api/bar or manual reset", running=False)
        return
    if isinstance(engine, MultiSymbolPaperMonitor):
        if args.source == "yahoo-chart":
            thread = threading.Thread(
                target=run_multi_yahoo_chart_poll,
                args=(engine, args.poll_seconds),
                daemon=True,
            )
            thread.start()
            return
        engine.set_source_status(
            args.source,
            f"{args.source} is only wired for single-symbol mode; use --source yahoo-chart for multi-symbol mode",
            running=False,
            last_error="unsupported_multi_symbol_source",
        )
        return
    if args.source == "replay":
        thread = threading.Thread(
            target=run_csv_replay,
            args=(
                engine,
                args.data,
                parse_timestamp(args.start) if args.start else None,
                parse_timestamp(args.end) if args.end else None,
                args.replay_delay,
            ),
            daemon=True,
        )
        thread.start()
        return
    if args.source == "csv-tail":
        thread = threading.Thread(
            target=run_csv_tail,
            args=(engine, args.data, args.poll_seconds, args.csv_tail_from_start),
            daemon=True,
        )
        thread.start()
        return
    if args.source == "yahoo-chart":
        thread = threading.Thread(
            target=run_yahoo_chart_poll,
            args=(engine, args.yahoo_symbol, args.poll_seconds),
            daemon=True,
        )
        thread.start()
        return
    if args.source == "yfinance":
        thread = threading.Thread(
            target=run_yfinance_poll,
            args=(engine, args.yfinance_symbol, args.poll_seconds),
            daemon=True,
        )
        thread.start()


def main() -> None:
    args = parse_args()
    symbols = resolve_symbols(args)
    if len(symbols) == 1:
        args.contract = symbols[0]
        engine: NQPaperEngine | MultiSymbolPaperMonitor = NQPaperEngine(
            config=build_config(args),
            output_dir=args.output_dir,
        )
    else:
        engine = MultiSymbolPaperMonitor(symbols=symbols, output_dir=args.output_dir)
    maybe_start_source_thread(engine, args)
    app = create_app(engine)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
