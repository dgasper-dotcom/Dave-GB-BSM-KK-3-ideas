#!/usr/bin/env python3
"""Live paper trade Topstep-style challenge accounts.

The default mode runs each monitored symbol as a separate paper portfolio.
Shared-account mode is still available for account-level basket experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time as time_module
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

from flask import Flask, Response, jsonify, request

from topstep_data_loader import FuturesBar, parse_decimal, parse_timestamp
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
    session_date_for_timestamp,
)
from topstep_nq_paper_monitor import (
    EIGHT_SYMBOLS,
    MARKET_DEFAULTS,
    SIX_SYMBOLS,
    SourceStatus,
    default_strategy_config,
    fetch_yahoo_chart_bars,
    parse_time,
    round_to_tick,
    safe_money_ratio,
    safe_ratio,
    utc_now,
)
from topstep_rule_simulator import (
    ChallengeStatus,
    RuleEvent,
    RuleEventType,
    TopstepRuleSimulator,
    money,
)


DEFAULT_OUTPUT_DIR = "topstep_prop_challenge_paper"


@dataclass
class SymbolRuntime:
    config: OpeningRangeBreakoutConfig
    session_state: Optional[SessionState] = None
    previous_bar: Optional[FuturesBar] = None
    previous_session_close: Optional[Decimal] = None
    rth_vwap_numerator: Decimal = Decimal("0")
    rth_vwap_denominator: Decimal = Decimal("0")
    rth_vwap: Optional[Decimal] = None
    bars_processed: int = 0
    duplicate_or_stale_bars: int = 0
    last_bar_time: Optional[datetime] = None
    risk_filter_skips: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.risk_filter_skips is None:
            self.risk_filter_skips = {}


@dataclass
class SharedActiveTrade:
    symbol: str
    trade: ActiveTrade


class PropChallengePaperEngine:
    def __init__(
        self,
        symbols: Iterable[str] = SIX_SYMBOLS,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        account_tier: str = "50K",
        max_concurrent_positions: int = 1,
        commission_per_contract: Optional[Decimal] = None,
        strategy_family: str = "orb_fade",
        scalp_reward_risk_ratio: Decimal = Decimal("0.6"),
        quantity: Optional[int] = None,
        stop_points: Optional[Decimal] = None,
        target_points: Optional[Decimal] = None,
        breakout_buffer_points: Optional[Decimal] = None,
        max_opening_range_points: Optional[Decimal] = None,
        max_opening_gap_points: Optional[Decimal] = None,
        slippage_ticks_per_side: Decimal = Decimal("0"),
        disable_session_filters: bool = False,
        max_trades_per_session: Optional[int] = None,
        max_hold_minutes: Optional[int] = None,
        last_entry_time: Optional[time] = None,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        unknown = [symbol for symbol in self.symbols if symbol not in MARKET_DEFAULTS]
        if unknown:
            known = ", ".join(sorted(MARKET_DEFAULTS))
            raise ValueError(f"unsupported symbols {unknown}; expected one of {known}")
        if max_concurrent_positions <= 0:
            raise ValueError("max_concurrent_positions must be positive")

        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.account_tier = account_tier
        self.max_concurrent_positions = max_concurrent_positions
        self.commission_per_contract = (
            money(commission_per_contract)
            if commission_per_contract is not None
            else None
        )
        self.strategy_family = strategy_family.lower()
        self.scalp_reward_risk_ratio = Decimal(str(scalp_reward_risk_ratio))
        self.quantity_override = quantity
        self.stop_points_override = Decimal(str(stop_points)) if stop_points is not None else None
        self.target_points_override = Decimal(str(target_points)) if target_points is not None else None
        self.breakout_buffer_points_override = (
            Decimal(str(breakout_buffer_points))
            if breakout_buffer_points is not None
            else None
        )
        self.max_opening_range_points_override = (
            Decimal(str(max_opening_range_points))
            if max_opening_range_points is not None
            else None
        )
        self.max_opening_gap_points_override = (
            Decimal(str(max_opening_gap_points))
            if max_opening_gap_points is not None
            else None
        )
        self.slippage_ticks_per_side = Decimal(str(slippage_ticks_per_side))
        self.disable_session_filters = disable_session_filters
        self.max_trades_per_session_override = max_trades_per_session
        self.max_hold_minutes_override = max_hold_minutes
        self.last_entry_time_override = last_entry_time
        self._lock = threading.RLock()
        self._source_status = SourceStatus(updated_at=utc_now())
        self._reset_unlocked()

    def reset(self) -> dict[str, Any]:
        with self._lock:
            self._reset_unlocked()
            self._record_event("reset", "shared challenge account reset", None)
            self._persist_state_unlocked()
            return self.snapshot_unlocked()

    def _reset_unlocked(self) -> None:
        self.contract_specs = {
            symbol: ContractSpec(
                spec.symbol,
                tick_size=spec.tick_size,
                tick_value=spec.tick_value,
                commission_per_contract=self._commission_for_symbol(symbol),
                is_micro=spec.is_micro,
            )
            for symbol, spec in DEFAULT_CONTRACT_SPECS.items()
        }
        self.sim = TopstepRuleSimulator.from_tier(self.account_tier)
        self.adapter = FuturesExecutionAdapter(self.sim, contract_specs=self.contract_specs)
        self.runtimes = {
            symbol: SymbolRuntime(
                config=self._config_for_symbol(symbol),
            )
            for symbol in self.symbols
        }
        self.current_session_date = None
        self.active_trade: Optional[SharedActiveTrade] = None
        self.last_closed_balance = self.sim.closed_balance
        self.last_rule_event_type = ""
        self.last_rule_event_detail = ""
        self.last_exit_reason = ""
        self.trades: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.equity_points: list[dict[str, Any]] = []
        self.bars_processed = 0
        self.skipped_entry_conflicts = 0

    def _config_for_symbol(self, symbol: str) -> OpeningRangeBreakoutConfig:
        base = default_strategy_config(symbol)
        stop_points = self.stop_points_override or base.stop_points
        target_points = self.target_points_override or base.target_points
        breakout_buffer_points = (
            self.breakout_buffer_points_override
            if self.breakout_buffer_points_override is not None
            else base.breakout_buffer_points
        )
        max_trades_per_session = base.max_trades_per_session
        max_hold_minutes = base.max_hold_minutes
        if self.strategy_family in {"scalp_reversion", "vwap_reversion"}:
            if self.target_points_override is None:
                target_points = self._scaled_target_points(symbol, stop_points)
            max_trades_per_session = self.max_trades_per_session_override or 3
            max_hold_minutes = self.max_hold_minutes_override or 10
        return OpeningRangeBreakoutConfig(
            strategy_family=self.strategy_family,
            account_tier=self.account_tier,
            data_symbol=symbol,
            contract_symbol=symbol,
            quantity=self.quantity_override or base.quantity,
            opening_range_minutes=base.opening_range_minutes,
            breakout_buffer_points=breakout_buffer_points,
            stop_points=stop_points,
            target_points=target_points,
            max_hold_minutes=max_hold_minutes,
            max_trades_per_session=max_trades_per_session,
            rth_start=base.rth_start,
            last_entry_time=self.last_entry_time_override or base.last_entry_time,
            force_exit_time=base.force_exit_time,
            commission_per_contract=self._commission_for_symbol(symbol),
            max_opening_range_points=self._session_filter_value(
                self.max_opening_range_points_override,
                base.max_opening_range_points,
            ),
            max_opening_gap_points=self._session_filter_value(
                self.max_opening_gap_points_override,
                base.max_opening_gap_points,
            ),
        )

    def _session_filter_value(
        self,
        override: Optional[Decimal],
        default: Optional[Decimal],
    ) -> Optional[Decimal]:
        if self.disable_session_filters:
            return None
        return override if override is not None else default

    def _scaled_target_points(self, symbol: str, stop_points: Decimal) -> Decimal:
        spec = DEFAULT_CONTRACT_SPECS[symbol]
        raw_target = stop_points * self.scalp_reward_risk_ratio
        target = round_to_tick(raw_target, spec.tick_size)
        return max(spec.tick_size, target)

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
            accepted = self._process_bar_unlocked(bar, allow_entries=allow_entries)
            if accepted:
                self._persist_state_unlocked()
                return {"accepted": True, "snapshot": self.snapshot_unlocked()}
            return {"accepted": False, "reason": "stale_or_unmonitored_bar"}

    def process_bars(self, bars: Iterable[tuple[FuturesBar, bool]]) -> dict[str, int]:
        accepted = 0
        stale = 0
        tradable = 0
        sorted_bars = sorted(
            bars,
            key=lambda item: (
                item[0].timestamp,
                self.symbols.index(item[0].symbol)
                if item[0].symbol in self.symbols
                else len(self.symbols),
            ),
        )
        with self._lock:
            for bar, allow_entries in sorted_bars:
                if self._process_bar_unlocked(bar, allow_entries=allow_entries):
                    accepted += 1
                    if allow_entries:
                        tradable += 1
                else:
                    stale += 1
            if accepted:
                self._persist_state_unlocked()
        return {"accepted": accepted, "stale": stale, "tradable": tradable}

    def _process_bar_unlocked(self, bar: FuturesBar, allow_entries: bool = True) -> bool:
        symbol = bar.symbol.upper()
        runtime = self.runtimes.get(symbol)
        if runtime is None:
            return False
        if runtime.last_bar_time is not None and bar.timestamp <= runtime.last_bar_time:
            runtime.duplicate_or_stale_bars += 1
            return False

        runtime.bars_processed += 1
        runtime.last_bar_time = bar.timestamp
        self.bars_processed += 1

        bar_session_date = session_date_for_timestamp(bar.timestamp)
        if not allow_entries and self.active_trade is None:
            self._process_warmup_bar_unlocked(runtime, bar, bar_session_date)
            return True

        if self.current_session_date is None:
            self.current_session_date = bar_session_date
        elif bar_session_date != self.current_session_date:
            self._roll_global_session_unlocked(bar_session_date)

        self._ensure_symbol_session(runtime, bar_session_date)
        self._update_rth_vwap_unlocked(runtime, bar)

        if self.sim.status != ChallengeStatus.ACTIVE:
            runtime.previous_bar = bar
            self._append_equity_point(bar)
            return True

        if self.active_trade is not None and self.active_trade.symbol == symbol:
            terminal_now = self._manage_open_trade_unlocked(symbol, runtime, bar)
            runtime.previous_bar = bar
            self._append_equity_point(bar)
            if terminal_now:
                return True

        if self.active_trade is None:
            self._update_opening_range_unlocked(runtime, bar)
            if allow_entries and self.sim.status == ChallengeStatus.ACTIVE and not self.sim.day_locked:
                self._maybe_enter_unlocked(symbol, runtime, bar)
        else:
            self._update_opening_range_unlocked(runtime, bar)
            if allow_entries and self._entry_side(runtime, bar) is not None:
                self.skipped_entry_conflicts += 1

        runtime.previous_bar = bar
        self._append_equity_point(bar)
        return True

    def _process_warmup_bar_unlocked(
        self,
        runtime: SymbolRuntime,
        bar: FuturesBar,
        bar_session_date,
    ) -> None:
        self._ensure_symbol_session(runtime, bar_session_date)
        self._update_rth_vwap_unlocked(runtime, bar)
        self._update_opening_range_unlocked(runtime, bar)
        runtime.previous_bar = bar

    def _roll_global_session_unlocked(self, new_session_date) -> None:
        if self.active_trade is not None:
            active_runtime = self.runtimes[self.active_trade.symbol]
            previous_bar = active_runtime.previous_bar
            if previous_bar is not None:
                self._force_exit_unlocked(
                    self.active_trade.symbol,
                    previous_bar.timestamp,
                    previous_bar.close,
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
                self._record_event("session_error", str(exc), None)

        self.current_session_date = new_session_date

    def _ensure_symbol_session(self, runtime: SymbolRuntime, session_date) -> None:
        if (
            runtime.session_state is not None
            and runtime.session_state.session_date == session_date
        ):
            return
        if runtime.previous_bar is not None:
            runtime.previous_session_close = runtime.previous_bar.close
        rth_start_dt = datetime.combine(session_date, runtime.config.rth_start)
        runtime.session_state = SessionState(
            session_date=session_date,
            opening_range_end=rth_start_dt
            + timedelta(minutes=runtime.config.opening_range_minutes),
            previous_session_close=runtime.previous_session_close,
        )
        runtime.rth_vwap_numerator = Decimal("0")
        runtime.rth_vwap_denominator = Decimal("0")
        runtime.rth_vwap = None

    def _update_rth_vwap_unlocked(self, runtime: SymbolRuntime, bar: FuturesBar) -> None:
        if bar.timestamp.time() < runtime.config.rth_start:
            return
        volume = bar.volume if bar.volume > 0 else Decimal("1")
        typical_price = (bar.high + bar.low + bar.close) / Decimal("3")
        runtime.rth_vwap_numerator += typical_price * volume
        runtime.rth_vwap_denominator += volume
        if runtime.rth_vwap_denominator > 0:
            runtime.rth_vwap = runtime.rth_vwap_numerator / runtime.rth_vwap_denominator

    def _update_opening_range_unlocked(self, runtime: SymbolRuntime, bar: FuturesBar) -> None:
        session_state = runtime.session_state
        if session_state is None:
            return
        if (
            session_state.opening_gap_points is None
            and session_state.previous_session_close is not None
        ):
            session_state.opening_gap_points = abs(bar.open - session_state.previous_session_close)

        if not (
            bar.timestamp.time() >= runtime.config.rth_start
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

    def _maybe_enter_unlocked(self, symbol: str, runtime: SymbolRuntime, bar: FuturesBar) -> None:
        session_state = runtime.session_state
        if session_state is None:
            return
        if session_state.trades_taken >= runtime.config.max_trades_per_session:
            return
        if not (
            session_state.opening_range_end <= bar.timestamp
            and bar.timestamp.time() <= runtime.config.last_entry_time
        ):
            return

        side = self._entry_side(runtime, bar)
        if side is None:
            return

        skip_reason = self._session_filter_skip_reason(runtime)
        if skip_reason is not None:
            self._record_risk_filter_skip(symbol, runtime, skip_reason)
            return

        if self.active_trade is not None:
            self.skipped_entry_conflicts += 1
            return

        entry_price = self._apply_entry_slippage(symbol, bar.close, side)
        stop_price, target_price = self._stop_target(runtime.config, entry_price, side)
        quantity = runtime.config.quantity

        try:
            event = self.adapter.process_fill(
                Fill(
                    symbol,
                    side,
                    quantity,
                    entry_price,
                    timestamp=bar.timestamp.isoformat(sep=" "),
                )
            ).rule_event
        except ExecutionError as exc:
            self._record_event("entry_rejected", f"{symbol}: {exc}", bar)
            return

        self._remember_rule_event(event, "entry")
        session_state.trades_taken += 1
        entry_mll_buffer = money(self.sim.valuation - self.sim.active_mll)
        self.active_trade = SharedActiveTrade(
            symbol=symbol,
            trade=ActiveTrade(
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
            ),
        )
        self._record_event("entry", f"{symbol} {side.value.upper()} {quantity} @ {entry_price}", bar)

    def _entry_side(self, runtime: SymbolRuntime, bar: FuturesBar) -> Optional[FillSide]:
        session_state = runtime.session_state
        if runtime.config.strategy_family in {"scalp_reversion", "vwap_reversion"}:
            if runtime.rth_vwap is None:
                return None
            distance = bar.close - runtime.rth_vwap
            if distance >= runtime.config.breakout_buffer_points:
                return FillSide.SELL
            if distance <= -runtime.config.breakout_buffer_points:
                return FillSide.BUY
            return None

        if session_state is None or not session_state.opening_range_ready:
            return None
        assert session_state.opening_high is not None
        assert session_state.opening_low is not None
        long_trigger = session_state.opening_high + runtime.config.breakout_buffer_points
        short_trigger = session_state.opening_low - runtime.config.breakout_buffer_points
        if bar.close > long_trigger:
            return FillSide.SELL
        if bar.close < short_trigger:
            return FillSide.BUY
        return None

    def _session_filter_skip_reason(self, runtime: SymbolRuntime) -> Optional[str]:
        session_state = runtime.session_state
        if session_state is None:
            return None
        if runtime.config.max_opening_range_points is not None:
            opening_range_size = session_state.opening_range_size
            if (
                opening_range_size is not None
                and opening_range_size > runtime.config.max_opening_range_points
            ):
                return "opening_range"
        if (
            runtime.config.max_opening_gap_points is not None
            and session_state.opening_gap_points is not None
            and session_state.opening_gap_points > runtime.config.max_opening_gap_points
        ):
            return "opening_gap"
        return None

    def _record_risk_filter_skip(
        self,
        symbol: str,
        runtime: SymbolRuntime,
        reason: str,
    ) -> None:
        session_state = runtime.session_state
        if session_state is None or reason in session_state.recorded_skip_reasons:
            return
        session_state.recorded_skip_reasons.add(reason)
        assert runtime.risk_filter_skips is not None
        runtime.risk_filter_skips[reason] = runtime.risk_filter_skips.get(reason, 0) + 1
        self._record_event("skip", f"{symbol}: session skipped by {reason} filter", None)

    def _manage_open_trade_unlocked(
        self,
        symbol: str,
        runtime: SymbolRuntime,
        bar: FuturesBar,
    ) -> bool:
        assert self.active_trade is not None
        active = self.active_trade.trade
        forced_time_exit = (
            bar.timestamp.time() >= runtime.config.force_exit_time
            or int((bar.timestamp - active.entry_time).total_seconds() // 60)
            >= runtime.config.max_hold_minutes
        )
        if forced_time_exit:
            self._update_trade_excursions(symbol, runtime, active, self._trade_bar_path(active, bar))
            self._force_exit_unlocked(symbol, bar.timestamp, bar.close, "time_exit")
            return self.sim.status != ChallengeStatus.ACTIVE

        exit_price = None
        exit_reason = None
        if active.side == FillSide.BUY:
            if bar.low <= active.stop_price:
                exit_price = active.stop_price
                exit_reason = "stop"
            elif bar.high >= active.target_price:
                exit_price = active.target_price
                exit_reason = "target"
        else:
            if bar.high >= active.stop_price:
                exit_price = active.stop_price
                exit_reason = "stop"
            elif bar.low <= active.target_price:
                exit_price = active.target_price
                exit_reason = "target"

        if exit_price is not None and exit_reason is not None:
            self._update_trade_excursions(
                symbol,
                runtime,
                active,
                self._trade_bar_path_until_exit(active, bar, exit_price, exit_reason),
            )
            self._force_exit_unlocked(symbol, bar.timestamp, exit_price, exit_reason)
            return self.sim.status != ChallengeStatus.ACTIVE

        terminal_price = self._mark_bar_path_until_terminal(symbol, runtime, active, bar)
        if terminal_price is not None:
            self._record_forced_rule_exit_unlocked(symbol, bar.timestamp, terminal_price)
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

    def _update_trade_excursions(
        self,
        symbol: str,
        runtime: SymbolRuntime,
        trade: ActiveTrade,
        price_path: Iterable[Decimal],
    ) -> None:
        spec = self._contract_spec(symbol)
        underwater = False
        for mark in price_path:
            favorable_points = (mark - trade.entry_price) * Decimal(trade.side.sign)
            open_pnl = money(
                favorable_points * spec.multiplier * Decimal(trade.quantity)
                - runtime.config.commission_per_contract * Decimal(trade.quantity)
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

    def _mark_bar_path_until_terminal(
        self,
        symbol: str,
        runtime: SymbolRuntime,
        trade: ActiveTrade,
        bar: FuturesBar,
    ) -> Optional[Decimal]:
        observed_path = []
        for mark in self._trade_bar_path(trade, bar):
            observed_path.append(mark)
            event = self.adapter.mark_price(symbol, mark).rule_event
            self._remember_rule_event(event, "mark")
            if event.event_type in {RuleEventType.DLL_BREACH, RuleEventType.MLL_BREACH}:
                self._update_trade_excursions(symbol, runtime, trade, tuple(observed_path))
                return mark
        self._update_trade_excursions(symbol, runtime, trade, tuple(observed_path))
        return None

    def _force_exit_unlocked(
        self,
        symbol: str,
        timestamp: datetime,
        exit_price: Decimal,
        exit_reason: str,
    ) -> None:
        if self.active_trade is None or self.sim.status != ChallengeStatus.ACTIVE:
            self.active_trade = None
            return
        trade = self.active_trade.trade
        exit_side = FillSide.SELL if trade.side == FillSide.BUY else FillSide.BUY
        fill_price = (
            self._apply_exit_slippage(symbol, exit_price, trade.side)
            if exit_reason in {"time_exit", "session_end", "data_end"}
            else round_to_tick(exit_price, self._contract_spec(symbol).tick_size)
        )
        event = self.adapter.process_fill(
            Fill(
                symbol,
                exit_side,
                trade.quantity,
                fill_price,
                timestamp=timestamp.isoformat(sep=" "),
            )
        ).rule_event
        self._remember_rule_event(event, exit_reason)
        realized = money(self.sim.closed_balance - self.last_closed_balance)
        self.last_closed_balance = self.sim.closed_balance
        row = self._closed_trade_row(symbol, trade, timestamp, fill_price, realized, exit_reason, event)
        self.trades.append(row)
        self._append_csv_row(self.output_dir / "trades.csv", row)
        self._record_event("exit", f"{symbol}: {exit_reason} @ {row['exit_price']}; pnl {realized}", None)
        self.active_trade = None

    def _record_forced_rule_exit_unlocked(
        self,
        symbol: str,
        timestamp: datetime,
        exit_price: Decimal,
    ) -> None:
        if self.active_trade is None:
            return
        trade = self.active_trade.trade
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
        row = self._closed_trade_row(symbol, trade, timestamp, exit_price, realized, exit_reason, event)
        self.trades.append(row)
        self._append_csv_row(self.output_dir / "trades.csv", row)
        self._record_event("rule_exit", f"{symbol}: {event.event_type.value} @ {exit_price}; pnl {realized}", None)
        self.active_trade = None

    def _closed_trade_row(
        self,
        symbol: str,
        trade: ActiveTrade,
        timestamp: datetime,
        exit_price: Decimal,
        realized: Decimal,
        exit_reason: str,
        event: RuleEvent,
    ) -> dict[str, Any]:
        return {
            "symbol": symbol,
            "session_date": trade.session_date.isoformat(),
            "risk_state": trade.risk_state,
            "side": trade.side.value,
            "quantity": trade.quantity,
            "entry_time": trade.entry_time.isoformat(sep=" "),
            "exit_time": timestamp.isoformat(sep=" "),
            "entry_price": str(trade.entry_price),
            "exit_price": str(round_to_tick(exit_price, self._contract_spec(symbol).tick_size)),
            "realized_pnl": str(money(realized)),
            "exit_reason": exit_reason,
            "rule_event": event.event_type.value,
            "mae_points": str(trade.mae_points),
            "mfe_points": str(trade.mfe_points),
            "mae_pnl": str(trade.mae_pnl),
            "mfe_pnl": str(trade.mfe_pnl),
            "time_to_target_minutes": ""
            if exit_reason != "target"
            else max(0, int((timestamp - trade.entry_time).total_seconds() // 60)),
            "time_underwater_minutes": trade.time_underwater_minutes,
            "bars_held": trade.bars_held,
            "entry_mll_buffer": str(trade.entry_mll_buffer),
            "min_mll_buffer": str(trade.min_mll_buffer),
            "account_closed_balance": str(self.sim.closed_balance),
            "account_valuation": str(self.sim.valuation),
            "active_mll": str(self.sim.active_mll),
        }

    def _stop_target(
        self,
        config: OpeningRangeBreakoutConfig,
        entry_price: Decimal,
        side: FillSide,
    ) -> tuple[Decimal, Decimal]:
        if side == FillSide.BUY:
            return entry_price - config.stop_points, entry_price + config.target_points
        return entry_price + config.stop_points, entry_price - config.target_points

    def _slippage_points(self, symbol: str) -> Decimal:
        return self._contract_spec(symbol).tick_size * self.slippage_ticks_per_side

    def _apply_entry_slippage(
        self,
        symbol: str,
        price: Decimal,
        side: FillSide,
    ) -> Decimal:
        tick_size = self._contract_spec(symbol).tick_size
        slipped = price + self._slippage_points(symbol) * Decimal(side.sign)
        return round_to_tick(slipped, tick_size)

    def _apply_exit_slippage(
        self,
        symbol: str,
        price: Decimal,
        entry_side: FillSide,
    ) -> Decimal:
        tick_size = self._contract_spec(symbol).tick_size
        slipped = price - self._slippage_points(symbol) * Decimal(entry_side.sign)
        return round_to_tick(slipped, tick_size)

    def _contract_spec(self, symbol: str) -> ContractSpec:
        return self.contract_specs[symbol]

    def _remember_rule_event(self, event: RuleEvent, exit_reason: str) -> None:
        self.last_rule_event_type = event.event_type.value
        self.last_rule_event_detail = event.detail
        self.last_exit_reason = exit_reason

    def _append_equity_point(self, bar: FuturesBar) -> None:
        self.equity_points.append(
            {
                "ts": bar.timestamp.isoformat(sep=" "),
                "symbol": bar.symbol,
                "closed_balance": decimal_to_float(self.sim.closed_balance),
                "open_pnl": decimal_to_float(self.sim.open_pnl),
                "valuation": decimal_to_float(self.sim.valuation),
                "active_mll": decimal_to_float(self.sim.active_mll),
                "dll_floor": decimal_to_float(self.sim.active_dll_threshold),
            }
        )
        if len(self.equity_points) > 3000:
            self.equity_points = self.equity_points[-3000:]

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

    def _append_csv_row(self, path: Path, row: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _persist_state_unlocked(self) -> None:
        (self.output_dir / "state.json").write_text(
            json.dumps(self.snapshot_unlocked(), indent=2) + "\n",
            encoding="utf-8",
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self.snapshot_unlocked()

    def snapshot_unlocked(self) -> dict[str, Any]:
        snap = self.sim.snapshot()
        return {
            "ts": utc_now(),
            "mode": "shared_prop_challenge",
            "symbols": list(self.symbols),
            "output_dir": str(self.output_dir),
            "source": self._source_status.to_dict(),
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
            "active_trade": self._active_trade_snapshot(),
            "stats": self._stats_snapshot(),
            "symbols_state": {
                symbol: self._symbol_snapshot(symbol, runtime)
                for symbol, runtime in self.runtimes.items()
            },
            "bars_processed": self.bars_processed,
            "skipped_entry_conflicts": self.skipped_entry_conflicts,
            "recent_trades": self.trades[-50:],
            "recent_events": self.events[-100:],
            "equity_curve": self.equity_points,
        }

    def _active_trade_snapshot(self) -> dict[str, Any]:
        if self.active_trade is None:
            return {}
        trade = self.active_trade.trade
        return {
            "symbol": self.active_trade.symbol,
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
        }

    def _symbol_snapshot(self, symbol: str, runtime: SymbolRuntime) -> dict[str, Any]:
        session_state = runtime.session_state
        opening_high = session_state.opening_high if session_state else None
        opening_low = session_state.opening_low if session_state else None
        return {
            "last_bar_time": runtime.last_bar_time.isoformat(sep=" ")
            if runtime.last_bar_time
            else "",
            "bars_processed": runtime.bars_processed,
            "duplicate_or_stale_bars": runtime.duplicate_or_stale_bars,
            "opening_high": decimal_to_float(opening_high),
            "opening_low": decimal_to_float(opening_low),
            "opening_range_size": decimal_to_float(
                session_state.opening_range_size if session_state else None
            ),
            "opening_gap_points": decimal_to_float(
                session_state.opening_gap_points if session_state else None
            ),
            "skip_reason": self._session_filter_skip_reason(runtime) or "",
            "trades_taken_today": session_state.trades_taken if session_state else 0,
            "risk_filter_skips": dict(runtime.risk_filter_skips or {}),
            "rth_vwap": decimal_to_float(runtime.rth_vwap),
            "config": {
                "strategy_family": runtime.config.strategy_family,
                "stop_points": str(runtime.config.stop_points),
                "target_points": str(runtime.config.target_points),
                "breakout_buffer_points": str(runtime.config.breakout_buffer_points),
                "max_trades_per_session": runtime.config.max_trades_per_session,
                "max_hold_minutes": runtime.config.max_hold_minutes,
                "last_entry_time": runtime.config.last_entry_time.strftime("%H:%M"),
                "max_opening_range_points": str(runtime.config.max_opening_range_points),
                "max_opening_gap_points": str(runtime.config.max_opening_gap_points),
                "slippage_ticks_per_side": str(self.slippage_ticks_per_side),
                "round_turn_cost": str(self._commission_for_symbol(symbol) * Decimal("2")),
            },
        }

    def _stats_snapshot(self) -> dict[str, Any]:
        wins = [t for t in self.trades if Decimal(str(t["realized_pnl"])) > 0]
        losses = [t for t in self.trades if Decimal(str(t["realized_pnl"])) < 0]
        total = sum((Decimal(str(t["realized_pnl"])) for t in self.trades), Decimal("0"))
        return {
            "trades": len(self.trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": decimal_to_float(safe_ratio(len(wins), len(self.trades))),
            "realized_pnl": decimal_to_float(money(total)),
            "avg_trade_pnl": decimal_to_float(safe_money_ratio(total, len(self.trades))),
            "targets": sum(1 for t in self.trades if t["exit_reason"] == "target"),
            "stops": sum(1 for t in self.trades if t["exit_reason"] == "stop"),
            "rule_exits": sum(
                1
                for t in self.trades
                if t["rule_event"] in {RuleEventType.DLL_BREACH.value, RuleEventType.MLL_BREACH.value}
            ),
        }

    def _commission_for_symbol(self, symbol: str) -> Decimal:
        if self.commission_per_contract is not None:
            return self.commission_per_contract
        return topstep_commission_per_side(symbol)


class SeparatePortfolioPaperMonitor:
    def __init__(
        self,
        symbols: Iterable[str] = SIX_SYMBOLS,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        account_tier: str = "50K",
        commission_per_contract: Optional[Decimal] = None,
        strategy_family: str = "orb_fade",
        scalp_reward_risk_ratio: Decimal = Decimal("0.6"),
        quantity: Optional[int] = None,
        stop_points: Optional[Decimal] = None,
        target_points: Optional[Decimal] = None,
        breakout_buffer_points: Optional[Decimal] = None,
        max_opening_range_points: Optional[Decimal] = None,
        max_opening_gap_points: Optional[Decimal] = None,
        slippage_ticks_per_side: Decimal = Decimal("0"),
        disable_session_filters: bool = False,
        max_trades_per_session: Optional[int] = None,
        max_hold_minutes: Optional[int] = None,
        last_entry_time: Optional[time] = None,
    ) -> None:
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        if not self.symbols:
            raise ValueError("at least one symbol is required")
        unknown = [symbol for symbol in self.symbols if symbol not in MARKET_DEFAULTS]
        if unknown:
            known = ", ".join(sorted(MARKET_DEFAULTS))
            raise ValueError(f"unsupported symbols {unknown}; expected one of {known}")
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._source_status = SourceStatus(updated_at=utc_now())
        self.engines = {
            symbol: PropChallengePaperEngine(
                symbols=(symbol,),
                output_dir=self.output_dir / symbol,
                account_tier=account_tier,
                max_concurrent_positions=1,
                commission_per_contract=commission_per_contract,
                strategy_family=strategy_family,
                scalp_reward_risk_ratio=scalp_reward_risk_ratio,
                quantity=quantity,
                stop_points=stop_points,
                target_points=target_points,
                breakout_buffer_points=breakout_buffer_points,
                max_opening_range_points=max_opening_range_points,
                max_opening_gap_points=max_opening_gap_points,
                slippage_ticks_per_side=slippage_ticks_per_side,
                disable_session_filters=disable_session_filters,
                max_trades_per_session=max_trades_per_session,
                max_hold_minutes=max_hold_minutes,
                last_entry_time=last_entry_time,
            )
            for symbol in self.symbols
        }

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
        for engine in self.engines.values():
            engine.set_source_status(
                mode,
                message,
                running=running,
                last_error=last_error,
            )

    def process_bar(self, bar: FuturesBar, allow_entries: bool = True) -> dict[str, Any]:
        engine = self.engines.get(bar.symbol.upper())
        if engine is None:
            return {"accepted": False, "reason": f"unmonitored_symbol:{bar.symbol}"}
        return engine.process_bar(bar, allow_entries=allow_entries)

    def process_bars(self, bars: Iterable[tuple[FuturesBar, bool]]) -> dict[str, int]:
        grouped: dict[str, list[tuple[FuturesBar, bool]]] = {symbol: [] for symbol in self.symbols}
        stale = 0
        for bar, allow_entries in bars:
            symbol = bar.symbol.upper()
            if symbol not in grouped:
                stale += 1
                continue
            grouped[symbol].append((bar, allow_entries))

        accepted = 0
        tradable = 0
        for symbol in self.symbols:
            counts = self.engines[symbol].process_bars(grouped[symbol])
            accepted += counts["accepted"]
            stale += counts["stale"]
            tradable += counts["tradable"]
        return {"accepted": accepted, "stale": stale, "tradable": tradable}

    def snapshot(self) -> dict[str, Any]:
        portfolio_snaps = {symbol: engine.snapshot() for symbol, engine in self.engines.items()}
        stats = self._aggregate_stats(portfolio_snaps)
        account = self._aggregate_account(portfolio_snaps)
        active_trades = {
            symbol: snap["active_trade"]
            for symbol, snap in portfolio_snaps.items()
            if snap.get("active_trade")
        }
        recent_trades = []
        recent_events = []
        for symbol, snap in portfolio_snaps.items():
            for trade in snap.get("recent_trades", [])[-10:]:
                recent_trades.append(dict(trade))
            for event in snap.get("recent_events", [])[-10:]:
                row = dict(event)
                row["symbol"] = symbol
                recent_events.append(row)

        recent_trades.sort(key=lambda row: row.get("exit_time") or row.get("entry_time") or "")
        recent_events.sort(key=lambda row: row.get("ts") or "")
        return {
            "ts": utc_now(),
            "mode": "separate_prop_challenge_portfolios",
            "portfolio_mode": "separate",
            "symbols": list(self.symbols),
            "output_dir": str(self.output_dir),
            "source": self._source_status.to_dict(),
            "account": account,
            "active_trade": next(iter(active_trades.values()), {}),
            "active_trades": active_trades,
            "stats": stats,
            "symbols_state": self._symbols_state(portfolio_snaps),
            "portfolios": portfolio_snaps,
            "bars_processed": sum(int(snap["bars_processed"]) for snap in portfolio_snaps.values()),
            "skipped_entry_conflicts": 0,
            "recent_trades": recent_trades[-50:],
            "recent_events": recent_events[-100:],
            "equity_curve": [],
        }

    def _symbols_state(self, portfolio_snaps: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        rows = {}
        for symbol, snap in portfolio_snaps.items():
            state = dict(snap["symbols_state"][symbol])
            account = snap["account"]
            stats = snap["stats"]
            state.update(
                {
                    "account_status": account["status"],
                    "valuation": account["valuation"],
                    "closed_balance": account["closed_balance"],
                    "open_pnl": account["open_pnl"],
                    "mll_buffer": account["mll_buffer"],
                    "dll_buffer": account["dll_buffer"],
                    "trades": stats["trades"],
                    "realized_pnl": stats["realized_pnl"],
                    "active_trade": bool(snap.get("active_trade")),
                }
            )
            rows[symbol] = state
        return rows

    def _aggregate_stats(self, portfolio_snaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
        trades = sum(int(snap["stats"]["trades"]) for snap in portfolio_snaps.values())
        wins = sum(int(snap["stats"]["wins"]) for snap in portfolio_snaps.values())
        losses = sum(int(snap["stats"]["losses"]) for snap in portfolio_snaps.values())
        realized = sum(Decimal(str(snap["stats"]["realized_pnl"])) for snap in portfolio_snaps.values())
        return {
            "trades": trades,
            "wins": wins,
            "losses": losses,
            "win_rate": decimal_to_float(safe_ratio(wins, trades)),
            "realized_pnl": decimal_to_float(money(realized)),
            "avg_trade_pnl": decimal_to_float(safe_money_ratio(realized, trades)),
            "targets": sum(int(snap["stats"]["targets"]) for snap in portfolio_snaps.values()),
            "stops": sum(int(snap["stats"]["stops"]) for snap in portfolio_snaps.values()),
            "rule_exits": sum(int(snap["stats"]["rule_exits"]) for snap in portfolio_snaps.values()),
        }

    def _aggregate_account(self, portfolio_snaps: dict[str, dict[str, Any]]) -> dict[str, Any]:
        active_count = sum(1 for snap in portfolio_snaps.values() if snap["account"]["status"] == "active")
        closed_balance = self._sum_account_field(portfolio_snaps, "closed_balance")
        open_pnl = self._sum_account_field(portfolio_snaps, "open_pnl")
        valuation = self._sum_account_field(portfolio_snaps, "valuation")
        mll_buffer = self._sum_account_field(portfolio_snaps, "mll_buffer")
        dll_buffer = self._sum_account_field(portfolio_snaps, "dll_buffer")
        return {
            "status": f"{active_count}/{len(portfolio_snaps)} active portfolios",
            "closed_balance": decimal_to_float(money(closed_balance)),
            "open_pnl": decimal_to_float(money(open_pnl)),
            "valuation": decimal_to_float(money(valuation)),
            "mll_buffer": decimal_to_float(money(mll_buffer)),
            "dll_buffer": decimal_to_float(money(dll_buffer)),
            "portfolio_count": len(portfolio_snaps),
            "active_portfolios": active_count,
        }

    def _sum_account_field(
        self,
        portfolio_snaps: dict[str, dict[str, Any]],
        field: str,
    ) -> Decimal:
        total = Decimal("0")
        for snap in portfolio_snaps.values():
            value = snap["account"].get(field)
            if value is not None:
                total += Decimal(str(value))
        return money(total)


def decimal_to_float(value: Optional[Decimal]) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def parse_bar_payload(payload: dict[str, Any], default_symbol: str = "NQ") -> FuturesBar:
    return FuturesBar(
        timestamp=parse_timestamp(str(payload["timestamp"])),
        symbol=str(payload.get("symbol") or default_symbol).upper(),
        open=parse_decimal(str(payload["open"])),
        high=parse_decimal(str(payload["high"])),
        low=parse_decimal(str(payload["low"])),
        close=parse_decimal(str(payload["close"])),
        volume=parse_decimal(str(payload.get("volume", "0"))),
    )


def run_yahoo_chart_poll(
    engine: PropChallengePaperEngine | SeparatePortfolioPaperMonitor,
    poll_seconds: int,
) -> None:
    engine.set_source_status("yahoo-chart", "polling Yahoo chart feed", running=True)
    source_started_at = datetime.now(timezone.utc).astimezone(
        ZoneInfo("America/New_York")
    ).replace(tzinfo=None, second=0, microsecond=0)
    while True:
        accepted_total = 0
        tradable_total = 0
        errors: list[str] = []
        pairs: list[tuple[FuturesBar, bool]] = []
        for symbol in engine.symbols:
            yahoo_symbol = MARKET_DEFAULTS[symbol]["yahoo_symbol"]
            try:
                bars = fetch_yahoo_chart_bars(yahoo_symbol, symbol)
                pairs.extend((bar, bar.timestamp >= source_started_at) for bar in bars)
            except Exception as exc:
                errors.append(f"{symbol}:{exc}")

        if pairs:
            counts = engine.process_bars(pairs)
            accepted_total += counts["accepted"]
            tradable_total += counts["tradable"]

        engine.set_source_status(
            "yahoo-chart",
            f"polled {len(engine.symbols)} symbols; accepted {accepted_total} bars, "
            f"{tradable_total} tradable",
            running=True,
            last_error="; ".join(errors[:3]),
        )
        time_module.sleep(max(5, poll_seconds))


HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Prop Challenge Paper Trader</title>
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
    .metrics { display: grid; grid-template-columns: repeat(6, minmax(140px, 1fr)); gap: 12px; }
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
    .grid-2 { display: grid; grid-template-columns: .8fr 1.2fr; gap: 16px; }
    .kv { display: grid; grid-template-columns: 1fr 1fr; gap: 8px 14px; }
    .kv div { border-bottom: 1px solid #edf0f4; padding: 5px 0; display: flex; justify-content: space-between; gap: 8px; }
    @media (max-width: 1000px) {
      header { flex-direction: column; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .grid-2 { grid-template-columns: 1fr; }
      table { min-width: 980px; }
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>Prop Challenge Paper Trader</h1>
      <div class="muted" id="subtitle"></div>
    </div>
    <div class="status"><span class="dot" id="statusDot"></span><span id="statusText">Loading</span></div>
  </header>
  <main>
    <section class="metrics">
      <div class="card"><div class="label">Account</div><div class="value" id="accountStatus">-</div></div>
      <div class="card"><div class="label">Valuation</div><div class="value" id="valuation">-</div></div>
      <div class="card"><div class="label">Realized PnL</div><div class="value" id="realized">-</div></div>
      <div class="card"><div class="label">MLL Buffer</div><div class="value" id="mllBuffer">-</div></div>
      <div class="card"><div class="label">DLL Buffer</div><div class="value" id="dllBuffer">-</div></div>
      <div class="card"><div class="label">Trades</div><div class="value" id="trades">-</div></div>
    </section>
    <section class="grid-2">
      <div class="panel">
        <h2>Active Trade</h2>
        <div class="kv" id="activeTrade"></div>
      </div>
      <div class="panel">
        <h2>Symbols</h2>
        <table>
          <thead><tr><th>Symbol</th><th>Status</th><th>Valuation</th><th>MLL</th><th>DLL</th><th>OR Size</th><th>Skip</th><th>Trades</th><th>RT Cost</th></tr></thead>
          <tbody id="symbolsTable"></tbody>
        </table>
      </div>
    </section>
    <section class="panel">
      <h2>Recent Trades</h2>
      <table>
        <thead><tr><th>Symbol</th><th>Exit</th><th>Side</th><th>Entry</th><th>Exit Price</th><th>PnL</th><th>MAE</th><th>MFE</th></tr></thead>
        <tbody id="tradesTable"></tbody>
      </table>
    </section>
    <section class="panel">
      <h2>Recent Events</h2>
      <table>
        <thead><tr><th>Time</th><th>Kind</th><th>Message</th><th>Valuation</th><th>MLL Buffer</th></tr></thead>
        <tbody id="eventsTable"></tbody>
      </table>
    </section>
  </main>
  <script>
    const money = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', maximumFractionDigits: 0}) : '-';
    const money2 = v => Number.isFinite(Number(v)) ? Number(v).toLocaleString(undefined, {style: 'currency', currency: 'USD', minimumFractionDigits: 2, maximumFractionDigits: 2}) : '-';
    const fixed = (v, d=2) => Number.isFinite(Number(v)) ? Number(v).toFixed(d) : '-';
    const set = (id, value) => document.getElementById(id).textContent = value;
    const pnlClass = v => Number(v) >= 0 ? 'good' : 'bad';
    const rowHtml = cells => '<tr>' + cells.map(c => `<td>${c}</td>`).join('') + '</tr>';
    function kvHtml(rows) { return rows.map(([k, v]) => `<div><span class="muted">${k}</span><strong>${v}</strong></div>`).join(''); }
    function renderTable(id, rows, mapper, empty, cols) {
      document.getElementById(id).innerHTML = rows.length ? rows.map(mapper).join('') : `<tr><td colspan="${cols}" class="muted">${empty}</td></tr>`;
    }
    async function refresh() {
      const response = await fetch('/api/snapshot', {cache: 'no-store'});
      const data = await response.json();
      const account = data.account || {};
      const stats = data.stats || {};
      const active = data.active_trade || {};
      const activeTrades = data.active_trades || {};
      const source = data.source || {};
      const states = data.symbols_state || {};
      const symbols = data.symbols || Object.keys(states);
      const separate = data.portfolio_mode === 'separate';
      set('subtitle', `${symbols.join(', ')}; ${separate ? 'separate Topstep-style portfolios' : 'one shared Topstep-style account'}`);
      set('accountStatus', account.status || '-');
      set('valuation', money(account.valuation));
      set('realized', money2(stats.realized_pnl));
      set('mllBuffer', money2(account.mll_buffer));
      set('dllBuffer', money2(account.dll_buffer));
      set('trades', String(stats.trades ?? 0));
      const dot = document.getElementById('statusDot');
      dot.className = 'dot ' + (source.last_error || account.status === 'failed_mll' ? 'bad' : 'ok');
      set('statusText', `${source.mode || '-'}: ${source.message || '-'}${source.last_error ? '; ' + source.last_error : ''}`);
      const activeRows = separate
        ? Object.entries(activeTrades).map(([symbol, trade]) => `${symbol} ${trade.side} ${trade.quantity} @ ${money2(trade.entry_price)}; stop ${money2(trade.stop_price)}; target ${money2(trade.target_price)}`)
        : [];
      document.getElementById('activeTrade').innerHTML = activeRows.length
        ? activeRows.map(row => `<div><span class="muted">Trade</span><strong>${row}</strong></div>`).join('')
        : Object.keys(active).length ? kvHtml([
        ['Symbol', active.symbol],
        ['Side', active.side],
        ['Qty', active.quantity],
        ['Entry', money2(active.entry_price)],
        ['Stop', money2(active.stop_price)],
        ['Target', money2(active.target_price)],
        ['MAE / MFE', `${money2(active.mae_pnl)} / ${money2(active.mfe_pnl)}`],
        ['Min MLL Buffer', money2(active.min_mll_buffer)],
      ]) : '<div class="muted">No open paper trade.</div>';
      renderTable('symbolsTable', symbols, symbol => {
        const s = states[symbol] || {};
        const cfg = s.config || {};
        return rowHtml([
          symbol,
          s.account_status || '-',
          money2(s.valuation),
          money2(s.mll_buffer),
          money2(s.dll_buffer),
          fixed(s.opening_range_size),
          s.skip_reason || '-',
          String(s.trades ?? s.trades_taken_today ?? 0),
          money2(cfg.round_turn_cost),
        ]);
      }, 'No symbol state yet.', 9);
      renderTable('tradesTable', (data.recent_trades || []).slice().reverse().slice(0, 20), row => rowHtml([
        row.symbol,
        row.exit_reason,
        row.side,
        money2(row.entry_price),
        money2(row.exit_price),
        `<span class="${pnlClass(row.realized_pnl)}">${money2(row.realized_pnl)}</span>`,
        money2(row.mae_pnl),
        money2(row.mfe_pnl),
      ]), 'No closed trades yet.', 8);
      renderTable('eventsTable', (data.recent_events || []).slice().reverse().slice(0, 20), row => rowHtml([
        String(row.ts || '').slice(5, 19),
        row.kind,
        row.message,
        money2(row.valuation),
        money2(row.mll_buffer),
      ]), 'No events yet.', 5);
    }
    refresh().catch(console.error);
    setInterval(() => refresh().catch(console.error), 3000);
  </script>
</body>
</html>
"""


def create_app(engine: PropChallengePaperEngine) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(HTML, mimetype="text/html")

    @app.get("/api/snapshot")
    def api_snapshot() -> Any:
        return jsonify(engine.snapshot())

    @app.post("/api/bar")
    def api_bar() -> Any:
        payload = request.get_json(force=True)
        bar = parse_bar_payload(payload)
        return jsonify(engine.process_bar(bar))

    @app.post("/api/reset")
    def api_reset() -> Any:
        return jsonify(engine.reset())

    @app.get("/health")
    def health() -> Any:
        return jsonify({"ok": True, "ts": utc_now(), "snapshot": engine.snapshot()})

    return app


def resolve_symbols(args: argparse.Namespace) -> tuple[str, ...]:
    if args.symbols:
        return tuple(symbol.strip().upper() for symbol in args.symbols.split(",") if symbol.strip())
    if args.symbol_set == "eight":
        return EIGHT_SYMBOLS
    return SIX_SYMBOLS


def parse_optional_decimal_arg(value: Optional[str]) -> Optional[Decimal]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "off"}:
        return None
    return Decimal(normalized)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8789)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", choices=["manual", "yahoo-chart"], default="yahoo-chart")
    parser.add_argument("--symbol-set", choices=["six", "eight"], default="six")
    parser.add_argument("--symbols", help="Comma-separated custom symbols, e.g. NQ,ES,RTY")
    parser.add_argument("--account", default="50K")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--portfolio-mode", choices=["separate", "shared"], default="separate")
    parser.add_argument("--max-concurrent-positions", type=int, default=1)
    parser.add_argument(
        "--strategy-family",
        choices=["orb_fade", "scalp_reversion", "vwap_reversion"],
        default="orb_fade",
    )
    parser.add_argument("--scalp-reward-risk-ratio", default="0.6")
    parser.add_argument("--quantity", type=int)
    parser.add_argument("--stop-points")
    parser.add_argument("--target-points")
    parser.add_argument("--breakout-buffer-points")
    parser.add_argument("--max-opening-range-points")
    parser.add_argument("--max-opening-gap-points")
    parser.add_argument("--disable-session-filters", action="store_true")
    parser.add_argument("--slippage-ticks-per-side", default="0")
    parser.add_argument("--max-trades-per-session", type=int)
    parser.add_argument("--max-hold-minutes", type=int)
    parser.add_argument("--last-entry-time")
    parser.add_argument(
        "--commission-per-contract",
        help="Optional per-side cost override. Omit to use Topstep published product costs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commission_override = (
        Decimal(args.commission_per_contract)
        if args.commission_per_contract
        else None
    )
    last_entry_time = parse_time(args.last_entry_time) if args.last_entry_time else None
    stop_points = parse_optional_decimal_arg(args.stop_points)
    target_points = parse_optional_decimal_arg(args.target_points)
    breakout_buffer_points = parse_optional_decimal_arg(args.breakout_buffer_points)
    max_opening_range_points = parse_optional_decimal_arg(args.max_opening_range_points)
    max_opening_gap_points = parse_optional_decimal_arg(args.max_opening_gap_points)
    slippage_ticks_per_side = Decimal(args.slippage_ticks_per_side)
    if args.portfolio_mode == "separate":
        engine: PropChallengePaperEngine | SeparatePortfolioPaperMonitor = SeparatePortfolioPaperMonitor(
            symbols=resolve_symbols(args),
            output_dir=args.output_dir,
            account_tier=args.account,
            commission_per_contract=commission_override,
            strategy_family=args.strategy_family,
            scalp_reward_risk_ratio=Decimal(args.scalp_reward_risk_ratio),
            quantity=args.quantity,
            stop_points=stop_points,
            target_points=target_points,
            breakout_buffer_points=breakout_buffer_points,
            max_opening_range_points=max_opening_range_points,
            max_opening_gap_points=max_opening_gap_points,
            slippage_ticks_per_side=slippage_ticks_per_side,
            disable_session_filters=args.disable_session_filters,
            max_trades_per_session=args.max_trades_per_session,
            max_hold_minutes=args.max_hold_minutes,
            last_entry_time=last_entry_time,
        )
    else:
        engine = PropChallengePaperEngine(
            symbols=resolve_symbols(args),
            output_dir=args.output_dir,
            account_tier=args.account,
            max_concurrent_positions=args.max_concurrent_positions,
            commission_per_contract=commission_override,
            strategy_family=args.strategy_family,
            scalp_reward_risk_ratio=Decimal(args.scalp_reward_risk_ratio),
            quantity=args.quantity,
            stop_points=stop_points,
            target_points=target_points,
            breakout_buffer_points=breakout_buffer_points,
            max_opening_range_points=max_opening_range_points,
            max_opening_gap_points=max_opening_gap_points,
            slippage_ticks_per_side=slippage_ticks_per_side,
            disable_session_filters=args.disable_session_filters,
            max_trades_per_session=args.max_trades_per_session,
            max_hold_minutes=args.max_hold_minutes,
            last_entry_time=last_entry_time,
        )
    if args.source == "manual":
        engine.set_source_status("manual", "waiting for POST /api/bar", running=False)
    else:
        thread = threading.Thread(
            target=run_yahoo_chart_poll,
            args=(engine, args.poll_seconds),
            daemon=True,
        )
        thread.start()
    app = create_app(engine)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
