"""NQ-only Topstep strategy runner.

The first strategy is intentionally simple: an RTH opening-range breakout with
configurable stop/target geometry. Its purpose is to measure whether a raw trade
edge survives Topstep DLL/MLL constraints, not to be the final research system.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from topstep_data_loader import FuturesBar, iter_futures_bars, parse_timestamp
from topstep_execution_adapter import (
    ContractSpec,
    ExecutionError,
    Fill,
    FillSide,
    FuturesExecutionAdapter,
)
from topstep_rule_simulator import (
    ChallengeStatus,
    RuleEvent,
    RuleEventType,
    TopstepRuleSimulator,
    money,
)


NQ_SPEC = ContractSpec("NQ", tick_size="0.25", tick_value="5.00")
MNQ_SPEC = ContractSpec("MNQ", tick_size="0.25", tick_value="0.50", is_micro=True)


@dataclass(frozen=True)
class OpeningRangeBreakoutConfig:
    strategy_family: str = "orb_breakout"
    account_tier: str = "50K"
    data_symbol: str = "NQ"
    contract_symbol: str = "NQ"
    quantity: int = 1
    pre_lock_quantity: Optional[int] = None
    post_lock_quantity: Optional[int] = None
    opening_range_minutes: int = 15
    breakout_buffer_points: Decimal = Decimal("0")
    stop_points: Decimal = Decimal("30")
    target_points: Decimal = Decimal("60")
    max_hold_minutes: int = 180
    max_trades_per_session: int = 1
    rth_start: time = time(9, 30)
    last_entry_time: time = time(15, 0)
    force_exit_time: time = time(16, 0)
    commission_per_contract: Decimal = Decimal("0")
    max_opening_range_points: Optional[Decimal] = None
    max_opening_gap_points: Optional[Decimal] = None
    pre_lock_filter_mode: str = "any"
    pre_lock_min_mll_buffer: Optional[Decimal] = None
    pre_lock_max_opening_range_points: Optional[Decimal] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategy_family", self.strategy_family.lower())
        object.__setattr__(self, "pre_lock_filter_mode", self.pre_lock_filter_mode.lower())
        object.__setattr__(self, "data_symbol", self.data_symbol.upper())
        object.__setattr__(self, "contract_symbol", self.contract_symbol.upper())
        object.__setattr__(
            self, "breakout_buffer_points", Decimal(str(self.breakout_buffer_points))
        )
        object.__setattr__(self, "stop_points", Decimal(str(self.stop_points)))
        object.__setattr__(self, "target_points", Decimal(str(self.target_points)))
        object.__setattr__(
            self, "commission_per_contract", money(self.commission_per_contract)
        )
        if self.pre_lock_min_mll_buffer is not None:
            object.__setattr__(
                self,
                "pre_lock_min_mll_buffer",
                money(self.pre_lock_min_mll_buffer),
            )
        if self.max_opening_range_points is not None:
            object.__setattr__(
                self,
                "max_opening_range_points",
                Decimal(str(self.max_opening_range_points)),
            )
        if self.max_opening_gap_points is not None:
            object.__setattr__(
                self,
                "max_opening_gap_points",
                Decimal(str(self.max_opening_gap_points)),
            )
        if self.pre_lock_max_opening_range_points is not None:
            object.__setattr__(
                self,
                "pre_lock_max_opening_range_points",
                Decimal(str(self.pre_lock_max_opening_range_points)),
            )
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.pre_lock_quantity is not None and self.pre_lock_quantity <= 0:
            raise ValueError("pre_lock_quantity must be positive")
        if self.post_lock_quantity is not None and self.post_lock_quantity <= 0:
            raise ValueError("post_lock_quantity must be positive")
        if self.opening_range_minutes <= 0:
            raise ValueError("opening_range_minutes must be positive")
        if self.stop_points <= 0:
            raise ValueError("stop_points must be positive")
        if self.target_points <= 0:
            raise ValueError("target_points must be positive")
        if (
            self.pre_lock_min_mll_buffer is not None
            and self.pre_lock_min_mll_buffer < 0
        ):
            raise ValueError("pre_lock_min_mll_buffer must be non-negative")
        if (
            self.max_opening_range_points is not None
            and self.max_opening_range_points <= 0
        ):
            raise ValueError("max_opening_range_points must be positive")
        if (
            self.max_opening_gap_points is not None
            and self.max_opening_gap_points <= 0
        ):
            raise ValueError("max_opening_gap_points must be positive")
        if (
            self.pre_lock_max_opening_range_points is not None
            and self.pre_lock_max_opening_range_points <= 0
        ):
            raise ValueError("pre_lock_max_opening_range_points must be positive")
        if self.max_hold_minutes <= 0:
            raise ValueError("max_hold_minutes must be positive")
        if self.max_trades_per_session <= 0:
            raise ValueError("max_trades_per_session must be positive")
        if self.strategy_family not in {
            "orb_breakout",
            "orb_fade",
            "scalp_reversion",
            "vwap_reversion",
        }:
            raise ValueError(f"unsupported strategy_family: {self.strategy_family}")
        if self.pre_lock_filter_mode not in {"any", "combined"}:
            raise ValueError("pre_lock_filter_mode must be any or combined")


@dataclass
class SessionState:
    session_date: date
    opening_range_end: datetime
    previous_session_close: Optional[Decimal] = None
    opening_high: Optional[Decimal] = None
    opening_low: Optional[Decimal] = None
    opening_gap_points: Optional[Decimal] = None
    trades_taken: int = 0
    recorded_skip_reasons: set[str] = field(default_factory=set)

    @property
    def opening_range_ready(self) -> bool:
        return self.opening_high is not None and self.opening_low is not None

    @property
    def opening_range_size(self) -> Optional[Decimal]:
        if self.opening_high is None or self.opening_low is None:
            return None
        return self.opening_high - self.opening_low


@dataclass
class ActiveTrade:
    attempt_id: int
    session_date: date
    risk_state: str
    side: FillSide
    entry_time: datetime
    entry_price: Decimal
    quantity: int
    stop_price: Decimal
    target_price: Decimal
    entry_mll_buffer: Decimal
    min_mll_buffer: Decimal
    mae_points: Decimal = Decimal("0")
    mfe_points: Decimal = Decimal("0")
    mae_pnl: Decimal = Decimal("0")
    mfe_pnl: Decimal = Decimal("0")
    time_underwater_minutes: int = 0
    bars_held: int = 0


@dataclass(frozen=True)
class TradeRecord:
    attempt_id: int
    session_date: str
    risk_state: str
    side: str
    quantity: int
    entry_time: str
    exit_time: str
    entry_price: Decimal
    exit_price: Decimal
    realized_pnl: Decimal
    exit_reason: str
    rule_event: str
    mae_points: Decimal
    mfe_points: Decimal
    mae_pnl: Decimal
    mfe_pnl: Decimal
    time_to_target_minutes: Optional[int]
    time_underwater_minutes: int
    bars_held: int
    entry_mll_buffer: Decimal
    min_mll_buffer: Decimal

    def to_row(self) -> Dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "session_date": self.session_date,
            "risk_state": self.risk_state,
            "side": self.side,
            "quantity": self.quantity,
            "entry_time": self.entry_time,
            "exit_time": self.exit_time,
            "entry_price": str(self.entry_price),
            "exit_price": str(self.exit_price),
            "realized_pnl": str(self.realized_pnl),
            "exit_reason": self.exit_reason,
            "rule_event": self.rule_event,
            "mae_points": str(self.mae_points),
            "mfe_points": str(self.mfe_points),
            "mae_pnl": str(self.mae_pnl),
            "mfe_pnl": str(self.mfe_pnl),
            "time_to_target_minutes": ""
            if self.time_to_target_minutes is None
            else self.time_to_target_minutes,
            "time_underwater_minutes": self.time_underwater_minutes,
            "bars_held": self.bars_held,
            "entry_mll_buffer": str(self.entry_mll_buffer),
            "min_mll_buffer": str(self.min_mll_buffer),
        }


@dataclass(frozen=True)
class AttemptRecord:
    attempt_id: int
    start_time: str
    end_time: str
    outcome: str
    days: int
    trades: int
    start_balance: Decimal
    end_balance: Decimal
    net_pnl: Decimal
    reached_mll_lock: bool
    dll_breaches: int
    terminal_rule_event: str
    terminal_detail: str
    failure_reason: str

    def to_row(self) -> Dict[str, object]:
        return {
            "attempt_id": self.attempt_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "outcome": self.outcome,
            "days": self.days,
            "trades": self.trades,
            "start_balance": str(self.start_balance),
            "end_balance": str(self.end_balance),
            "net_pnl": str(self.net_pnl),
            "reached_mll_lock": int(self.reached_mll_lock),
            "dll_breaches": self.dll_breaches,
            "terminal_rule_event": self.terminal_rule_event,
            "terminal_detail": self.terminal_detail,
            "failure_reason": self.failure_reason,
        }


@dataclass
class BacktestResult:
    config: OpeningRangeBreakoutConfig
    attempts: List[AttemptRecord] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    risk_filter_skips: Dict[str, int] = field(default_factory=dict)
    rows_processed: int = 0
    first_timestamp: Optional[datetime] = None
    last_timestamp: Optional[datetime] = None

    def summary(self) -> Dict[str, object]:
        completed = [
            a
            for a in self.attempts
            if a.outcome in (ChallengeStatus.PASSED.value, ChallengeStatus.FAILED_MLL.value)
        ]
        passes = [a for a in completed if a.outcome == ChallengeStatus.PASSED.value]
        failures = [a for a in completed if a.outcome == ChallengeStatus.FAILED_MLL.value]
        failure_reasons: Dict[str, int] = {}
        for attempt in failures:
            reason = attempt.failure_reason or "unknown"
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
        wins = [t for t in self.trades if t.realized_pnl > 0]
        losses = [t for t in self.trades if t.realized_pnl < 0]
        total_trade_pnl = sum((t.realized_pnl for t in self.trades), Decimal("0"))
        target_trades = [t for t in self.trades if t.exit_reason == "target"]
        pre_lock_trades = [t for t in self.trades if t.risk_state == "pre_lock"]
        post_lock_trades = [t for t in self.trades if t.risk_state == "post_lock"]
        winners_with_adverse_excursion = [t for t in wins if t.mae_points > 0]
        total_mae_pnl = sum((t.mae_pnl for t in self.trades), Decimal("0"))
        total_mfe_pnl = sum((t.mfe_pnl for t in self.trades), Decimal("0"))
        total_underwater_minutes = sum(t.time_underwater_minutes for t in self.trades)
        target_time_sum = sum(
            (
                Decimal(t.time_to_target_minutes)
                for t in target_trades
                if t.time_to_target_minutes is not None
            ),
            Decimal("0"),
        )
        winner_mae_sum = sum((t.mae_pnl for t in wins), Decimal("0"))
        pass_rate = _safe_ratio(len(passes), len(completed))
        win_rate = _safe_ratio(len(wins), len(self.trades))
        avg_trade_pnl = _safe_decimal_ratio(total_trade_pnl, len(self.trades))
        avg_days_to_pass = _safe_decimal_ratio(
            sum((Decimal(a.days) for a in passes), Decimal("0")), len(passes)
        )
        return {
            "strategy": self.config.strategy_family,
            "account_tier": self.config.account_tier,
            "data_symbol": self.config.data_symbol,
            "contract_symbol": self.config.contract_symbol,
            "quantity": self.config.quantity,
            "pre_lock_quantity": self.config.pre_lock_quantity,
            "post_lock_quantity": self.config.post_lock_quantity,
            "opening_range_minutes": self.config.opening_range_minutes,
            "stop_points": str(self.config.stop_points),
            "target_points": str(self.config.target_points),
            "breakout_buffer_points": str(self.config.breakout_buffer_points),
            "max_opening_range_points": None
            if self.config.max_opening_range_points is None
            else str(self.config.max_opening_range_points),
            "max_opening_gap_points": None
            if self.config.max_opening_gap_points is None
            else str(self.config.max_opening_gap_points),
            "pre_lock_filter_mode": self.config.pre_lock_filter_mode,
            "pre_lock_min_mll_buffer": None
            if self.config.pre_lock_min_mll_buffer is None
            else str(self.config.pre_lock_min_mll_buffer),
            "pre_lock_max_opening_range_points": None
            if self.config.pre_lock_max_opening_range_points is None
            else str(self.config.pre_lock_max_opening_range_points),
            "rows_processed": self.rows_processed,
            "first_timestamp": self.first_timestamp.isoformat(sep=" ")
            if self.first_timestamp
            else None,
            "last_timestamp": self.last_timestamp.isoformat(sep=" ")
            if self.last_timestamp
            else None,
            "attempts": len(self.attempts),
            "completed_attempts": len(completed),
            "passes": len(passes),
            "failures": len(failures),
            "pass_rate": str(pass_rate),
            "trade_count": len(self.trades),
            "pre_lock_trade_count": len(pre_lock_trades),
            "post_lock_trade_count": len(post_lock_trades),
            "win_rate": str(win_rate),
            "avg_trade_pnl": str(avg_trade_pnl),
            "total_trade_pnl": str(money(total_trade_pnl)),
            "avg_mae_pnl": str(_safe_decimal_ratio(total_mae_pnl, len(self.trades))),
            "avg_mfe_pnl": str(_safe_decimal_ratio(total_mfe_pnl, len(self.trades))),
            "avg_time_underwater_minutes": str(
                _safe_decimal_ratio(Decimal(total_underwater_minutes), len(self.trades))
            ),
            "avg_time_to_target_minutes": str(
                _safe_decimal_ratio(target_time_sum, len(target_trades))
            ),
            "winners_with_adverse_excursion_rate": str(
                _safe_ratio(len(winners_with_adverse_excursion), len(wins))
            ),
            "avg_winner_mae_pnl": str(_safe_decimal_ratio(winner_mae_sum, len(wins))),
            "avg_days_to_pass": str(avg_days_to_pass),
            "dll_breaches": sum(a.dll_breaches for a in self.attempts),
            "mll_lock_reached": sum(1 for a in self.attempts if a.reached_mll_lock),
            "failure_reasons": failure_reasons,
            "risk_filter_skips": dict(self.risk_filter_skips),
        }


class OpeningRangeBreakoutRunner:
    def __init__(self, config: OpeningRangeBreakoutConfig) -> None:
        self.config = config
        self.contract_specs = {
            "NQ": ContractSpec(
                "NQ",
                tick_size=NQ_SPEC.tick_size,
                tick_value=NQ_SPEC.tick_value,
                commission_per_contract=config.commission_per_contract,
            ),
            "MNQ": ContractSpec(
                "MNQ",
                tick_size=MNQ_SPEC.tick_size,
                tick_value=MNQ_SPEC.tick_value,
                commission_per_contract=config.commission_per_contract,
                is_micro=True,
            ),
        }

    def run(
        self,
        path: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        max_rows: Optional[int] = None,
    ) -> BacktestResult:
        return self.run_bars(
            iter_futures_bars(
                path,
                symbol=self.config.data_symbol,
                start=start,
                end=end,
            ),
            max_rows=max_rows,
        )

    def run_bars(
        self,
        bars: Iterable[FuturesBar],
        max_rows: Optional[int] = None,
    ) -> BacktestResult:
        result = BacktestResult(config=self.config)
        attempt = _AttemptContext.new(1, self.config, None, self.contract_specs)
        session_state: Optional[SessionState] = None
        active_trade: Optional[ActiveTrade] = None
        previous_bar: Optional[FuturesBar] = None
        previous_session_close: Optional[Decimal] = None
        pending_new_attempt = False

        for bar in bars:
            if max_rows is not None and result.rows_processed >= max_rows:
                break
            result.rows_processed += 1
            if result.first_timestamp is None:
                result.first_timestamp = bar.timestamp
            result.last_timestamp = bar.timestamp

            bar_session_date = session_date_for_timestamp(bar.timestamp)
            if session_state is None:
                session_state = self._new_session_state(
                    bar_session_date,
                    previous_session_close,
                )
                if attempt.start_time is None:
                    attempt.start_time = bar.timestamp
            elif bar_session_date != session_state.session_date:
                if active_trade is not None and previous_bar is not None:
                    active_trade = self._force_exit(
                        result,
                        attempt,
                        active_trade,
                        previous_bar.timestamp,
                        previous_bar.close,
                        "session_end",
                    )
                    pending_new_attempt = self._record_if_terminal(
                        result, attempt, previous_bar.timestamp
                    )

                if not pending_new_attempt and attempt.sim.status == ChallengeStatus.ACTIVE:
                    attempt.sim.end_session()
                    if attempt.sim.status == ChallengeStatus.ACTIVE:
                        attempt.sim.start_next_session()

                if previous_bar is not None:
                    previous_session_close = previous_bar.close

                if pending_new_attempt:
                    attempt_id = attempt.attempt_id + 1
                    attempt = _AttemptContext.new(
                        attempt_id,
                        self.config,
                        bar.timestamp,
                        self.contract_specs,
                    )
                    pending_new_attempt = False

                session_state = self._new_session_state(
                    bar_session_date,
                    previous_session_close,
                )

            if pending_new_attempt:
                previous_bar = bar
                continue

            if active_trade is not None:
                active_trade, terminal_now = self._manage_open_trade(
                    result=result,
                    attempt=attempt,
                    trade=active_trade,
                    bar=bar,
                )
                if terminal_now:
                    pending_new_attempt = self._record_if_terminal(
                        result, attempt, bar.timestamp
                    )
                    previous_bar = bar
                    continue

            if active_trade is None and attempt.sim.status == ChallengeStatus.ACTIVE:
                self._update_opening_range(session_state, bar)
                active_trade = self._maybe_enter(result, attempt, session_state, bar)
                if active_trade is not None:
                    terminal_now = self._record_if_terminal(result, attempt, bar.timestamp)
                    if terminal_now:
                        pending_new_attempt = True
                        active_trade = None

            attempt.reached_mll_lock = attempt.reached_mll_lock or attempt.sim.mll_locked
            previous_bar = bar

        if previous_bar is not None:
            if active_trade is not None:
                self._force_exit(
                    result,
                    attempt,
                    active_trade,
                    previous_bar.timestamp,
                    previous_bar.close,
                    "data_end",
                )
            if attempt.sim.status == ChallengeStatus.ACTIVE:
                try:
                    attempt.sim.end_session()
                except Exception:
                    pass
            if attempt.start_time is not None and not _attempt_recorded(
                result, attempt.attempt_id
            ):
                self._record_attempt(
                    result,
                    attempt,
                    previous_bar.timestamp,
                    outcome=attempt.sim.status.value
                    if attempt.sim.status != ChallengeStatus.ACTIVE
                    else "incomplete",
                )

        return result

    def _new_session_state(
        self,
        session_date: date,
        previous_session_close: Optional[Decimal] = None,
    ) -> SessionState:
        rth_start_dt = datetime.combine(session_date, self.config.rth_start)
        return SessionState(
            session_date=session_date,
            opening_range_end=rth_start_dt
            + timedelta(minutes=self.config.opening_range_minutes),
            previous_session_close=previous_session_close,
        )

    def _update_opening_range(self, session_state: SessionState, bar: FuturesBar) -> None:
        if (
            session_state.opening_gap_points is None
            and session_state.previous_session_close is not None
        ):
            session_state.opening_gap_points = abs(
                bar.open - session_state.previous_session_close
            )
        if not _is_rth_opening_range_bar(
            bar.timestamp, self.config.rth_start, session_state.opening_range_end
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

    def _maybe_enter(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        session_state: SessionState,
        bar: FuturesBar,
    ) -> Optional[ActiveTrade]:
        if session_state.trades_taken >= self.config.max_trades_per_session:
            return None
        if not (
            session_state.opening_range_end <= bar.timestamp
            and bar.timestamp.time() <= self.config.last_entry_time
        ):
            return None

        side = self._entry_side(session_state, bar)
        if side is None:
            return None

        session_skip_reason = self._session_filter_skip_reason(session_state)
        if session_skip_reason is not None:
            self._record_risk_filter_skip(result, session_state, session_skip_reason)
            return None

        skip_reason = self._risk_state_skip_reason(attempt, session_state)
        if skip_reason is not None:
            self._record_risk_filter_skip(result, session_state, skip_reason)
            return None

        quantity = self._entry_quantity(attempt)
        entry_price = _round_to_tick(bar.close, Decimal("0.25"))
        stop_price, target_price = self._stop_target(entry_price, side)
        try:
            event = attempt.adapter.process_fill(
                Fill(
                    self.config.contract_symbol,
                    side,
                    quantity,
                    entry_price,
                )
            ).rule_event
            attempt.remember_rule_event(event, "entry")
        except ExecutionError:
            return None

        session_state.trades_taken += 1
        attempt.trade_count += 1
        entry_mll_buffer = money(attempt.sim.valuation - attempt.sim.active_mll)
        return ActiveTrade(
            attempt_id=attempt.attempt_id,
            session_date=session_state.session_date,
            risk_state=self._risk_state(attempt),
            side=side,
            entry_time=bar.timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_price=stop_price,
            target_price=target_price,
            entry_mll_buffer=entry_mll_buffer,
            min_mll_buffer=entry_mll_buffer,
        )

    def _risk_state_skip_reason(
        self,
        attempt: "_AttemptContext",
        session_state: SessionState,
    ) -> Optional[str]:
        if attempt.sim.mll_locked:
            return None

        configured_filters = []
        triggered_filters = []
        if self.config.pre_lock_min_mll_buffer is not None:
            configured_filters.append("mll_buffer")
            mll_buffer = money(attempt.sim.valuation - attempt.sim.active_mll)
            if mll_buffer < self.config.pre_lock_min_mll_buffer:
                triggered_filters.append("mll_buffer")

        if self.config.pre_lock_max_opening_range_points is not None:
            configured_filters.append("opening_range")
            opening_range_size = session_state.opening_range_size
            if (
                opening_range_size is not None
                and opening_range_size > self.config.pre_lock_max_opening_range_points
            ):
                triggered_filters.append("opening_range")

        if not triggered_filters:
            return None

        if self.config.pre_lock_filter_mode == "combined":
            if len(triggered_filters) != len(configured_filters):
                return None
            if set(triggered_filters) == {"mll_buffer", "opening_range"}:
                return "pre_lock_combined_mll_buffer_opening_range"
            return f"pre_lock_{triggered_filters[0]}"

        if "mll_buffer" in triggered_filters:
            return "pre_lock_mll_buffer"
        if "opening_range" in triggered_filters:
            return "pre_lock_opening_range"

        return None

    def _session_filter_skip_reason(
        self,
        session_state: SessionState,
    ) -> Optional[str]:
        if self.config.max_opening_range_points is not None:
            opening_range_size = session_state.opening_range_size
            if (
                opening_range_size is not None
                and opening_range_size > self.config.max_opening_range_points
            ):
                return "opening_range"
        if (
            self.config.max_opening_gap_points is not None
            and session_state.opening_gap_points is not None
            and session_state.opening_gap_points > self.config.max_opening_gap_points
        ):
            return "opening_gap"
        return None

    def _record_risk_filter_skip(
        self,
        result: BacktestResult,
        session_state: SessionState,
        reason: str,
    ) -> None:
        if reason in session_state.recorded_skip_reasons:
            return
        session_state.recorded_skip_reasons.add(reason)
        result.risk_filter_skips[reason] = result.risk_filter_skips.get(reason, 0) + 1

    def _risk_state(self, attempt: "_AttemptContext") -> str:
        return "post_lock" if attempt.sim.mll_locked else "pre_lock"

    def _entry_quantity(self, attempt: "_AttemptContext") -> int:
        if attempt.sim.mll_locked:
            return self.config.post_lock_quantity or self.config.quantity
        return self.config.pre_lock_quantity or self.config.quantity

    def _entry_side(
        self,
        session_state: SessionState,
        bar: FuturesBar,
    ) -> Optional[FillSide]:
        if self.config.strategy_family in {"vwap_reversion", "scalp_reversion"}:
            if bar.vwap_rth is None or bar.vwap_rth == 0:
                return None
            distance = bar.close - bar.vwap_rth
            if distance >= self.config.breakout_buffer_points:
                return FillSide.SELL
            if distance <= -self.config.breakout_buffer_points:
                return FillSide.BUY
            return None

        if not session_state.opening_range_ready:
            return None
        assert session_state.opening_high is not None
        assert session_state.opening_low is not None

        long_trigger = session_state.opening_high + self.config.breakout_buffer_points
        short_trigger = session_state.opening_low - self.config.breakout_buffer_points
        if self.config.strategy_family == "orb_breakout":
            if bar.close > long_trigger:
                return FillSide.BUY
            if bar.close < short_trigger:
                return FillSide.SELL
            return None

        if self.config.strategy_family == "orb_fade":
            if bar.close > long_trigger:
                return FillSide.SELL
            if bar.close < short_trigger:
                return FillSide.BUY
            return None

        return None

    def _manage_open_trade(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        trade: ActiveTrade,
        bar: FuturesBar,
    ) -> Tuple[Optional[ActiveTrade], bool]:
        forced_time_exit = (
            bar.timestamp.time() >= self.config.force_exit_time
            or int((bar.timestamp - trade.entry_time).total_seconds() // 60)
            >= self.config.max_hold_minutes
        )
        if forced_time_exit:
            self._update_trade_excursions(attempt, trade, self._trade_bar_path(trade, bar))
            closed = self._force_exit(
                result,
                attempt,
                trade,
                bar.timestamp,
                bar.close,
                "time_exit",
            )
            return closed, attempt.sim.status != ChallengeStatus.ACTIVE

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
                attempt,
                trade,
                self._trade_bar_path_until_exit(trade, bar, exit_price, exit_reason),
            )
            closed = self._force_exit(
                result,
                attempt,
                trade,
                bar.timestamp,
                exit_price,
                exit_reason,
            )
            return closed, attempt.sim.status != ChallengeStatus.ACTIVE

        terminal_price = self._mark_bar_path_until_terminal(attempt, trade, bar)
        if terminal_price is not None:
            self._record_forced_rule_exit(
                result,
                attempt,
                trade,
                bar.timestamp,
                terminal_price,
            )
            return None, True

        return trade, False

    def _trade_bar_path(self, trade: ActiveTrade, bar: FuturesBar) -> Tuple[Decimal, ...]:
        if trade.side == FillSide.BUY:
            return (bar.open, bar.low, bar.high, bar.close)
        return (bar.open, bar.high, bar.low, bar.close)

    def _trade_bar_path_until_exit(
        self,
        trade: ActiveTrade,
        bar: FuturesBar,
        exit_price: Decimal,
        exit_reason: str,
    ) -> Tuple[Decimal, ...]:
        path = self._trade_bar_path(trade, bar)
        if exit_reason == "time_exit":
            return path
        if exit_reason == "stop":
            return (path[0], exit_price)
        if exit_reason == "target":
            return (path[0], path[1], exit_price)
        return path

    def _update_trade_excursions(
        self,
        attempt: "_AttemptContext",
        trade: ActiveTrade,
        price_path: Tuple[Decimal, ...],
    ) -> None:
        if not price_path:
            return
        spec = self.contract_specs[self.config.contract_symbol]
        underwater = False
        for mark in price_path:
            favorable_points = (mark - trade.entry_price) * Decimal(trade.side.sign)
            open_pnl = money(
                favorable_points * spec.multiplier * Decimal(trade.quantity)
                - self.config.commission_per_contract * Decimal(trade.quantity)
            )
            mll_buffer = money(attempt.sim.closed_balance + open_pnl - attempt.sim.active_mll)
            trade.min_mll_buffer = min(trade.min_mll_buffer, mll_buffer)
            if favorable_points < 0:
                adverse_points = -favorable_points
                trade.mae_points = max(trade.mae_points, adverse_points)
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
        attempt: "_AttemptContext",
        trade: ActiveTrade,
        bar: FuturesBar,
    ) -> Optional[Decimal]:
        path = self._trade_bar_path(trade, bar)
        observed_path = []
        for mark in path:
            observed_path.append(mark)
            event = attempt.adapter.mark_price(self.config.contract_symbol, mark).rule_event
            if event.event_type in {RuleEventType.DLL_BREACH, RuleEventType.MLL_BREACH}:
                attempt.remember_rule_event(event, event.event_type.value)
                self._update_trade_excursions(
                    attempt, trade, tuple(observed_path)
                )
                return mark
            attempt.remember_rule_event(event, "mark")
        self._update_trade_excursions(attempt, trade, tuple(observed_path))
        return None

    def _force_exit(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        trade: ActiveTrade,
        timestamp: datetime,
        exit_price: Decimal,
        exit_reason: str,
    ) -> Optional[ActiveTrade]:
        if attempt.sim.status != ChallengeStatus.ACTIVE:
            return None
        exit_side = FillSide.SELL if trade.side == FillSide.BUY else FillSide.BUY
        event = attempt.adapter.process_fill(
            Fill(
                self.config.contract_symbol,
                exit_side,
                trade.quantity,
                _round_to_tick(exit_price, Decimal("0.25")),
            )
        ).rule_event
        attempt.remember_rule_event(event, exit_reason)
        realized = attempt.sim.closed_balance - attempt.last_closed_balance
        attempt.last_closed_balance = attempt.sim.closed_balance
        result.trades.append(
            TradeRecord(
                attempt_id=trade.attempt_id,
                session_date=trade.session_date.isoformat(),
                risk_state=trade.risk_state,
                side=trade.side.value,
                quantity=trade.quantity,
                entry_time=trade.entry_time.isoformat(sep=" "),
                exit_time=timestamp.isoformat(sep=" "),
                entry_price=trade.entry_price,
                exit_price=_round_to_tick(exit_price, Decimal("0.25")),
                realized_pnl=money(realized),
                exit_reason=exit_reason,
                rule_event=event.event_type.value,
                mae_points=trade.mae_points,
                mfe_points=trade.mfe_points,
                mae_pnl=trade.mae_pnl,
                mfe_pnl=trade.mfe_pnl,
                time_to_target_minutes=_time_to_target_minutes(
                    trade, timestamp, exit_reason
                ),
                time_underwater_minutes=trade.time_underwater_minutes,
                bars_held=trade.bars_held,
                entry_mll_buffer=trade.entry_mll_buffer,
                min_mll_buffer=trade.min_mll_buffer,
            )
        )
        return None

    def _record_forced_rule_exit(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        trade: ActiveTrade,
        timestamp: datetime,
        exit_price: Decimal,
    ) -> None:
        realized = attempt.sim.closed_balance - attempt.last_closed_balance
        attempt.last_closed_balance = attempt.sim.closed_balance
        result.trades.append(
            TradeRecord(
                attempt_id=trade.attempt_id,
                session_date=trade.session_date.isoformat(),
                risk_state=trade.risk_state,
                side=trade.side.value,
                quantity=trade.quantity,
                entry_time=trade.entry_time.isoformat(sep=" "),
                exit_time=timestamp.isoformat(sep=" "),
                entry_price=trade.entry_price,
                exit_price=_round_to_tick(exit_price, Decimal("0.25")),
                realized_pnl=money(realized),
                exit_reason=attempt.last_exit_reason
                or (
                    attempt.sim.status.value
                    if attempt.sim.status != ChallengeStatus.ACTIVE
                    else "dll_breach"
                ),
                rule_event=attempt.last_rule_event_type
                or (
                    RuleEventType.MLL_BREACH.value
                    if attempt.sim.status == ChallengeStatus.FAILED_MLL
                    else RuleEventType.DLL_BREACH.value
                ),
                mae_points=trade.mae_points,
                mfe_points=trade.mfe_points,
                mae_pnl=trade.mae_pnl,
                mfe_pnl=trade.mfe_pnl,
                time_to_target_minutes=None,
                time_underwater_minutes=trade.time_underwater_minutes,
                bars_held=trade.bars_held,
                entry_mll_buffer=trade.entry_mll_buffer,
                min_mll_buffer=trade.min_mll_buffer,
            )
        )

    def _record_if_terminal(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        timestamp: datetime,
    ) -> bool:
        if attempt.sim.status == ChallengeStatus.ACTIVE:
            return False
        if _attempt_recorded(result, attempt.attempt_id):
            return True
        self._record_attempt(result, attempt, timestamp, attempt.sim.status.value)
        return True

    def _record_attempt(
        self,
        result: BacktestResult,
        attempt: "_AttemptContext",
        timestamp: datetime,
        outcome: str,
    ) -> None:
        start_time = attempt.start_time or timestamp
        days = max(1, attempt.sim.day_number)
        terminal_rule_event = _terminal_rule_event(outcome, attempt)
        terminal_detail = attempt.last_rule_event_detail or ""
        failure_reason = _failure_reason(outcome, attempt)
        result.attempts.append(
            AttemptRecord(
                attempt_id=attempt.attempt_id,
                start_time=start_time.isoformat(sep=" "),
                end_time=timestamp.isoformat(sep=" "),
                outcome=outcome,
                days=days,
                trades=attempt.trade_count,
                start_balance=attempt.start_balance,
                end_balance=attempt.sim.closed_balance,
                net_pnl=money(attempt.sim.closed_balance - attempt.start_balance),
                reached_mll_lock=attempt.reached_mll_lock or attempt.sim.mll_locked,
                dll_breaches=attempt.sim.dll_breach_count,
                terminal_rule_event=terminal_rule_event,
                terminal_detail=terminal_detail,
                failure_reason=failure_reason,
            )
        )

    def _stop_target(self, entry_price: Decimal, side: FillSide) -> Tuple[Decimal, Decimal]:
        if side == FillSide.BUY:
            return entry_price - self.config.stop_points, entry_price + self.config.target_points
        return entry_price + self.config.stop_points, entry_price - self.config.target_points


@dataclass
class _AttemptContext:
    attempt_id: int
    sim: TopstepRuleSimulator
    adapter: FuturesExecutionAdapter
    start_time: Optional[datetime]
    start_balance: Decimal
    last_closed_balance: Decimal
    trade_count: int = 0
    reached_mll_lock: bool = False
    last_rule_event_type: str = ""
    last_rule_event_detail: str = ""
    last_exit_reason: str = ""

    def remember_rule_event(self, event: RuleEvent, exit_reason: str) -> None:
        self.last_rule_event_type = event.event_type.value
        self.last_rule_event_detail = event.detail
        self.last_exit_reason = exit_reason

    @classmethod
    def new(
        cls,
        attempt_id: int,
        config: OpeningRangeBreakoutConfig,
        start_time: Optional[datetime],
        contract_specs: Dict[str, ContractSpec],
    ) -> "_AttemptContext":
        sim = TopstepRuleSimulator.from_tier(config.account_tier)
        adapter = FuturesExecutionAdapter(sim, contract_specs=contract_specs)
        return cls(
            attempt_id=attempt_id,
            sim=sim,
            adapter=adapter,
            start_time=start_time,
            start_balance=sim.closed_balance,
            last_closed_balance=sim.closed_balance,
        )


def session_date_for_timestamp(ts: datetime) -> date:
    if ts.time() >= time(18, 0):
        return (ts + timedelta(days=1)).date()
    return ts.date()


def _is_rth_opening_range_bar(
    ts: datetime,
    rth_start: time,
    opening_range_end: datetime,
) -> bool:
    return ts.time() >= rth_start and ts < opening_range_end


def _attempt_recorded(result: BacktestResult, attempt_id: int) -> bool:
    return any(attempt.attempt_id == attempt_id for attempt in result.attempts)


def _terminal_rule_event(outcome: str, attempt: _AttemptContext) -> str:
    if outcome in {
        ChallengeStatus.PASSED.value,
        ChallengeStatus.FAILED_MLL.value,
    }:
        return attempt.last_rule_event_type
    return ""


def _failure_reason(outcome: str, attempt: _AttemptContext) -> str:
    if outcome != ChallengeStatus.FAILED_MLL.value:
        return ""

    detail = attempt.last_rule_event_detail
    if detail == "DLL liquidation fill breached active MLL":
        return "dll_liquidation_breached_mll"
    if attempt.last_rule_event_type != RuleEventType.MLL_BREACH.value:
        return outcome
    if attempt.last_exit_reason == "stop":
        return "stop_exit_breached_mll"
    if attempt.last_exit_reason == RuleEventType.MLL_BREACH.value:
        return "unrealized_mll_breach"
    return "mll_breach"


def _time_to_target_minutes(
    trade: ActiveTrade,
    timestamp: datetime,
    exit_reason: str,
) -> Optional[int]:
    if exit_reason != "target":
        return None
    return max(0, int((timestamp - trade.entry_time).total_seconds() // 60))


def _round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    ticks = (Decimal(value) / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return ticks * tick


def _safe_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def _safe_decimal_ratio(numerator: Decimal, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal("0.00")
    return money(numerator / Decimal(denominator))


def write_backtest_outputs(result: BacktestResult, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(result.summary(), indent=2) + "\n")
    _write_csv(out / "attempts.csv", [a.to_row() for a in result.attempts])
    _write_csv(out / "trades.csv", [t.to_row() for t in result.trades])


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_time(value: str) -> time:
    return datetime.strptime(value, "%H:%M").time()


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="topstep_nq_orb_results")
    parser.add_argument(
        "--strategy-family",
        default="orb_breakout",
        choices=["orb_breakout", "orb_fade", "scalp_reversion", "vwap_reversion"],
    )
    parser.add_argument("--account", default="50K")
    parser.add_argument("--contract", default="NQ", choices=["NQ", "MNQ"])
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--pre-lock-quantity", type=int)
    parser.add_argument("--post-lock-quantity", type=int)
    parser.add_argument("--opening-range-minutes", type=int, default=15)
    parser.add_argument("--stop-points", default="30")
    parser.add_argument("--target-points", default="60")
    parser.add_argument("--breakout-buffer-points", default="0")
    parser.add_argument("--max-hold-minutes", type=int, default=180)
    parser.add_argument("--max-trades-per-session", type=int, default=1)
    parser.add_argument("--last-entry-time", default="15:00")
    parser.add_argument("--force-exit-time", default="16:00")
    parser.add_argument("--commission-per-contract", default="0")
    parser.add_argument("--max-opening-range-points")
    parser.add_argument("--max-opening-gap-points")
    parser.add_argument(
        "--pre-lock-filter-mode",
        default="any",
        choices=["any", "combined"],
    )
    parser.add_argument("--pre-lock-min-mll-buffer")
    parser.add_argument("--pre-lock-max-opening-range-points")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args(argv)

    config = OpeningRangeBreakoutConfig(
        strategy_family=args.strategy_family,
        account_tier=args.account,
        contract_symbol=args.contract,
        quantity=args.quantity,
        pre_lock_quantity=args.pre_lock_quantity,
        post_lock_quantity=args.post_lock_quantity,
        opening_range_minutes=args.opening_range_minutes,
        stop_points=Decimal(args.stop_points),
        target_points=Decimal(args.target_points),
        breakout_buffer_points=Decimal(args.breakout_buffer_points),
        max_hold_minutes=args.max_hold_minutes,
        max_trades_per_session=args.max_trades_per_session,
        last_entry_time=parse_time(args.last_entry_time),
        force_exit_time=parse_time(args.force_exit_time),
        commission_per_contract=Decimal(args.commission_per_contract),
        max_opening_range_points=Decimal(args.max_opening_range_points)
        if args.max_opening_range_points
        else None,
        max_opening_gap_points=Decimal(args.max_opening_gap_points)
        if args.max_opening_gap_points
        else None,
        pre_lock_filter_mode=args.pre_lock_filter_mode,
        pre_lock_min_mll_buffer=Decimal(args.pre_lock_min_mll_buffer)
        if args.pre_lock_min_mll_buffer
        else None,
        pre_lock_max_opening_range_points=Decimal(
            args.pre_lock_max_opening_range_points
        )
        if args.pre_lock_max_opening_range_points
        else None,
    )
    runner = OpeningRangeBreakoutRunner(config)
    result = runner.run(
        path=args.data,
        start=parse_timestamp(args.start) if args.start else None,
        end=parse_timestamp(args.end) if args.end else None,
        max_rows=args.max_rows,
    )
    write_backtest_outputs(result, args.output_dir)
    print(json.dumps(result.summary(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
