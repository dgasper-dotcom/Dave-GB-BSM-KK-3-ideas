"""Topstep Express Funded Account payout simulator.

The challenge simulator is a barrier-to-target model. An Express Funded Account
is a payout-lifetime model: start from a zero PnL balance, avoid the funded MLL,
qualify for withdrawals, and preserve enough buffer to keep the account alive.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import Enum
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Optional, Sequence

from topstep_rule_simulator import MoneyLike, money


class XFAStatus(str, Enum):
    ACTIVE = "active"
    FAILED = "failed"


class XFAEventType(str, Enum):
    SESSION_STARTED = "session_started"
    SESSION_ENDED = "session_ended"
    MARK_OK = "mark_ok"
    REALIZED_OK = "realized_ok"
    MLL_LOCKED = "mll_locked"
    DLL_BREACH = "dll_breach"
    MLL_BREACH = "mll_breach"
    PAYOUT = "payout"
    PAYOUT_SKIPPED = "payout_skipped"


class XFAPayoutPath(str, Enum):
    STANDARD = "standard"
    CONSISTENCY = "consistency"


class XFARiskState(str, Enum):
    DANGER = "danger"
    HEALTHY = "healthy"
    PROTECTED = "protected"


@dataclass(frozen=True)
class XFAAccountConfig:
    name: str = "50K"
    starting_balance: MoneyLike = 0
    max_loss_limit: MoneyLike = 2_000
    mll_lock_profit: MoneyLike = 2_000
    locked_mll: MoneyLike = 0
    daily_loss_limit: Optional[MoneyLike] = None
    payout_split: Decimal = Decimal("0.90")
    max_payout_fraction: Decimal = Decimal("0.50")
    standard_payout_cap: MoneyLike = 2_000
    consistency_payout_cap: MoneyLike = 3_000
    standard_payout_cap_with_dll: MoneyLike = 4_000
    consistency_payout_cap_with_dll: MoneyLike = 6_000
    min_winning_day: MoneyLike = 150
    standard_winning_days: int = 5
    consistency_trading_days: int = 3
    consistency_max_best_day_share: Decimal = Decimal("0.40")

    def __post_init__(self) -> None:
        object.__setattr__(self, "starting_balance", money(self.starting_balance))
        object.__setattr__(self, "max_loss_limit", money(self.max_loss_limit))
        object.__setattr__(self, "mll_lock_profit", money(self.mll_lock_profit))
        object.__setattr__(self, "locked_mll", money(self.locked_mll))
        if self.daily_loss_limit is not None:
            object.__setattr__(self, "daily_loss_limit", money(self.daily_loss_limit))
        object.__setattr__(self, "payout_split", Decimal(str(self.payout_split)))
        object.__setattr__(
            self, "max_payout_fraction", Decimal(str(self.max_payout_fraction))
        )
        object.__setattr__(self, "standard_payout_cap", money(self.standard_payout_cap))
        object.__setattr__(
            self, "consistency_payout_cap", money(self.consistency_payout_cap)
        )
        object.__setattr__(
            self,
            "standard_payout_cap_with_dll",
            money(self.standard_payout_cap_with_dll),
        )
        object.__setattr__(
            self,
            "consistency_payout_cap_with_dll",
            money(self.consistency_payout_cap_with_dll),
        )
        object.__setattr__(self, "min_winning_day", money(self.min_winning_day))
        object.__setattr__(
            self,
            "consistency_max_best_day_share",
            Decimal(str(self.consistency_max_best_day_share)),
        )

        if self.max_loss_limit <= 0:
            raise ValueError("max_loss_limit must be positive")
        if self.mll_lock_profit < 0:
            raise ValueError("mll_lock_profit must be non-negative")
        if self.daily_loss_limit is not None and self.daily_loss_limit <= 0:
            raise ValueError("daily_loss_limit must be positive")
        if not (Decimal("0") < self.payout_split <= Decimal("1")):
            raise ValueError("payout_split must be in (0, 1]")
        if not (Decimal("0") < self.max_payout_fraction <= Decimal("1")):
            raise ValueError("max_payout_fraction must be in (0, 1]")
        for name, value in {
            "standard_payout_cap": self.standard_payout_cap,
            "consistency_payout_cap": self.consistency_payout_cap,
            "standard_payout_cap_with_dll": self.standard_payout_cap_with_dll,
            "consistency_payout_cap_with_dll": self.consistency_payout_cap_with_dll,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.standard_winning_days <= 0:
            raise ValueError("standard_winning_days must be positive")
        if self.consistency_trading_days <= 0:
            raise ValueError("consistency_trading_days must be positive")
        if not (Decimal("0") < self.consistency_max_best_day_share <= Decimal("1")):
            raise ValueError("consistency_max_best_day_share must be in (0, 1]")

    @property
    def initial_mll(self) -> Decimal:
        return money(self.starting_balance - self.max_loss_limit)


@dataclass(frozen=True)
class XFAPayoutPolicy:
    path: XFAPayoutPath = XFAPayoutPath.CONSISTENCY
    retained_buffer: MoneyLike = 0
    min_payout: MoneyLike = 125
    request_on_eligible: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.path, XFAPayoutPath):
            object.__setattr__(self, "path", XFAPayoutPath(str(self.path).lower()))
        object.__setattr__(self, "retained_buffer", money(self.retained_buffer))
        object.__setattr__(self, "min_payout", money(self.min_payout))
        if self.retained_buffer < 0:
            raise ValueError("retained_buffer must be non-negative")
        if self.min_payout < 0:
            raise ValueError("min_payout must be non-negative")


@dataclass(frozen=True)
class XFARiskPolicy:
    danger_buffer: Optional[MoneyLike] = None
    danger_scale: Decimal = Decimal("1")
    healthy_scale: Decimal = Decimal("1")
    protected_profit: Optional[MoneyLike] = None
    protected_scale: Decimal = Decimal("1")

    def __post_init__(self) -> None:
        if self.danger_buffer is not None:
            object.__setattr__(self, "danger_buffer", money(self.danger_buffer))
        if self.protected_profit is not None:
            object.__setattr__(self, "protected_profit", money(self.protected_profit))
        object.__setattr__(self, "danger_scale", Decimal(str(self.danger_scale)))
        object.__setattr__(self, "healthy_scale", Decimal(str(self.healthy_scale)))
        object.__setattr__(self, "protected_scale", Decimal(str(self.protected_scale)))
        if self.danger_buffer is not None and self.danger_buffer < 0:
            raise ValueError("danger_buffer must be non-negative")
        if self.protected_profit is not None and self.protected_profit < 0:
            raise ValueError("protected_profit must be non-negative")
        for name, value in {
            "danger_scale": self.danger_scale,
            "healthy_scale": self.healthy_scale,
            "protected_scale": self.protected_scale,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

    def scale_for(self, state: XFARiskState) -> Decimal:
        if state == XFARiskState.DANGER:
            return self.danger_scale
        if state == XFARiskState.PROTECTED:
            return self.protected_scale
        return self.healthy_scale


@dataclass(frozen=True)
class XFAEvent:
    event_type: XFAEventType
    detail: str
    day_number: int
    closed_balance: Decimal
    open_pnl: Decimal
    valuation: Decimal
    active_mll: Decimal
    active_dll_threshold: Optional[Decimal]
    status: XFAStatus
    day_locked: bool
    mll_locked: bool
    gross_payouts: Decimal
    trader_payouts: Decimal
    payout_count: int


@dataclass
class XFAAccountSimulator:
    config: XFAAccountConfig = field(default_factory=XFAAccountConfig)
    closed_balance: Decimal = field(init=False)
    open_pnl: Decimal = field(init=False)
    active_mll: Decimal = field(init=False)
    mll_locked: bool = field(init=False)
    session_start_closed_balance: Decimal = field(init=False)
    status: XFAStatus = field(init=False)
    day_number: int = field(init=False)
    session_active: bool = field(init=False)
    day_locked: bool = field(init=False)
    dll_breach_count: int = field(init=False)
    payout_count: int = field(init=False)
    gross_payouts: Decimal = field(init=False)
    trader_payouts: Decimal = field(init=False)
    min_valuation: Decimal = field(init=False)
    period_start_balance: Decimal = field(init=False)
    period_trading_days: int = field(init=False)
    period_winning_days: int = field(init=False)
    period_best_day_pnl: Decimal = field(init=False)
    consistency_block_count: int = field(init=False)
    session_trade_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.closed_balance = self.config.starting_balance
        self.open_pnl = money(0)
        self.active_mll = self.config.initial_mll
        self.mll_locked = self.active_mll >= self.config.locked_mll
        self.session_start_closed_balance = self.closed_balance
        self.status = XFAStatus.ACTIVE
        self.day_number = 1
        self.session_active = True
        self.day_locked = False
        self.dll_breach_count = 0
        self.payout_count = 0
        self.gross_payouts = money(0)
        self.trader_payouts = money(0)
        self.min_valuation = self.valuation
        self.period_start_balance = self.closed_balance
        self.period_trading_days = 0
        self.period_winning_days = 0
        self.period_best_day_pnl = money(0)
        self.consistency_block_count = 0
        self.session_trade_count = 0

    @property
    def valuation(self) -> Decimal:
        return money(self.closed_balance + self.open_pnl)

    @property
    def active_dll_threshold(self) -> Optional[Decimal]:
        if self.config.daily_loss_limit is None:
            return None
        return money(self.session_start_closed_balance - self.config.daily_loss_limit)

    def start_next_session(self) -> XFAEvent:
        self._require_active()
        if self.session_active:
            raise RuntimeError("end current XFA session before starting the next one")
        self.day_number += 1
        self.session_active = True
        self.day_locked = False
        self.open_pnl = money(0)
        self.session_start_closed_balance = self.closed_balance
        self.session_trade_count = 0
        return self._event(XFAEventType.SESSION_STARTED, "XFA session started")

    def end_session(self) -> XFAEvent:
        self._require_active()
        if not self.session_active:
            raise RuntimeError("no active XFA session")
        if self.open_pnl != money(0):
            raise RuntimeError("close open positions before ending XFA session")

        day_pnl = money(self.closed_balance - self.session_start_closed_balance)
        if self.session_trade_count > 0:
            self.period_trading_days += 1
        if day_pnl >= self.config.min_winning_day:
            self.period_winning_days += 1
        if day_pnl > self.period_best_day_pnl:
            self.period_best_day_pnl = day_pnl

        self.session_active = False
        return self._event(XFAEventType.SESSION_ENDED, "XFA session ended")

    def mark_to_market(
        self,
        open_pnl: MoneyLike,
        liquidation_slippage: MoneyLike = 0,
    ) -> XFAEvent:
        self._require_tradable_session()
        self.open_pnl = money(open_pnl)
        self.min_valuation = min(self.min_valuation, self.valuation)
        return self._evaluate_limits(money(liquidation_slippage)) or self._event(
            XFAEventType.MARK_OK,
            "valuation remains above XFA limits",
        )

    def mark_to_market_path(
        self,
        open_pnl_values: Iterable[MoneyLike],
        liquidation_slippage: MoneyLike = 0,
    ) -> XFAEvent:
        last_event = self._event(XFAEventType.MARK_OK, "no marks supplied")
        for open_pnl in open_pnl_values:
            last_event = self.mark_to_market(open_pnl, liquidation_slippage)
            if last_event.event_type in {
                XFAEventType.DLL_BREACH,
                XFAEventType.MLL_BREACH,
            }:
                return last_event
        return last_event

    def close_position(
        self,
        realized_pnl: MoneyLike,
        liquidation_slippage: MoneyLike = 0,
        remaining_open_pnl: MoneyLike = 0,
    ) -> XFAEvent:
        self._require_tradable_session()
        self.session_trade_count += 1
        self.closed_balance = money(self.closed_balance + money(realized_pnl))
        self.open_pnl = money(remaining_open_pnl)
        self.min_valuation = min(self.min_valuation, self.valuation)

        limit_event = self._evaluate_limits(money(liquidation_slippage))
        if limit_event is not None:
            return limit_event

        lock_event = self._maybe_lock_mll("XFA balance reached MLL lock profit")
        if lock_event is not None:
            return lock_event

        return self._event(XFAEventType.REALIZED_OK, "realized PnL applied")

    def payout_eligible(self, policy: XFAPayoutPolicy) -> bool:
        if self.status != XFAStatus.ACTIVE:
            return False
        if self.closed_balance <= self.config.starting_balance:
            return False
        if policy.path == XFAPayoutPath.STANDARD:
            return self.period_winning_days >= self.config.standard_winning_days

        if self.period_trading_days < self.config.consistency_trading_days:
            return False
        period_profit = money(self.closed_balance - self.period_start_balance)
        if period_profit <= 0:
            return False
        max_best_day = money(period_profit * self.config.consistency_max_best_day_share)
        return self.period_best_day_pnl <= max_best_day

    def request_payout(self, policy: XFAPayoutPolicy) -> XFAEvent:
        self._require_active()
        if self.session_active:
            raise RuntimeError("request payouts after ending the current session")
        if not self.payout_eligible(policy):
            self.consistency_block_count += (
                1 if policy.path == XFAPayoutPath.CONSISTENCY else 0
            )
            return self._event(XFAEventType.PAYOUT_SKIPPED, "payout not eligible")

        withdrawable = money(self.closed_balance - money(policy.retained_buffer))
        payout_cap = money(self.closed_balance * self.config.max_payout_fraction)
        gross_payout = min(
            withdrawable,
            payout_cap,
            self._payout_request_cap(policy.path),
        )
        if gross_payout < policy.min_payout or gross_payout <= 0:
            return self._event(
                XFAEventType.PAYOUT_SKIPPED,
                "payout below requested minimum or retained buffer",
            )

        self.closed_balance = money(self.closed_balance - gross_payout)
        self.gross_payouts = money(self.gross_payouts + gross_payout)
        self.trader_payouts = money(
            self.trader_payouts + money(gross_payout * self.config.payout_split)
        )
        self.payout_count += 1
        self._lock_mll("first payout processed; XFA MLL set to zero")
        self._reset_payout_period()
        return self._event(XFAEventType.PAYOUT, "XFA payout processed")

    def _payout_request_cap(self, path: XFAPayoutPath) -> Decimal:
        has_dll = self.config.daily_loss_limit is not None
        if path == XFAPayoutPath.STANDARD:
            return (
                self.config.standard_payout_cap_with_dll
                if has_dll
                else self.config.standard_payout_cap
            )
        return (
            self.config.consistency_payout_cap_with_dll
            if has_dll
            else self.config.consistency_payout_cap
        )

    def _reset_payout_period(self) -> None:
        self.period_start_balance = self.closed_balance
        self.period_trading_days = 0
        self.period_winning_days = 0
        self.period_best_day_pnl = money(0)

    def _maybe_lock_mll(self, detail: str) -> Optional[XFAEvent]:
        if self.mll_locked:
            return None
        if self.closed_balance >= money(
            self.config.starting_balance + self.config.mll_lock_profit
        ):
            self._lock_mll(detail)
            return self._event(XFAEventType.MLL_LOCKED, detail)
        return None

    def _lock_mll(self, detail: str) -> None:
        del detail
        self.active_mll = self.config.locked_mll
        self.mll_locked = True

    def _evaluate_limits(self, slippage: Decimal) -> Optional[XFAEvent]:
        if self.valuation <= self.active_mll:
            return self._fail(slippage, "XFA MLL breached")

        if self.active_dll_threshold is not None and self.valuation <= self.active_dll_threshold:
            self.closed_balance = money(self.valuation - slippage)
            self.open_pnl = money(0)
            self.dll_breach_count += 1
            self.day_locked = True
            self.min_valuation = min(self.min_valuation, self.closed_balance)
            if self.closed_balance <= self.active_mll:
                return self._fail(money(0), "XFA DLL liquidation breached MLL")
            return self._event(
                XFAEventType.DLL_BREACH,
                "XFA DLL breached; trading locked for session",
            )
        return None

    def _fail(self, slippage: Decimal, detail: str) -> XFAEvent:
        self.closed_balance = money(self.valuation - slippage)
        self.open_pnl = money(0)
        self.status = XFAStatus.FAILED
        self.session_active = False
        self.day_locked = True
        self.min_valuation = min(self.min_valuation, self.closed_balance)
        return self._event(XFAEventType.MLL_BREACH, detail)

    def _require_active(self) -> None:
        if self.status != XFAStatus.ACTIVE:
            raise RuntimeError("XFA account is no longer active")

    def _require_tradable_session(self) -> None:
        self._require_active()
        if not self.session_active:
            raise RuntimeError("no active XFA session")
        if self.day_locked:
            raise RuntimeError("XFA DLL lockout is active until next session")

    def _event(self, event_type: XFAEventType, detail: str) -> XFAEvent:
        return XFAEvent(
            event_type=event_type,
            detail=detail,
            day_number=self.day_number,
            closed_balance=self.closed_balance,
            open_pnl=self.open_pnl,
            valuation=self.valuation,
            active_mll=self.active_mll,
            active_dll_threshold=self.active_dll_threshold,
            status=self.status,
            day_locked=self.day_locked,
            mll_locked=self.mll_locked,
            gross_payouts=self.gross_payouts,
            trader_payouts=self.trader_payouts,
            payout_count=self.payout_count,
        )


@dataclass(frozen=True)
class TradePath:
    session_date: date
    realized_pnl: Decimal
    mae_pnl: Decimal
    mfe_pnl: Decimal
    exit_reason: str = ""
    risk_state: str = ""


@dataclass(frozen=True)
class TradingDay:
    session_date: date
    trades: tuple[TradePath, ...]


@dataclass(frozen=True)
class XFATrialResult:
    trial: int
    outcome: str
    days: int
    trades: int
    end_balance: Decimal
    gross_payouts: Decimal
    trader_payouts: Decimal
    payout_count: int
    dll_breaches: int
    consistency_blocks: int
    failure_reason: str
    min_valuation: Decimal
    danger_trades: int
    healthy_trades: int
    protected_trades: int
    skipped_trades: int

    def to_row(self) -> dict[str, Any]:
        return {
            "trial": self.trial,
            "outcome": self.outcome,
            "days": self.days,
            "trades": self.trades,
            "end_balance": str(self.end_balance),
            "gross_payouts": str(self.gross_payouts),
            "trader_payouts": str(self.trader_payouts),
            "payout_count": self.payout_count,
            "dll_breaches": self.dll_breaches,
            "consistency_blocks": self.consistency_blocks,
            "failure_reason": self.failure_reason,
            "min_valuation": str(self.min_valuation),
            "danger_trades": self.danger_trades,
            "healthy_trades": self.healthy_trades,
            "protected_trades": self.protected_trades,
            "skipped_trades": self.skipped_trades,
        }


def load_trade_days(path: str | Path) -> list[TradingDay]:
    grouped: dict[date, list[TradePath]] = {}
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"session_date", "realized_pnl", "mae_pnl", "mfe_pnl"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing required trade columns: {sorted(missing)}")
        for row in reader:
            session_date = date.fromisoformat(row["session_date"])
            grouped.setdefault(session_date, []).append(
                TradePath(
                    session_date=session_date,
                    realized_pnl=money(row["realized_pnl"]),
                    mae_pnl=money(row["mae_pnl"]),
                    mfe_pnl=money(row["mfe_pnl"]),
                    exit_reason=row.get("exit_reason", ""),
                    risk_state=row.get("risk_state", ""),
                )
            )
    return [
        TradingDay(session_date=session_date, trades=tuple(trades))
        for session_date, trades in sorted(grouped.items())
    ]


def xfa_risk_state(
    sim: XFAAccountSimulator,
    risk_policy: XFARiskPolicy,
) -> XFARiskState:
    drawdown_buffer = money(sim.valuation - sim.active_mll)
    if risk_policy.danger_buffer is not None and drawdown_buffer <= risk_policy.danger_buffer:
        return XFARiskState.DANGER
    if (
        risk_policy.protected_profit is not None
        and sim.closed_balance >= risk_policy.protected_profit
    ):
        return XFARiskState.PROTECTED
    return XFARiskState.HEALTHY


def run_xfa_bootstrap(
    days: Sequence[TradingDay],
    trials: int,
    account_config: XFAAccountConfig,
    payout_policy: XFAPayoutPolicy,
    risk_policy: Optional[XFARiskPolicy] = None,
    max_days: int = 504,
    seed: int = 7,
    pnl_haircut_per_trade: MoneyLike = 0,
    liquidation_slippage: MoneyLike = 0,
) -> list[XFATrialResult]:
    if not days:
        raise ValueError("at least one trading day is required")
    if trials <= 0:
        raise ValueError("trials must be positive")
    if max_days <= 0:
        raise ValueError("max_days must be positive")
    rng = random.Random(seed)
    return [
        run_xfa_trial(
            trial=index,
            days=days,
            rng=rng,
            account_config=account_config,
            payout_policy=payout_policy,
            risk_policy=risk_policy or XFARiskPolicy(),
            max_days=max_days,
            pnl_haircut_per_trade=money(pnl_haircut_per_trade),
            liquidation_slippage=money(liquidation_slippage),
        )
        for index in range(1, trials + 1)
    ]


def run_xfa_trial(
    trial: int,
    days: Sequence[TradingDay],
    rng: random.Random,
    account_config: XFAAccountConfig,
    payout_policy: XFAPayoutPolicy,
    risk_policy: XFARiskPolicy,
    max_days: int,
    pnl_haircut_per_trade: Decimal = Decimal("0"),
    liquidation_slippage: Decimal = Decimal("0"),
) -> XFATrialResult:
    sim = XFAAccountSimulator(account_config)
    trade_count = 0
    skipped_trades = 0
    state_counts = {
        XFARiskState.DANGER: 0,
        XFARiskState.HEALTHY: 0,
        XFARiskState.PROTECTED: 0,
    }
    failure_reason = ""

    for day_number in range(1, max_days + 1):
        sampled_day = rng.choice(days)
        for trade in sampled_day.trades:
            state = xfa_risk_state(sim, risk_policy)
            scale = risk_policy.scale_for(state)
            if scale == 0:
                skipped_trades += 1
                continue

            trade_count += 1
            state_counts[state] += 1
            event = replay_xfa_trade(
                sim=sim,
                trade=trade,
                pnl_haircut_per_trade=pnl_haircut_per_trade,
                liquidation_slippage=liquidation_slippage,
                risk_scale=scale,
            )
            if event.event_type == XFAEventType.MLL_BREACH:
                failure_reason = event.detail
                return xfa_trial_result(
                    trial,
                    sim,
                    day_number,
                    trade_count,
                    XFAStatus.FAILED.value,
                    failure_reason,
                    state_counts,
                    skipped_trades,
                )
            if event.event_type == XFAEventType.DLL_BREACH:
                break
            if sim.day_locked:
                break

        if sim.status != XFAStatus.ACTIVE:
            return xfa_trial_result(
                trial,
                sim,
                day_number,
                trade_count,
                XFAStatus.FAILED.value,
                failure_reason or "XFA failed",
                state_counts,
                skipped_trades,
            )

        if sim.session_active:
            sim.end_session()
        if payout_policy.request_on_eligible and sim.payout_eligible(payout_policy):
            sim.request_payout(payout_policy)

        if day_number < max_days and sim.status == XFAStatus.ACTIVE:
            sim.start_next_session()

    return xfa_trial_result(
        trial,
        sim,
        max_days,
        trade_count,
        "survived_horizon",
        "",
        state_counts,
        skipped_trades,
    )


def replay_xfa_trade(
    sim: XFAAccountSimulator,
    trade: TradePath,
    pnl_haircut_per_trade: Decimal,
    liquidation_slippage: Decimal,
    risk_scale: Decimal = Decimal("1"),
) -> XFAEvent:
    scaled_haircut = money(pnl_haircut_per_trade * risk_scale)
    mark_path = xfa_mark_path(
        money(trade.mae_pnl * risk_scale),
        money(trade.mfe_pnl * risk_scale),
        scaled_haircut,
    )
    event = sim.mark_to_market_path(mark_path, liquidation_slippage)
    if event.event_type in {XFAEventType.DLL_BREACH, XFAEventType.MLL_BREACH}:
        return event
    return sim.close_position(
        realized_pnl=money(trade.realized_pnl * risk_scale - scaled_haircut),
        liquidation_slippage=liquidation_slippage,
    )


def xfa_mark_path(
    mae_pnl: Decimal,
    mfe_pnl: Decimal,
    pnl_haircut_per_trade: Decimal,
) -> tuple[Decimal, ...]:
    marks: list[Decimal] = []
    if mae_pnl < 0:
        marks.append(money(mae_pnl - pnl_haircut_per_trade))
    if mfe_pnl > 0:
        marks.append(money(mfe_pnl - pnl_haircut_per_trade))
    return tuple(marks)


def xfa_trial_result(
    trial: int,
    sim: XFAAccountSimulator,
    days: int,
    trades: int,
    outcome: str,
    failure_reason: str,
    state_counts: dict[XFARiskState, int],
    skipped_trades: int,
) -> XFATrialResult:
    return XFATrialResult(
        trial=trial,
        outcome=outcome,
        days=days,
        trades=trades,
        end_balance=sim.closed_balance,
        gross_payouts=sim.gross_payouts,
        trader_payouts=sim.trader_payouts,
        payout_count=sim.payout_count,
        dll_breaches=sim.dll_breach_count,
        consistency_blocks=sim.consistency_block_count,
        failure_reason=failure_reason,
        min_valuation=sim.min_valuation,
        danger_trades=state_counts[XFARiskState.DANGER],
        healthy_trades=state_counts[XFARiskState.HEALTHY],
        protected_trades=state_counts[XFARiskState.PROTECTED],
        skipped_trades=skipped_trades,
    )


def summarize_xfa_trials(
    results: Sequence[XFATrialResult],
    *,
    challenge_cost: MoneyLike = 85,
    challenge_pass_rate: Decimal = Decimal("0.33"),
    activation_fee: MoneyLike = 0,
) -> dict[str, Any]:
    if not results:
        raise ValueError("at least one result is required")
    challenge_cost = money(challenge_cost)
    activation_fee = money(activation_fee)
    challenge_pass_rate = Decimal(str(challenge_pass_rate))
    if not (Decimal("0") < challenge_pass_rate <= Decimal("1")):
        raise ValueError("challenge_pass_rate must be in (0, 1]")

    payouts = [result.trader_payouts for result in results]
    payout_counts = [result.payout_count for result in results]
    days = [result.days for result in results]
    failed = [result for result in results if result.outcome == XFAStatus.FAILED.value]
    survived = [result for result in results if result.outcome != XFAStatus.FAILED.value]
    any_payout = [result for result in results if result.trader_payouts > 0]
    avg_trader_payout = decimal_mean(payouts)
    expected_cost_per_passed_account = money(challenge_cost / challenge_pass_rate)
    net_per_passed_account = money(
        avg_trader_payout - activation_fee - expected_cost_per_passed_account
    )
    ev_per_challenge_attempt = money(
        challenge_pass_rate * (avg_trader_payout - activation_fee) - challenge_cost
    )

    return {
        "trials": len(results),
        "survival_probability": str(rate(len(survived), len(results))),
        "failure_probability": str(rate(len(failed), len(results))),
        "probability_of_any_payout": str(rate(len(any_payout), len(results))),
        "avg_trader_payout": str(avg_trader_payout),
        "median_trader_payout": str(money(Decimal(str(median(payouts))))),
        "avg_payout_count": str(decimal_mean([Decimal(v) for v in payout_counts])),
        "median_lifespan_days": int(median(days)),
        "avg_lifespan_days": str(decimal_mean([Decimal(v) for v in days])),
        "avg_end_balance": str(decimal_mean([r.end_balance for r in results])),
        "avg_min_valuation": str(decimal_mean([r.min_valuation for r in results])),
        "challenge_cost": str(challenge_cost),
        "challenge_pass_rate": str(challenge_pass_rate),
        "activation_fee": str(activation_fee),
        "expected_challenge_cost_per_passed_account": str(
            expected_cost_per_passed_account
        ),
        "net_ev_per_passed_account": str(net_per_passed_account),
        "ev_per_challenge_attempt": str(ev_per_challenge_attempt),
        "failure_reasons": failure_reasons(failed),
    }


def decimal_mean(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return money(0)
    return money(Decimal(str(mean(values))))


def rate(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def failure_reasons(results: Sequence[XFATrialResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        reason = result.failure_reason or "unknown"
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def write_xfa_outputs(
    results: Sequence[XFATrialResult],
    output_dir: str | Path,
    metadata: dict[str, Any],
    summary: dict[str, Any],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps({"metadata": metadata, "summary": summary}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (out / "trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].to_row()))
        writer.writeheader()
        writer.writerows(result.to_row() for result in results)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output-dir", default="topstep_xfa_monte_carlo")
    parser.add_argument("--trials", type=int, default=10_000)
    parser.add_argument("--max-days", type=int, default=504)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--payout-path", choices=["standard", "consistency"], default="consistency")
    parser.add_argument("--retained-buffer", default="0")
    parser.add_argument("--min-payout", default="125")
    parser.add_argument("--pnl-haircut-per-trade", default="0")
    parser.add_argument("--liquidation-slippage", default="0")
    parser.add_argument("--daily-loss-limit")
    parser.add_argument("--danger-buffer")
    parser.add_argument("--danger-scale", default="1")
    parser.add_argument("--healthy-scale", default="1")
    parser.add_argument("--protected-profit")
    parser.add_argument("--protected-scale", default="1")
    parser.add_argument("--challenge-cost", default="85")
    parser.add_argument("--challenge-pass-rate", default="0.33")
    parser.add_argument("--activation-fee", default="0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    days = load_trade_days(args.trades)
    account_config = XFAAccountConfig(
        daily_loss_limit=Decimal(args.daily_loss_limit)
        if args.daily_loss_limit
        else None
    )
    payout_policy = XFAPayoutPolicy(
        path=XFAPayoutPath(args.payout_path),
        retained_buffer=args.retained_buffer,
        min_payout=args.min_payout,
    )
    risk_policy = XFARiskPolicy(
        danger_buffer=Decimal(args.danger_buffer) if args.danger_buffer else None,
        danger_scale=Decimal(args.danger_scale),
        healthy_scale=Decimal(args.healthy_scale),
        protected_profit=Decimal(args.protected_profit)
        if args.protected_profit
        else None,
        protected_scale=Decimal(args.protected_scale),
    )
    results = run_xfa_bootstrap(
        days=days,
        trials=args.trials,
        account_config=account_config,
        payout_policy=payout_policy,
        risk_policy=risk_policy,
        max_days=args.max_days,
        seed=args.seed,
        pnl_haircut_per_trade=args.pnl_haircut_per_trade,
        liquidation_slippage=args.liquidation_slippage,
    )
    metadata = {
        "trades": args.trades,
        "trading_days": len(days),
        "trials": args.trials,
        "max_days": args.max_days,
        "seed": args.seed,
        "payout_path": payout_policy.path.value,
        "retained_buffer": str(payout_policy.retained_buffer),
        "min_payout": str(payout_policy.min_payout),
        "pnl_haircut_per_trade": str(money(args.pnl_haircut_per_trade)),
        "liquidation_slippage": str(money(args.liquidation_slippage)),
        "danger_buffer": None
        if risk_policy.danger_buffer is None
        else str(risk_policy.danger_buffer),
        "danger_scale": str(risk_policy.danger_scale),
        "healthy_scale": str(risk_policy.healthy_scale),
        "protected_profit": None
        if risk_policy.protected_profit is None
        else str(risk_policy.protected_profit),
        "protected_scale": str(risk_policy.protected_scale),
    }
    summary = summarize_xfa_trials(
        results,
        challenge_cost=args.challenge_cost,
        challenge_pass_rate=Decimal(args.challenge_pass_rate),
        activation_fee=args.activation_fee,
    )
    write_xfa_outputs(results, args.output_dir, metadata, summary)
    print(json.dumps({"output_dir": args.output_dir, "metadata": metadata, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
