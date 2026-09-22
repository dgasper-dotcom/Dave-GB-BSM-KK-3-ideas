"""Topstep Trading Combine rule simulator.

This module models the account-level challenge rules only. Strategy signals,
contract accounting, and execution fills should feed realized and unrealized PnL
events into this state machine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Iterable, Optional, Union


MoneyLike = Union[str, int, float, Decimal]
CENT = Decimal("0.01")


def money(value: MoneyLike) -> Decimal:
    """Normalize dollars to cents using Decimal arithmetic."""
    if isinstance(value, Decimal):
        raw = value
    else:
        raw = Decimal(str(value))
    return raw.quantize(CENT, rounding=ROUND_HALF_UP)


class RuleStateError(RuntimeError):
    """Raised when the caller tries to apply events to an invalid rule state."""


class ChallengeStatus(str, Enum):
    ACTIVE = "active"
    PASSED = "passed"
    FAILED_MLL = "failed_mll"


class RuleEventType(str, Enum):
    SESSION_STARTED = "session_started"
    EOD_NO_CHANGE = "eod_no_change"
    EOD_MLL_UPDATED = "eod_mll_updated"
    EOD_MLL_LOCKED = "eod_mll_locked"
    MARK_OK = "mark_ok"
    REALIZED_OK = "realized_ok"
    DLL_BREACH = "dll_breach"
    MLL_BREACH = "mll_breach"
    PASSED = "passed"


@dataclass(frozen=True)
class AccountConfig:
    name: str
    starting_balance: MoneyLike
    profit_target: MoneyLike
    mll_distance: MoneyLike
    daily_loss_limit: Optional[MoneyLike]
    max_contracts: int
    max_micro_contracts: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "starting_balance", money(self.starting_balance))
        object.__setattr__(self, "profit_target", money(self.profit_target))
        object.__setattr__(self, "mll_distance", money(self.mll_distance))
        if self.daily_loss_limit is not None:
            object.__setattr__(
                self, "daily_loss_limit", money(self.daily_loss_limit)
            )

    @property
    def target_balance(self) -> Decimal:
        return self.starting_balance + self.profit_target

    @property
    def initial_mll(self) -> Decimal:
        return self.starting_balance - self.mll_distance


DEFAULT_ACCOUNT_CONFIGS = {
    "50K": AccountConfig(
        name="50K",
        starting_balance=50_000,
        profit_target=3_000,
        mll_distance=2_000,
        daily_loss_limit=1_000,
        max_contracts=5,
        max_micro_contracts=50,
    ),
    "100K": AccountConfig(
        name="100K",
        starting_balance=100_000,
        profit_target=6_000,
        mll_distance=3_000,
        daily_loss_limit=2_000,
        max_contracts=10,
        max_micro_contracts=100,
    ),
    "150K": AccountConfig(
        name="150K",
        starting_balance=150_000,
        profit_target=9_000,
        mll_distance=4_500,
        daily_loss_limit=3_000,
        max_contracts=15,
        max_micro_contracts=150,
    ),
}


def get_account_config(tier: str) -> AccountConfig:
    normalized = str(tier).upper().replace("$", "").replace(" ", "")
    aliases = {
        "50": "50K",
        "50000": "50K",
        "50K": "50K",
        "100": "100K",
        "100000": "100K",
        "100K": "100K",
        "150": "150K",
        "150000": "150K",
        "150K": "150K",
    }
    try:
        return DEFAULT_ACCOUNT_CONFIGS[aliases[normalized]]
    except KeyError as exc:
        known = ", ".join(sorted(DEFAULT_ACCOUNT_CONFIGS))
        raise ValueError(f"unknown account tier {tier!r}; expected one of {known}") from exc


@dataclass(frozen=True)
class RuleEvent:
    event_type: RuleEventType
    detail: str
    day_number: int
    closed_balance: Decimal
    open_pnl: Decimal
    valuation: Decimal
    active_mll: Decimal
    active_dll_threshold: Optional[Decimal]
    status: ChallengeStatus
    day_locked: bool
    mll_locked: bool


@dataclass(frozen=True)
class RuleSnapshot:
    day_number: int
    closed_balance: Decimal
    open_pnl: Decimal
    valuation: Decimal
    session_start_closed_balance: Decimal
    highest_eod_closed_balance: Decimal
    active_mll: Decimal
    active_dll_threshold: Optional[Decimal]
    target_balance: Decimal
    status: ChallengeStatus
    session_active: bool
    day_locked: bool
    mll_locked: bool
    dll_breach_count: int


@dataclass
class TopstepRuleSimulator:
    config: AccountConfig
    closed_balance: Decimal = field(init=False)
    open_pnl: Decimal = field(init=False)
    highest_eod_closed_balance: Decimal = field(init=False)
    active_mll: Decimal = field(init=False)
    session_start_closed_balance: Decimal = field(init=False)
    active_dll_threshold: Optional[Decimal] = field(init=False)
    status: ChallengeStatus = field(init=False)
    day_number: int = field(init=False)
    session_active: bool = field(init=False)
    day_locked: bool = field(init=False)
    mll_locked: bool = field(init=False)
    dll_breach_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.closed_balance = self.config.starting_balance
        self.open_pnl = money(0)
        self.highest_eod_closed_balance = self.config.starting_balance
        self.active_mll = self.config.initial_mll
        self.session_start_closed_balance = self.config.starting_balance
        self.active_dll_threshold = self._dll_threshold_from_session_start()
        self.status = ChallengeStatus.ACTIVE
        self.day_number = 1
        self.session_active = True
        self.day_locked = False
        self.mll_locked = self.active_mll >= self.config.starting_balance
        self.dll_breach_count = 0

    @classmethod
    def from_tier(cls, tier: str) -> "TopstepRuleSimulator":
        return cls(config=get_account_config(tier))

    @property
    def valuation(self) -> Decimal:
        return self.closed_balance + self.open_pnl

    def snapshot(self) -> RuleSnapshot:
        return RuleSnapshot(
            day_number=self.day_number,
            closed_balance=self.closed_balance,
            open_pnl=self.open_pnl,
            valuation=self.valuation,
            session_start_closed_balance=self.session_start_closed_balance,
            highest_eod_closed_balance=self.highest_eod_closed_balance,
            active_mll=self.active_mll,
            active_dll_threshold=self.active_dll_threshold,
            target_balance=self.config.target_balance,
            status=self.status,
            session_active=self.session_active,
            day_locked=self.day_locked,
            mll_locked=self.mll_locked,
            dll_breach_count=self.dll_breach_count,
        )

    def start_next_session(self) -> RuleEvent:
        self._require_non_terminal()
        if self.session_active:
            raise RuleStateError("end the current session before starting a new one")

        self.day_number += 1
        self.session_active = True
        self.day_locked = False
        self.open_pnl = money(0)
        self.session_start_closed_balance = self.closed_balance
        self.active_dll_threshold = self._dll_threshold_from_session_start()
        return self._event(
            RuleEventType.SESSION_STARTED,
            "daily loss limit reset for the new session",
        )

    def end_session(self) -> RuleEvent:
        self._require_non_terminal()
        if not self.session_active:
            raise RuleStateError("no active session to end")
        if self.open_pnl != money(0):
            raise RuleStateError(
                "close open positions before EOD; MLL updates from closed balance only"
            )

        event_type = RuleEventType.EOD_NO_CHANGE
        detail = "no new EOD closed balance high; MLL unchanged"

        if self.closed_balance > self.highest_eod_closed_balance:
            self.highest_eod_closed_balance = self.closed_balance
            if not self.mll_locked:
                candidate_mll = self._candidate_mll_from_highest_eod()
                if candidate_mll > self.active_mll:
                    self.active_mll = candidate_mll
                    event_type = RuleEventType.EOD_MLL_UPDATED
                    detail = "new EOD closed balance high; MLL trailed upward"
                    if self.active_mll >= self.config.starting_balance:
                        self.active_mll = self.config.starting_balance
                        self.mll_locked = True
                        event_type = RuleEventType.EOD_MLL_LOCKED
                        detail = "MLL reached starting balance and locked permanently"

        self.session_active = False
        return self._event(event_type, detail)

    def mark_to_market(
        self,
        open_pnl: MoneyLike,
        liquidation_slippage: MoneyLike = 0,
    ) -> RuleEvent:
        """Update unrealized PnL and immediately enforce MLL/DLL thresholds."""
        self._require_tradable_session()
        slippage = self._coerce_liquidation_slippage(liquidation_slippage)
        self.open_pnl = money(open_pnl)
        return self._evaluate_limits(slippage=slippage) or self._event(
            RuleEventType.MARK_OK,
            "valuation remains above active risk limits",
        )

    def mark_to_market_path(
        self,
        open_pnl_values: Iterable[MoneyLike],
        liquidation_slippage: MoneyLike = 0,
    ) -> RuleEvent:
        """Apply intrabar/tick marks and stop at the first rule event."""
        last_event = self._event(
            RuleEventType.MARK_OK,
            "no marks supplied; valuation unchanged",
        )
        for open_pnl in open_pnl_values:
            last_event = self.mark_to_market(
                open_pnl=open_pnl,
                liquidation_slippage=liquidation_slippage,
            )
            if last_event.event_type in {
                RuleEventType.DLL_BREACH,
                RuleEventType.MLL_BREACH,
            }:
                return last_event
        return last_event

    def close_position(
        self,
        realized_pnl: MoneyLike,
        liquidation_slippage: MoneyLike = 0,
        remaining_open_pnl: MoneyLike = 0,
    ) -> RuleEvent:
        """Realize PnL from a fill, inclusive of fees/slippage.

        ``remaining_open_pnl`` lets an execution adapter report a realized
        partial exit while preserving account-wide unrealized PnL from any
        remaining open contracts.
        """
        self._require_tradable_session()
        slippage = self._coerce_liquidation_slippage(liquidation_slippage)
        self.closed_balance += money(realized_pnl)
        self.open_pnl = money(remaining_open_pnl)

        limit_event = self._evaluate_limits(slippage=slippage)
        if limit_event is not None:
            return limit_event

        if self.closed_balance >= self.config.target_balance:
            self.status = ChallengeStatus.PASSED
            self.session_active = False
            self.day_locked = True
            return self._event(
                RuleEventType.PASSED,
                "profit target reached before permanent failure",
            )

        return self._event(
            RuleEventType.REALIZED_OK,
            "realized PnL applied; challenge remains active",
        )

    def _candidate_mll_from_highest_eod(self) -> Decimal:
        return min(
            self.config.starting_balance,
            self.highest_eod_closed_balance - self.config.mll_distance,
        )

    def _dll_threshold_from_session_start(self) -> Optional[Decimal]:
        if self.config.daily_loss_limit is None:
            return None
        return self.session_start_closed_balance - self.config.daily_loss_limit

    def _coerce_liquidation_slippage(self, liquidation_slippage: MoneyLike) -> Decimal:
        slippage = money(liquidation_slippage)
        if slippage < money(0):
            raise ValueError("liquidation_slippage must be non-negative")
        return slippage

    def _evaluate_limits(self, slippage: Decimal) -> Optional[RuleEvent]:
        if self.valuation <= self.active_mll:
            return self._fail_mll(slippage=slippage, detail="active MLL breached")

        if (
            self.active_dll_threshold is not None
            and self.valuation <= self.active_dll_threshold
        ):
            valuation_at_trigger = self.valuation
            self.closed_balance = valuation_at_trigger - slippage
            self.open_pnl = money(0)
            self.dll_breach_count += 1
            self.day_locked = True

            if self.closed_balance <= self.active_mll:
                return self._fail_mll(
                    slippage=money(0),
                    detail="DLL liquidation fill breached active MLL",
                )

            return self._event(
                RuleEventType.DLL_BREACH,
                "active DLL breached; positions flattened and trading locked for day",
            )

        return None

    def _fail_mll(self, slippage: Decimal, detail: str) -> RuleEvent:
        valuation_at_trigger = self.valuation
        self.closed_balance = valuation_at_trigger - slippage
        self.open_pnl = money(0)
        self.status = ChallengeStatus.FAILED_MLL
        self.session_active = False
        self.day_locked = True
        return self._event(RuleEventType.MLL_BREACH, detail)

    def _require_non_terminal(self) -> None:
        if self.status == ChallengeStatus.PASSED:
            raise RuleStateError("challenge already passed")
        if self.status == ChallengeStatus.FAILED_MLL:
            raise RuleStateError("challenge already failed by MLL breach")

    def _require_tradable_session(self) -> None:
        self._require_non_terminal()
        if not self.session_active:
            raise RuleStateError("no active session")
        if self.day_locked:
            raise RuleStateError("DLL lockout is active until the next session")

    def _event(self, event_type: RuleEventType, detail: str) -> RuleEvent:
        return RuleEvent(
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
        )
