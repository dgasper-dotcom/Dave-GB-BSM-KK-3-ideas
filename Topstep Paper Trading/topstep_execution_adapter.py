"""Execution adapter for feeding fills and price paths into Topstep rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

from topstep_rule_simulator import (
    MoneyLike,
    RuleEvent,
    RuleEventType,
    RuleStateError,
    TopstepRuleSimulator,
    money,
)


PriceLike = Union[str, int, float, Decimal]


def price(value: PriceLike) -> Decimal:
    return Decimal(str(value))


class ExecutionError(RuntimeError):
    """Raised for invalid execution events or unsupported account states."""


class FillSide(str, Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self == FillSide.BUY else -1


class BarPathMode(str, Enum):
    CONSERVATIVE = "conservative"
    OHLC = "ohlc"
    OLHC = "olhc"
    CLOSE_ONLY = "close_only"


@dataclass(frozen=True)
class ContractSpec:
    symbol: str
    tick_size: PriceLike
    tick_value: MoneyLike
    commission_per_contract: MoneyLike = 0
    is_micro: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "tick_size", price(self.tick_size))
        object.__setattr__(self, "tick_value", money(self.tick_value))
        object.__setattr__(
            self, "commission_per_contract", money(self.commission_per_contract)
        )
        if self.tick_size <= 0:
            raise ValueError("tick_size must be positive")
        if self.tick_value <= 0:
            raise ValueError("tick_value must be positive")
        if self.commission_per_contract < 0:
            raise ValueError("commission_per_contract must be non-negative")

    @property
    def multiplier(self) -> Decimal:
        return self.tick_value / self.tick_size


DEFAULT_CONTRACT_SPECS: Mapping[str, ContractSpec] = MappingProxyType(
    {
        "ES": ContractSpec("ES", tick_size="0.25", tick_value="12.50"),
        "NQ": ContractSpec("NQ", tick_size="0.25", tick_value="5.00"),
        "CL": ContractSpec("CL", tick_size="0.01", tick_value="10.00"),
        "GC": ContractSpec("GC", tick_size="0.10", tick_value="10.00"),
        "RTY": ContractSpec("RTY", tick_size="0.10", tick_value="5.00"),
        "6E": ContractSpec("6E", tick_size="0.00005", tick_value="6.25"),
        "MES": ContractSpec("MES", tick_size="0.25", tick_value="1.25", is_micro=True),
        "MNQ": ContractSpec("MNQ", tick_size="0.25", tick_value="0.50", is_micro=True),
    }
)


TOPSTEP_ROUND_TURN_COSTS: Mapping[str, Decimal] = MappingProxyType(
    {
        "ES": money("3.80"),
        "NQ": money("3.80"),
        "RTY": money("3.80"),
        "MES": money("1.24"),
        "MNQ": money("1.24"),
        "CL": money("4.04"),
        "GC": money("4.24"),
        "6E": money("4.24"),
    }
)


def topstep_commission_per_side(symbol: str) -> Decimal:
    """Return Topstep's published round-turn product cost split per side."""
    normalized = symbol.upper()
    if normalized not in TOPSTEP_ROUND_TURN_COSTS:
        raise ValueError(f"no Topstep cost model configured for {normalized}")
    return money(TOPSTEP_ROUND_TURN_COSTS[normalized] / Decimal("2"))


@dataclass(frozen=True)
class Fill:
    symbol: str
    side: FillSide
    quantity: int
    fill_price: PriceLike
    commission_per_contract: Optional[MoneyLike] = None
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        if not isinstance(self.side, FillSide):
            object.__setattr__(self, "side", FillSide(str(self.side).lower()))
        object.__setattr__(self, "fill_price", price(self.fill_price))
        if self.quantity <= 0:
            raise ValueError("fill quantity must be positive")

    @property
    def signed_quantity(self) -> int:
        return self.quantity * self.side.sign


@dataclass(frozen=True)
class PriceBar:
    symbol: str
    open: PriceLike
    high: PriceLike
    low: PriceLike
    close: PriceLike
    timestamp: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "open", price(self.open))
        object.__setattr__(self, "high", price(self.high))
        object.__setattr__(self, "low", price(self.low))
        object.__setattr__(self, "close", price(self.close))
        if self.high < max(self.open, self.low, self.close):
            raise ValueError("bar high is below open/low/close")
        if self.low > min(self.open, self.high, self.close):
            raise ValueError("bar low is above open/high/close")


@dataclass(frozen=True)
class Position:
    symbol: str
    quantity: int
    average_price: Decimal
    open_entry_commissions: Decimal

    @property
    def direction(self) -> int:
        if self.quantity > 0:
            return 1
        if self.quantity < 0:
            return -1
        return 0

    @property
    def abs_quantity(self) -> int:
        return abs(self.quantity)


@dataclass(frozen=True)
class ExecutionResult:
    rule_event: RuleEvent
    realized_pnl: Decimal
    open_pnl: Decimal
    positions: Mapping[str, Position]


class FuturesExecutionAdapter:
    """Account-wide futures position adapter for the rule simulator."""

    def __init__(
        self,
        rule_simulator: TopstepRuleSimulator,
        contract_specs: Optional[Mapping[str, ContractSpec]] = None,
    ) -> None:
        self.rule_simulator = rule_simulator
        specs = dict(DEFAULT_CONTRACT_SPECS)
        if contract_specs:
            specs.update({symbol.upper(): spec for symbol, spec in contract_specs.items()})
        self.contract_specs: Dict[str, ContractSpec] = specs
        self.positions: Dict[str, Position] = {}
        self.last_prices: Dict[str, Decimal] = {}

    def process_fill(self, fill: Fill) -> ExecutionResult:
        """Apply an exchange fill and feed realized/unrealized PnL to rules."""
        self._require_tradable_account()
        spec = self._spec(fill.symbol)
        commission_pc = self._commission_per_contract(fill, spec)
        signed_fill_qty = fill.signed_quantity
        old_position = self.positions.get(fill.symbol)

        if old_position is None:
            self._check_contract_limits(fill.symbol, signed_fill_qty)
            self.last_prices[fill.symbol] = fill.fill_price
            entry_commissions = money(commission_pc * fill.quantity)
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=signed_fill_qty,
                average_price=fill.fill_price,
                open_entry_commissions=entry_commissions,
            )
            return self._mark_current_open_pnl(realized_pnl=money(0))

        if old_position.direction == fill.side.sign:
            new_quantity = old_position.quantity + signed_fill_qty
            self._check_contract_limits(fill.symbol, new_quantity)
            self.last_prices[fill.symbol] = fill.fill_price
            new_abs_qty = abs(new_quantity)
            average_price = (
                old_position.average_price * old_position.abs_quantity
                + fill.fill_price * fill.quantity
            ) / Decimal(new_abs_qty)
            entry_commissions = old_position.open_entry_commissions + money(
                commission_pc * fill.quantity
            )
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_quantity,
                average_price=average_price,
                open_entry_commissions=entry_commissions,
            )
            return self._mark_current_open_pnl(realized_pnl=money(0))

        realized_pnl = self._reduce_or_reverse_position(
            spec=spec,
            fill=fill,
            old_position=old_position,
            commission_pc=commission_pc,
        )
        self.last_prices[fill.symbol] = fill.fill_price
        return self._realize_fill_pnl(realized_pnl=realized_pnl)

    def mark_price(
        self,
        symbol: str,
        mark_price: PriceLike,
        liquidation_slippage: MoneyLike = 0,
    ) -> ExecutionResult:
        """Mark one symbol and enforce account risk against total open PnL."""
        self._require_tradable_account()
        symbol = symbol.upper()
        self._spec(symbol)
        self.last_prices[symbol] = price(mark_price)
        open_pnl = self.current_open_pnl()
        event = self.rule_simulator.mark_to_market(
            open_pnl=open_pnl,
            liquidation_slippage=liquidation_slippage,
        )
        self._clear_positions_if_rule_stopped_trading(event)
        return self._result(event=event, realized_pnl=money(0))

    def mark_price_path(
        self,
        symbol: str,
        price_values: Iterable[PriceLike],
        liquidation_slippage: MoneyLike = 0,
    ) -> ExecutionResult:
        """Feed a tick/intrabar path into ``mark_to_market_path``."""
        self._require_tradable_account()
        symbol = symbol.upper()
        self._spec(symbol)

        pnl_path = []
        for raw_price in price_values:
            self.last_prices[symbol] = price(raw_price)
            pnl_path.append(self.current_open_pnl())

        event = self.rule_simulator.mark_to_market_path(
            pnl_path,
            liquidation_slippage=liquidation_slippage,
        )
        self._clear_positions_if_rule_stopped_trading(event)
        return self._result(event=event, realized_pnl=money(0))

    def mark_bar(
        self,
        bar: PriceBar,
        path_mode: BarPathMode = BarPathMode.CONSERVATIVE,
        liquidation_slippage: MoneyLike = 0,
    ) -> ExecutionResult:
        """Convert an OHLC bar into an intrabar path and mark it."""
        if not isinstance(path_mode, BarPathMode):
            path_mode = BarPathMode(str(path_mode).lower())
        return self.mark_price_path(
            symbol=bar.symbol,
            price_values=self._bar_path(bar=bar, mode=path_mode),
            liquidation_slippage=liquidation_slippage,
        )

    def current_open_pnl(self) -> Decimal:
        total = money(0)
        for symbol, position in self.positions.items():
            mark = self.last_prices.get(symbol, position.average_price)
            total += self._position_open_pnl(position, self._spec(symbol), mark)
        return money(total)

    def has_open_position(self, symbol: Optional[str] = None) -> bool:
        if symbol is None:
            return bool(self.positions)
        return symbol.upper() in self.positions

    def position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol.upper())

    def market_fill(
        self,
        symbol: str,
        side: FillSide,
        quantity: int,
        mid_price: PriceLike,
        spread_ticks: PriceLike = 1,
        slippage_ticks: PriceLike = 0,
        commission_per_contract: Optional[MoneyLike] = None,
        timestamp: Optional[str] = None,
    ) -> Fill:
        """Create a conservative market fill from midpoint, spread, and slippage."""
        symbol = symbol.upper()
        spec = self._spec(symbol)
        if not isinstance(side, FillSide):
            side = FillSide(str(side).lower())
        adverse_ticks = price(spread_ticks) / Decimal(2) + price(slippage_ticks)
        fill_price = price(mid_price) + spec.tick_size * adverse_ticks * side.sign
        return Fill(
            symbol=symbol,
            side=side,
            quantity=quantity,
            fill_price=fill_price,
            commission_per_contract=commission_per_contract,
            timestamp=timestamp,
        )

    def _reduce_or_reverse_position(
        self,
        spec: ContractSpec,
        fill: Fill,
        old_position: Position,
        commission_pc: Decimal,
    ) -> Decimal:
        exit_qty = min(old_position.abs_quantity, fill.quantity)
        entry_fee_allocated = money(
            old_position.open_entry_commissions
            * Decimal(exit_qty)
            / Decimal(old_position.abs_quantity)
        )
        exit_fee = money(commission_pc * exit_qty)
        gross_realized = (
            (fill.fill_price - old_position.average_price)
            * spec.multiplier
            * old_position.direction
            * Decimal(exit_qty)
        )
        realized_pnl = money(gross_realized - entry_fee_allocated - exit_fee)

        remaining_old_qty = old_position.abs_quantity - exit_qty
        reversal_qty = fill.quantity - exit_qty

        if remaining_old_qty > 0:
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=old_position.direction * remaining_old_qty,
                average_price=old_position.average_price,
                open_entry_commissions=old_position.open_entry_commissions
                - entry_fee_allocated,
            )
            return realized_pnl

        if reversal_qty > 0:
            new_quantity = fill.side.sign * reversal_qty
            self._check_contract_limits(fill.symbol, new_quantity)
            self.positions[fill.symbol] = Position(
                symbol=fill.symbol,
                quantity=new_quantity,
                average_price=fill.fill_price,
                open_entry_commissions=money(commission_pc * reversal_qty),
            )
            return realized_pnl

        del self.positions[fill.symbol]
        return realized_pnl

    def _realize_fill_pnl(self, realized_pnl: Decimal) -> ExecutionResult:
        event = self.rule_simulator.close_position(
            realized_pnl=realized_pnl,
            remaining_open_pnl=self.current_open_pnl(),
        )
        self._clear_positions_if_rule_stopped_trading(event)
        return self._result(event=event, realized_pnl=realized_pnl)

    def _mark_current_open_pnl(self, realized_pnl: Decimal) -> ExecutionResult:
        event = self.rule_simulator.mark_to_market(open_pnl=self.current_open_pnl())
        self._clear_positions_if_rule_stopped_trading(event)
        return self._result(event=event, realized_pnl=realized_pnl)

    def _position_open_pnl(
        self,
        position: Position,
        spec: ContractSpec,
        mark_price: Decimal,
    ) -> Decimal:
        gross = (
            (mark_price - position.average_price)
            * spec.multiplier
            * Decimal(position.quantity)
        )
        return money(gross - position.open_entry_commissions)

    def _bar_path(self, bar: PriceBar, mode: BarPathMode) -> Tuple[Decimal, ...]:
        if mode == BarPathMode.CLOSE_ONLY:
            return (bar.close,)
        if mode == BarPathMode.OHLC:
            return (bar.open, bar.high, bar.low, bar.close)
        if mode == BarPathMode.OLHC:
            return (bar.open, bar.low, bar.high, bar.close)

        position = self.positions.get(bar.symbol)
        if position is None or position.quantity >= 0:
            return (bar.open, bar.low, bar.high, bar.close)
        return (bar.open, bar.high, bar.low, bar.close)

    def _check_contract_limits(self, symbol: str, new_signed_quantity: int) -> None:
        symbol = symbol.upper()
        spec = self._spec(symbol)
        regular_total = 0
        micro_total = 0

        for position_symbol, position in self.positions.items():
            if position_symbol == symbol:
                qty = abs(new_signed_quantity)
            else:
                qty = position.abs_quantity
            if self._spec(position_symbol).is_micro:
                micro_total += qty
            else:
                regular_total += qty

        if symbol not in self.positions:
            if spec.is_micro:
                micro_total += abs(new_signed_quantity)
            else:
                regular_total += abs(new_signed_quantity)

        if regular_total > self.rule_simulator.config.max_contracts:
            raise ExecutionError(
                f"max regular contracts exceeded: {regular_total} > "
                f"{self.rule_simulator.config.max_contracts}"
            )
        if micro_total > self.rule_simulator.config.max_micro_contracts:
            raise ExecutionError(
                f"max micro contracts exceeded: {micro_total} > "
                f"{self.rule_simulator.config.max_micro_contracts}"
            )

    def _commission_per_contract(self, fill: Fill, spec: ContractSpec) -> Decimal:
        if fill.commission_per_contract is None:
            return spec.commission_per_contract
        commission = money(fill.commission_per_contract)
        if commission < 0:
            raise ValueError("commission_per_contract must be non-negative")
        return commission

    def _spec(self, symbol: str) -> ContractSpec:
        symbol = symbol.upper()
        try:
            return self.contract_specs[symbol]
        except KeyError as exc:
            raise ExecutionError(f"no contract spec configured for {symbol}") from exc

    def _require_tradable_account(self) -> None:
        try:
            self.rule_simulator._require_tradable_session()
        except RuleStateError as exc:
            raise ExecutionError(str(exc)) from exc

    def _clear_positions_if_rule_stopped_trading(self, event: RuleEvent) -> None:
        if event.event_type in {
            RuleEventType.DLL_BREACH,
            RuleEventType.MLL_BREACH,
            RuleEventType.PASSED,
        }:
            self.positions.clear()
            self.last_prices.clear()

    def _result(self, event: RuleEvent, realized_pnl: Decimal) -> ExecutionResult:
        return ExecutionResult(
            rule_event=event,
            realized_pnl=money(realized_pnl),
            open_pnl=self.rule_simulator.open_pnl,
            positions=dict(self.positions),
        )
