#!/usr/bin/env python3
"""Synthetic tick-path simulator for Topstep scalp risk geometry.

This is not fitted to historical NQ outcomes. It generates plausible 1-tick
price paths in PnL space and replays them through the Topstep rule simulator.
Use it to ask what kind of tick-level process would be needed for a
high-win-rate, low-RR scalp to pass a Trading Combine.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional speed path
    np = None

from topstep_execution_adapter import DEFAULT_CONTRACT_SPECS, TOPSTEP_ROUND_TURN_COSTS
from topstep_rule_simulator import (
    AccountConfig,
    ChallengeStatus,
    RuleEventType,
    TopstepRuleSimulator,
    get_account_config,
    money,
)


@dataclass(frozen=True)
class SyntheticTickConfig:
    account_tier: str = "50K"
    symbol: str = "NQ"
    stop_points: Decimal = Decimal("20")
    reward_risk_ratio: Decimal = Decimal("0.5")
    favorable_tick_probability: Decimal = Decimal("0.5000")
    trades_per_day: int = 3
    max_days: int = 250
    max_ticks_per_trade: int = 2_000
    quantity: int = 1
    round_turn_cost: Decimal = Decimal("3.80")
    slippage_ticks_per_side: Decimal = Decimal("0")
    include_consistency: bool = True
    consistency_fraction: Decimal = Decimal("0.50")
    min_trading_days: int = 2
    use_daily_loss_limit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", self.symbol.upper())
        object.__setattr__(self, "stop_points", Decimal(str(self.stop_points)))
        object.__setattr__(
            self, "reward_risk_ratio", Decimal(str(self.reward_risk_ratio))
        )
        object.__setattr__(
            self,
            "favorable_tick_probability",
            Decimal(str(self.favorable_tick_probability)),
        )
        object.__setattr__(self, "round_turn_cost", money(self.round_turn_cost))
        object.__setattr__(
            self, "slippage_ticks_per_side", Decimal(str(self.slippage_ticks_per_side))
        )
        object.__setattr__(
            self, "consistency_fraction", Decimal(str(self.consistency_fraction))
        )
        if self.symbol not in DEFAULT_CONTRACT_SPECS:
            raise ValueError(f"unsupported symbol {self.symbol!r}")
        if self.stop_points <= 0:
            raise ValueError("stop_points must be positive")
        if self.reward_risk_ratio <= 0:
            raise ValueError("reward_risk_ratio must be positive")
        if not Decimal("0") <= self.favorable_tick_probability <= Decimal("1"):
            raise ValueError("favorable_tick_probability must be in [0, 1]")
        if self.trades_per_day <= 0:
            raise ValueError("trades_per_day must be positive")
        if self.max_days <= 0:
            raise ValueError("max_days must be positive")
        if self.max_ticks_per_trade <= 0:
            raise ValueError("max_ticks_per_trade must be positive")
        if self.quantity <= 0:
            raise ValueError("quantity must be positive")
        if self.round_turn_cost < 0:
            raise ValueError("round_turn_cost must be non-negative")
        if self.slippage_ticks_per_side < 0:
            raise ValueError("slippage_ticks_per_side must be non-negative")
        if not Decimal("0") < self.consistency_fraction <= Decimal("1"):
            raise ValueError("consistency_fraction must be in (0, 1]")
        if self.min_trading_days <= 0:
            raise ValueError("min_trading_days must be positive")

    @property
    def tick_size(self) -> Decimal:
        return DEFAULT_CONTRACT_SPECS[self.symbol].tick_size

    @property
    def tick_value(self) -> Decimal:
        return DEFAULT_CONTRACT_SPECS[self.symbol].tick_value

    @property
    def stop_ticks(self) -> int:
        return _points_to_ticks(self.stop_points, self.tick_size)

    @property
    def target_points(self) -> Decimal:
        return (self.stop_points * self.reward_risk_ratio).quantize(
            self.tick_size, rounding=ROUND_HALF_UP
        )

    @property
    def target_ticks(self) -> int:
        ticks = max(1, _points_to_ticks(self.target_points, self.tick_size))
        return ticks

    @property
    def gross_risk(self) -> Decimal:
        return money(self.stop_ticks * self.tick_value * self.quantity)

    @property
    def gross_reward(self) -> Decimal:
        return money(self.target_ticks * self.tick_value * self.quantity)

    @property
    def slippage_cost_round_turn(self) -> Decimal:
        return money(
            self.slippage_ticks_per_side
            * Decimal("2")
            * self.tick_value
            * self.quantity
        )

    @property
    def total_round_turn_cost(self) -> Decimal:
        return money(self.round_turn_cost * self.quantity + self.slippage_cost_round_turn)

    @property
    def entry_side_cost(self) -> Decimal:
        return money(self.total_round_turn_cost / Decimal("2"))

    @property
    def breakeven_win_rate(self) -> Decimal:
        breakeven = (self.gross_risk + self.total_round_turn_cost) / (
            self.gross_risk + self.gross_reward
        )
        return breakeven.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)

    def to_assumptions(self) -> Dict[str, object]:
        return {
            "account_tier": self.account_tier,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "stop_points": str(self.stop_points),
            "target_points": str(self.target_points),
            "reward_risk_ratio": str(self.reward_risk_ratio),
            "stop_ticks": self.stop_ticks,
            "target_ticks": self.target_ticks,
            "gross_risk": str(self.gross_risk),
            "gross_reward": str(self.gross_reward),
            "favorable_tick_probability": str(self.favorable_tick_probability),
            "trades_per_day": self.trades_per_day,
            "max_days": self.max_days,
            "max_ticks_per_trade": self.max_ticks_per_trade,
            "round_turn_cost": str(self.round_turn_cost),
            "slippage_ticks_per_side": str(self.slippage_ticks_per_side),
            "slippage_cost_round_turn": str(self.slippage_cost_round_turn),
            "total_round_turn_cost": str(self.total_round_turn_cost),
            "entry_side_cost": str(self.entry_side_cost),
            "breakeven_win_rate": str(self.breakeven_win_rate),
            "include_consistency": int(self.include_consistency),
            "consistency_fraction": str(self.consistency_fraction),
            "min_trading_days": self.min_trading_days,
            "use_daily_loss_limit": int(self.use_daily_loss_limit),
        }


@dataclass(frozen=True)
class SyntheticTrade:
    terminal_ticks: int
    open_pnl_path: Sequence[Decimal]
    realized_pnl: Decimal
    mae_pnl: Decimal
    mfe_pnl: Decimal
    ticks_held: int
    exit_reason: str

    @property
    def is_win(self) -> bool:
        return self.realized_pnl > money(0)


@dataclass(frozen=True)
class SyntheticTrial:
    outcome: str
    days: int
    trades: int
    net_pnl: Decimal
    reached_mll_lock: bool
    dll_breaches: int
    best_day_pnl: Decimal


@dataclass(frozen=True)
class SyntheticRow:
    config: SyntheticTickConfig
    trials: int
    passes: int
    failures: int
    incomplete: int
    pass_rate: Decimal
    fail_rate: Decimal
    incomplete_rate: Decimal
    lock_rate: Decimal
    pass_given_lock: Decimal
    avg_days_to_pass: Decimal
    avg_trades_to_pass: Decimal
    win_rate: Decimal
    target_hit_rate: Decimal
    stop_hit_rate: Decimal
    time_exit_rate: Decimal
    avg_trade_pnl: Decimal
    trade_pnl_stddev: Decimal
    avg_mae_pnl: Decimal
    avg_winner_mae_pnl: Decimal
    avg_mfe_pnl: Decimal
    avg_ticks_held: Decimal
    avg_dll_breaches: Decimal

    def to_row(self) -> Dict[str, object]:
        row = self.config.to_assumptions()
        row.update(
            {
                "trials": self.trials,
                "passes": self.passes,
                "failures": self.failures,
                "incomplete": self.incomplete,
                "pass_rate": str(self.pass_rate),
                "fail_rate": str(self.fail_rate),
                "incomplete_rate": str(self.incomplete_rate),
                "lock_rate": str(self.lock_rate),
                "pass_given_lock": str(self.pass_given_lock),
                "avg_days_to_pass": str(self.avg_days_to_pass),
                "avg_trades_to_pass": str(self.avg_trades_to_pass),
                "win_rate": str(self.win_rate),
                "target_hit_rate": str(self.target_hit_rate),
                "stop_hit_rate": str(self.stop_hit_rate),
                "time_exit_rate": str(self.time_exit_rate),
                "avg_trade_pnl": str(self.avg_trade_pnl),
                "trade_pnl_stddev": str(self.trade_pnl_stddev),
                "avg_mae_pnl": str(self.avg_mae_pnl),
                "avg_winner_mae_pnl": str(self.avg_winner_mae_pnl),
                "avg_mfe_pnl": str(self.avg_mfe_pnl),
                "avg_ticks_held": str(self.avg_ticks_held),
                "avg_dll_breaches": str(self.avg_dll_breaches),
            }
        )
        return row


def simulate_config(
    config: SyntheticTickConfig,
    trials: int,
    seed: int,
) -> SyntheticRow:
    if trials <= 0:
        raise ValueError("trials must be positive")
    rng = np.random.default_rng(seed) if np is not None else random.Random(seed)
    account = _manual_pass_account(config.account_tier, config.use_daily_loss_limit)
    account_target = get_account_config(config.account_tier).profit_target

    trial_results: List[SyntheticTrial] = []
    trades: List[SyntheticTrade] = []
    for _ in range(trials):
        trial, trial_trades = _run_trial(config, account, account_target, rng)
        trial_results.append(trial)
        trades.extend(trial_trades)

    passes = [trial for trial in trial_results if trial.outcome == "passed"]
    failures = [trial for trial in trial_results if trial.outcome == "failed_mll"]
    incomplete = [trial for trial in trial_results if trial.outcome == "incomplete"]
    locked = [trial for trial in trial_results if trial.reached_mll_lock]
    locked_passes = [trial for trial in passes if trial.reached_mll_lock]
    winners = [trade for trade in trades if trade.is_win]

    return SyntheticRow(
        config=config,
        trials=trials,
        passes=len(passes),
        failures=len(failures),
        incomplete=len(incomplete),
        pass_rate=_ratio(len(passes), trials),
        fail_rate=_ratio(len(failures), trials),
        incomplete_rate=_ratio(len(incomplete), trials),
        lock_rate=_ratio(len(locked), trials),
        pass_given_lock=_ratio(len(locked_passes), len(locked)),
        avg_days_to_pass=_mean_int([trial.days for trial in passes]),
        avg_trades_to_pass=_mean_int([trial.trades for trial in passes]),
        win_rate=_ratio(sum(1 for trade in trades if trade.is_win), len(trades)),
        target_hit_rate=_ratio(
            sum(1 for trade in trades if trade.exit_reason == "target"), len(trades)
        ),
        stop_hit_rate=_ratio(
            sum(1 for trade in trades if trade.exit_reason == "stop"), len(trades)
        ),
        time_exit_rate=_ratio(
            sum(1 for trade in trades if trade.exit_reason == "time_exit"), len(trades)
        ),
        avg_trade_pnl=_mean_money([trade.realized_pnl for trade in trades]),
        trade_pnl_stddev=_stddev_money([trade.realized_pnl for trade in trades]),
        avg_mae_pnl=_mean_money([trade.mae_pnl for trade in trades]),
        avg_winner_mae_pnl=_mean_money([trade.mae_pnl for trade in winners]),
        avg_mfe_pnl=_mean_money([trade.mfe_pnl for trade in trades]),
        avg_ticks_held=_mean_int([trade.ticks_held for trade in trades]),
        avg_dll_breaches=_mean_int([trial.dll_breaches for trial in trial_results]),
    )


def run_sweep(
    *,
    account_tier: str,
    symbol: str,
    stop_points: Sequence[Decimal],
    reward_risk_ratios: Sequence[Decimal],
    favorable_tick_probabilities: Sequence[Decimal],
    slippage_ticks_per_side: Sequence[Decimal],
    trades_per_day: int,
    max_days: int,
    max_ticks_per_trade: int,
    quantity: int,
    trials: int,
    seed: int,
    round_turn_cost: Optional[Decimal],
    include_consistency: bool,
    min_trading_days: int,
    use_daily_loss_limit: bool,
) -> List[SyntheticRow]:
    symbol = symbol.upper()
    if round_turn_cost is None:
        round_turn_cost = TOPSTEP_ROUND_TURN_COSTS[symbol]

    rows: List[SyntheticRow] = []
    index = 0
    for stop in stop_points:
        for rr in reward_risk_ratios:
            for p in favorable_tick_probabilities:
                for slip in slippage_ticks_per_side:
                    index += 1
                    config = SyntheticTickConfig(
                        account_tier=account_tier,
                        symbol=symbol,
                        stop_points=stop,
                        reward_risk_ratio=rr,
                        favorable_tick_probability=p,
                        trades_per_day=trades_per_day,
                        max_days=max_days,
                        max_ticks_per_trade=max_ticks_per_trade,
                        quantity=quantity,
                        round_turn_cost=round_turn_cost,
                        slippage_ticks_per_side=slip,
                        include_consistency=include_consistency,
                        min_trading_days=min_trading_days,
                        use_daily_loss_limit=use_daily_loss_limit,
                    )
                    rows.append(simulate_config(config, trials=trials, seed=seed + index))

    return sorted(
        rows,
        key=lambda row: (
            row.pass_rate,
            row.avg_trade_pnl,
            -abs(row.avg_mae_pnl),
            row.win_rate,
        ),
        reverse=True,
    )


def write_outputs(rows: Sequence[SyntheticRow], output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    row_dicts = [row.to_row() for row in rows]
    _write_csv(out / "leaderboard.csv", row_dicts)

    target_profile = [
        row.to_row()
        for row in rows
        if Decimal("0.65") <= row.win_rate <= Decimal("0.75")
        and Decimal("0.4") <= row.config.reward_risk_ratio <= Decimal("0.7")
        and row.avg_trade_pnl >= money(0)
    ]
    _write_csv(out / "target_profile_positive_ev.csv", target_profile)

    summary = {
        "rows": len(rows),
        "best": row_dicts[0] if row_dicts else {},
        "target_profile_positive_ev_rows": len(target_profile),
        "best_target_profile_positive_ev": target_profile[0] if target_profile else {},
        "model_warning": (
            "Synthetic tick paths are not fitted historical evidence. They are "
            "scenario tests for what a real tick-level scalp process would need "
            "to look like."
        ),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def _run_trial(
    config: SyntheticTickConfig,
    account: AccountConfig,
    account_target: Decimal,
    rng: Any,
) -> tuple[SyntheticTrial, List[SyntheticTrade]]:
    sim = TopstepRuleSimulator(account)
    trades: List[SyntheticTrade] = []
    reached_mll_lock = sim.mll_locked
    best_day_pnl = money(0)
    dll_breaches = 0

    for day in range(1, config.max_days + 1):
        if day > 1:
            sim.start_next_session()
        day_start_balance = sim.closed_balance

        for _ in range(config.trades_per_day):
            if sim.day_locked:
                break
            trade = _sample_trade(config, rng)
            trades.append(trade)
            event = sim.mark_to_market_path(trade.open_pnl_path)
            reached_mll_lock = reached_mll_lock or sim.mll_locked

            if event.event_type == RuleEventType.MLL_BREACH:
                return (
                    SyntheticTrial(
                        outcome="failed_mll",
                        days=day,
                        trades=len(trades),
                        net_pnl=money(sim.closed_balance - account.starting_balance),
                        reached_mll_lock=reached_mll_lock,
                        dll_breaches=dll_breaches,
                        best_day_pnl=best_day_pnl,
                    ),
                    trades,
                )

            if event.event_type == RuleEventType.DLL_BREACH:
                dll_breaches += 1
                break

            event = sim.close_position(trade.realized_pnl)
            reached_mll_lock = reached_mll_lock or sim.mll_locked
            if event.event_type == RuleEventType.MLL_BREACH:
                return (
                    SyntheticTrial(
                        outcome="failed_mll",
                        days=day,
                        trades=len(trades),
                        net_pnl=money(sim.closed_balance - account.starting_balance),
                        reached_mll_lock=reached_mll_lock,
                        dll_breaches=dll_breaches,
                        best_day_pnl=best_day_pnl,
                    ),
                    trades,
                )

            day_pnl = money(sim.closed_balance - day_start_balance)
            if _passes_manual(config, account, account_target, sim, day, best_day_pnl, day_pnl):
                return (
                    SyntheticTrial(
                        outcome="passed",
                        days=day,
                        trades=len(trades),
                        net_pnl=money(sim.closed_balance - account.starting_balance),
                        reached_mll_lock=reached_mll_lock,
                        dll_breaches=dll_breaches,
                        best_day_pnl=max(best_day_pnl, day_pnl),
                    ),
                    trades,
                )

        day_pnl = money(sim.closed_balance - day_start_balance)
        best_day_pnl = max(best_day_pnl, day_pnl)
        if sim.status != ChallengeStatus.ACTIVE:
            break
        if sim.session_active:
            sim.end_session()
            reached_mll_lock = reached_mll_lock or sim.mll_locked

    return (
        SyntheticTrial(
            outcome="incomplete",
            days=config.max_days,
            trades=len(trades),
            net_pnl=money(sim.closed_balance - account.starting_balance),
            reached_mll_lock=reached_mll_lock,
            dll_breaches=dll_breaches,
            best_day_pnl=best_day_pnl,
        ),
        trades,
    )


def _sample_trade(config: SyntheticTickConfig, rng: Any) -> SyntheticTrade:
    if np is not None and hasattr(rng, "integers"):
        return _sample_trade_numpy(config, rng)
    return _sample_trade_python(config, rng)


def _sample_trade_numpy(config: SyntheticTickConfig, rng: Any) -> SyntheticTrade:
    steps = np.where(
        rng.random(config.max_ticks_per_trade) < float(config.favorable_tick_probability),
        1,
        -1,
    )
    path = steps.cumsum()
    target_hits = np.flatnonzero(path >= config.target_ticks)
    stop_hits = np.flatnonzero(path <= -config.stop_ticks)

    exit_index = len(path) - 1
    exit_reason = "time_exit"
    if target_hits.size and stop_hits.size:
        if target_hits[0] < stop_hits[0]:
            exit_index = int(target_hits[0])
            exit_reason = "target"
        else:
            exit_index = int(stop_hits[0])
            exit_reason = "stop"
    elif target_hits.size:
        exit_index = int(target_hits[0])
        exit_reason = "target"
    elif stop_hits.size:
        exit_index = int(stop_hits[0])
        exit_reason = "stop"

    observed = path[: exit_index + 1]
    current_ticks = int(observed[-1])
    if exit_reason == "target":
        current_ticks = config.target_ticks
    elif exit_reason == "stop":
        current_ticks = -config.stop_ticks

    min_ticks = min(0, int(observed.min()))
    max_ticks = max(0, int(observed.max()))
    return _make_trade_from_tick_stats(
        config=config,
        current_ticks=current_ticks,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
        ticks_held=int(exit_index + 1),
        exit_reason=exit_reason,
    )


def _sample_trade_python(config: SyntheticTickConfig, rng: random.Random) -> SyntheticTrade:
    current_ticks = 0
    min_ticks = 0
    max_ticks = 0
    ticks_held = 0
    exit_reason = "time_exit"
    p = float(config.favorable_tick_probability)

    for _ in range(config.max_ticks_per_trade):
        ticks_held += 1
        current_ticks += 1 if rng.random() < p else -1
        min_ticks = min(min_ticks, current_ticks)
        max_ticks = max(max_ticks, current_ticks)
        if current_ticks >= config.target_ticks:
            current_ticks = config.target_ticks
            max_ticks = max(max_ticks, current_ticks)
            exit_reason = "target"
            break
        if current_ticks <= -config.stop_ticks:
            current_ticks = -config.stop_ticks
            min_ticks = min(min_ticks, current_ticks)
            exit_reason = "stop"
            break

    if ticks_held == 0:
        ticks_held = 1

    return _make_trade_from_tick_stats(
        config=config,
        current_ticks=current_ticks,
        min_ticks=min_ticks,
        max_ticks=max_ticks,
        ticks_held=ticks_held,
        exit_reason=exit_reason,
    )


def _make_trade_from_tick_stats(
    *,
    config: SyntheticTickConfig,
    current_ticks: int,
    min_ticks: int,
    max_ticks: int,
    ticks_held: int,
    exit_reason: str,
) -> SyntheticTrade:
    mae_pnl = money(
        min_ticks * config.tick_value * config.quantity - config.entry_side_cost
    )
    mfe_pnl = money(
        max_ticks * config.tick_value * config.quantity - config.entry_side_cost
    )
    realized_pnl = money(
        current_ticks * config.tick_value * config.quantity
        - config.total_round_turn_cost
    )
    return SyntheticTrade(
        terminal_ticks=current_ticks,
        open_pnl_path=(mae_pnl,),
        realized_pnl=realized_pnl,
        mae_pnl=mae_pnl,
        mfe_pnl=mfe_pnl,
        ticks_held=ticks_held,
        exit_reason=exit_reason,
    )


def _passes_manual(
    config: SyntheticTickConfig,
    account: AccountConfig,
    account_target: Decimal,
    sim: TopstepRuleSimulator,
    day: int,
    best_day_pnl: Decimal,
    current_day_pnl: Decimal,
) -> bool:
    if day < config.min_trading_days:
        return False
    total_profit = money(sim.closed_balance - account.starting_balance)
    if total_profit < account_target:
        return False
    if not config.include_consistency:
        return True
    candidate_best_day = max(best_day_pnl, current_day_pnl)
    required_profit = max(account_target, money(candidate_best_day / config.consistency_fraction))
    return total_profit >= required_profit


def _manual_pass_account(tier: str, use_daily_loss_limit: bool) -> AccountConfig:
    base = get_account_config(tier)
    return AccountConfig(
        name=base.name,
        starting_balance=base.starting_balance,
        # Prevent TopstepRuleSimulator from declaring pass before the manual
        # consistency/min-day gate has been checked.
        profit_target=Decimal("1000000000"),
        mll_distance=base.mll_distance,
        daily_loss_limit=base.daily_loss_limit if use_daily_loss_limit else None,
        max_contracts=base.max_contracts,
        max_micro_contracts=base.max_micro_contracts,
    )


def _points_to_ticks(points: Decimal, tick_size: Decimal) -> int:
    raw = Decimal(points) / tick_size
    ticks = raw.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    if ticks <= 0:
        raise ValueError("points must convert to at least one tick")
    return int(ticks)


def _ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def _mean_int(values: Sequence[int]) -> Decimal:
    if not values:
        return Decimal("0.00")
    return Decimal(str(mean(values))).quantize(Decimal("0.01"))


def _mean_money(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return money(0)
    return money(sum(values, Decimal("0")) / Decimal(len(values)))


def _stddev_money(values: Sequence[Decimal]) -> Decimal:
    if len(values) < 2:
        return money(0)
    avg = sum(values, Decimal("0")) / Decimal(len(values))
    variance = sum((value - avg) ** 2 for value in values) / Decimal(len(values))
    return money(variance.sqrt())


def _write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _parse_decimal_list(value: str) -> List[Decimal]:
    return [Decimal(part.strip()) for part in value.split(",") if part.strip()]


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="topstep_synthetic_tick_scalp")
    parser.add_argument("--account", default="50K")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument("--stop-points", default="10,15,20,25,30")
    parser.add_argument("--reward-risk-ratios", default="0.4,0.5,0.6,0.7")
    parser.add_argument(
        "--favorable-tick-probabilities",
        default="0.5000,0.5005,0.5010,0.5020,0.5030",
    )
    parser.add_argument("--slippage-ticks-per-side", default="0")
    parser.add_argument("--trades-per-day", type=int, default=3)
    parser.add_argument("--max-days", type=int, default=250)
    parser.add_argument("--max-ticks-per-trade", type=int, default=2000)
    parser.add_argument("--quantity", type=int, default=1)
    parser.add_argument("--trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument(
        "--round-turn-cost",
        default="",
        help="Override round-turn cost; default uses Topstep product cost for the symbol.",
    )
    parser.add_argument("--no-consistency", action="store_true")
    parser.add_argument("--min-trading-days", type=int, default=2)
    parser.add_argument("--use-daily-loss-limit", action="store_true")
    args = parser.parse_args(argv)

    round_turn_cost = Decimal(args.round_turn_cost) if args.round_turn_cost else None
    rows = run_sweep(
        account_tier=args.account,
        symbol=args.symbol,
        stop_points=_parse_decimal_list(args.stop_points),
        reward_risk_ratios=_parse_decimal_list(args.reward_risk_ratios),
        favorable_tick_probabilities=_parse_decimal_list(
            args.favorable_tick_probabilities
        ),
        slippage_ticks_per_side=_parse_decimal_list(args.slippage_ticks_per_side),
        trades_per_day=args.trades_per_day,
        max_days=args.max_days,
        max_ticks_per_trade=args.max_ticks_per_trade,
        quantity=args.quantity,
        trials=args.trials,
        seed=args.seed,
        round_turn_cost=round_turn_cost,
        include_consistency=not args.no_consistency,
        min_trading_days=args.min_trading_days,
        use_daily_loss_limit=args.use_daily_loss_limit,
    )
    write_outputs(rows, args.output_dir)
    print(
        json.dumps(
            {
                "output_dir": args.output_dir,
                "rows": len(rows),
                "trials_per_row": args.trials,
                "best": rows[0].to_row() if rows else {},
                "top": [row.to_row() for row in rows[:10]],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
