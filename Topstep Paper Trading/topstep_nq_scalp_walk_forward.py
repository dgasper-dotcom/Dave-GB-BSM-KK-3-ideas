"""Walk-forward optimize the NQ Topstep challenge scalp.

This script optimizes the challenge strategy parameters on a rolling training
window and evaluates the selected candidates on the next out-of-sample window.
It is intended for 1-minute OHLCV research. Tick/L1 validation should use the
same fold structure once the tick importer exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional, Sequence

from topstep_data_loader import (
    FuturesBar,
    detect_csv_schema,
    iter_futures_bars,
    parse_timestamp,
)
from topstep_execution_adapter import topstep_commission_per_side
from topstep_nq_strategy_runner import (
    OpeningRangeBreakoutConfig,
    OpeningRangeBreakoutRunner,
    parse_time,
    session_date_for_timestamp,
)
from topstep_rule_simulator import money


NQ_TICK_SIZE = Decimal("0.25")
RTH_START = time(9, 30)
RTH_END = time(16, 0)


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: int
    strategy_family: str
    account_tier: str
    data_symbol: str
    contract_symbol: str
    quantity: int
    opening_range_minutes: int
    stop_points: Decimal
    target_points: Decimal
    reward_risk_ratio: Decimal
    breakout_buffer_points: Decimal
    max_hold_minutes: int
    max_trades_per_session: int
    last_entry_time: time
    force_exit_time: time
    commission_per_contract: Decimal
    max_opening_range_points: Optional[Decimal]
    max_opening_gap_points: Optional[Decimal]
    pre_lock_filter_mode: str
    pre_lock_min_mll_buffer: Optional[Decimal]
    pre_lock_max_opening_range_points: Optional[Decimal]

    def to_config(self) -> OpeningRangeBreakoutConfig:
        return OpeningRangeBreakoutConfig(
            strategy_family=self.strategy_family,
            account_tier=self.account_tier,
            data_symbol=self.data_symbol,
            contract_symbol=self.contract_symbol,
            quantity=self.quantity,
            opening_range_minutes=self.opening_range_minutes,
            stop_points=self.stop_points,
            target_points=self.target_points,
            breakout_buffer_points=self.breakout_buffer_points,
            max_hold_minutes=self.max_hold_minutes,
            max_trades_per_session=self.max_trades_per_session,
            last_entry_time=self.last_entry_time,
            force_exit_time=self.force_exit_time,
            commission_per_contract=self.commission_per_contract,
            max_opening_range_points=self.max_opening_range_points,
            max_opening_gap_points=self.max_opening_gap_points,
            pre_lock_filter_mode=self.pre_lock_filter_mode,
            pre_lock_min_mll_buffer=self.pre_lock_min_mll_buffer,
            pre_lock_max_opening_range_points=self.pre_lock_max_opening_range_points,
        )

    def param_row(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "strategy_family": self.strategy_family,
            "account_tier": self.account_tier,
            "data_symbol": self.data_symbol,
            "contract_symbol": self.contract_symbol,
            "quantity": self.quantity,
            "opening_range_minutes": self.opening_range_minutes,
            "stop_points": str(self.stop_points),
            "target_points": str(self.target_points),
            "reward_risk_ratio": str(self.reward_risk_ratio),
            "breakout_buffer_points": str(self.breakout_buffer_points),
            "max_hold_minutes": self.max_hold_minutes,
            "max_trades_per_session": self.max_trades_per_session,
            "last_entry_time": self.last_entry_time.strftime("%H:%M"),
            "force_exit_time": self.force_exit_time.strftime("%H:%M"),
            "commission_per_side": str(self.commission_per_contract),
            "max_opening_range_points": _optional_decimal_str(
                self.max_opening_range_points
            ),
            "max_opening_gap_points": _optional_decimal_str(
                self.max_opening_gap_points
            ),
            "pre_lock_filter_mode": self.pre_lock_filter_mode,
            "pre_lock_min_mll_buffer": _optional_decimal_str(
                self.pre_lock_min_mll_buffer
            ),
            "pre_lock_max_opening_range_points": _optional_decimal_str(
                self.pre_lock_max_opening_range_points
            ),
        }


def parse_decimal_grid(value: str, *, allow_none: bool = False) -> list[Optional[Decimal]]:
    items: list[Optional[Decimal]] = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        if allow_none and item.lower() in {"none", "null", "off"}:
            items.append(None)
        else:
            items.append(Decimal(item))
    if not items:
        raise ValueError("decimal grid cannot be empty")
    return items


def parse_int_grid(value: str) -> list[int]:
    items = [int(raw.strip()) for raw in value.split(",") if raw.strip()]
    if not items:
        raise ValueError("integer grid cannot be empty")
    return items


def parse_time_grid(value: str) -> list[time]:
    items = [parse_time(raw.strip()) for raw in value.split(",") if raw.strip()]
    if not items:
        raise ValueError("time grid cannot be empty")
    return items


def build_candidates(args: argparse.Namespace) -> list[StrategyCandidate]:
    commission_per_side = (
        money(args.commission_per_side)
        if args.commission_per_side
        else topstep_commission_per_side(args.contract)
    )
    explicit_targets = (
        parse_decimal_grid(args.target_points) if args.target_points else None
    )
    reward_risk_ratios = parse_decimal_grid(args.reward_risk_ratios)
    candidates: list[StrategyCandidate] = []
    candidate_id = 0

    for quantity in parse_int_grid(args.quantities):
        for opening_range_minutes in parse_int_grid(args.opening_range_minutes):
            for stop_points in parse_decimal_grid(args.stop_points):
                target_values = explicit_targets or [
                    round_to_tick(stop_points * rr, NQ_TICK_SIZE)
                    for rr in reward_risk_ratios
                ]
                for target_points in target_values:
                    reward_risk_ratio = (target_points / stop_points).quantize(
                        Decimal("0.0001")
                    )
                    for breakout_buffer_points in parse_decimal_grid(
                        args.breakout_buffer_points
                    ):
                        for max_hold_minutes in parse_int_grid(args.max_hold_minutes):
                            for max_trades_per_session in parse_int_grid(
                                args.max_trades_per_session
                            ):
                                for last_entry_time in parse_time_grid(
                                    args.last_entry_times
                                ):
                                    for max_opening_range_points in parse_decimal_grid(
                                        args.max_opening_range_points, allow_none=True
                                    ):
                                        for max_opening_gap_points in parse_decimal_grid(
                                            args.max_opening_gap_points,
                                            allow_none=True,
                                        ):
                                            for pre_lock_min_mll_buffer in (
                                                parse_decimal_grid(
                                                    args.pre_lock_min_mll_buffer,
                                                    allow_none=True,
                                                )
                                            ):
                                                for pre_lock_max_or in (
                                                    parse_decimal_grid(
                                                        args.pre_lock_max_opening_range_points,
                                                        allow_none=True,
                                                    )
                                                ):
                                                    candidate_id += 1
                                                    candidates.append(
                                                        StrategyCandidate(
                                                            candidate_id=candidate_id,
                                                            strategy_family=args.strategy_family,
                                                            account_tier=args.account,
                                                            data_symbol=args.symbol,
                                                            contract_symbol=args.contract,
                                                            quantity=quantity,
                                                            opening_range_minutes=opening_range_minutes,
                                                            stop_points=stop_points,
                                                            target_points=target_points,
                                                            reward_risk_ratio=reward_risk_ratio,
                                                            breakout_buffer_points=breakout_buffer_points,
                                                            max_hold_minutes=max_hold_minutes,
                                                            max_trades_per_session=max_trades_per_session,
                                                            last_entry_time=last_entry_time,
                                                            force_exit_time=parse_time(
                                                                args.force_exit_time
                                                            ),
                                                            commission_per_contract=commission_per_side,
                                                            max_opening_range_points=max_opening_range_points,
                                                            max_opening_gap_points=max_opening_gap_points,
                                                            pre_lock_filter_mode=args.pre_lock_filter_mode,
                                                            pre_lock_min_mll_buffer=pre_lock_min_mll_buffer,
                                                            pre_lock_max_opening_range_points=pre_lock_max_or,
                                                        )
                                                    )
    if not candidates:
        raise ValueError("parameter grid produced no candidates")
    return candidates


def load_session_dates(
    data_path: str,
    symbol: str,
    *,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> list[date]:
    sessions: list[date] = []
    seen: set[date] = set()
    for bar in iter_futures_bars(data_path, symbol=symbol, start=start, end=end):
        session = session_date_for_timestamp(bar.timestamp)
        if session in seen:
            continue
        seen.add(session)
        sessions.append(session)
    if not sessions:
        raise ValueError("no sessions found in data range")
    return sessions


def contiguous_chunks(items: Sequence[date], chunks: int) -> list[list[date]]:
    if chunks <= 1:
        raise ValueError("folds must be greater than 1")
    if len(items) < chunks:
        raise ValueError("not enough sessions for requested folds")
    out: list[list[date]] = []
    for index in range(chunks):
        start = index * len(items) // chunks
        end = (index + 1) * len(items) // chunks
        out.append(list(items[start:end]))
    return out


def session_range_bounds(sessions: Sequence[date]) -> tuple[datetime, datetime]:
    if not sessions:
        raise ValueError("session range cannot be empty")
    start = datetime.combine(sessions[0] - timedelta(days=1), time(18, 0))
    end = datetime.combine(sessions[-1], time(17, 0))
    return start, end


def iter_research_bars(
    args: argparse.Namespace,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
) -> Iterator[FuturesBar]:
    schema = detect_csv_schema(Path(args.data))
    source = iter_futures_bars(
        args.data,
        symbol=args.symbol,
        schema=schema,
        start=start,
        end=end,
    )
    use_csv_vwap = args.vwap_mode == "csv" or (
        args.vwap_mode == "auto" and schema.vwap_rth_col is not None
    )
    if use_csv_vwap:
        yield from source
        return
    yield from compute_rth_vwap(source)


def compute_rth_vwap(bars: Iterable[FuturesBar]) -> Iterator[FuturesBar]:
    current_session: Optional[date] = None
    numerator = Decimal("0")
    denominator = Decimal("0")

    for bar in bars:
        session = session_date_for_timestamp(bar.timestamp)
        if session != current_session:
            current_session = session
            numerator = Decimal("0")
            denominator = Decimal("0")

        vwap_rth = bar.vwap_rth
        if RTH_START <= bar.timestamp.time() <= RTH_END:
            volume = bar.volume if bar.volume > 0 else Decimal("1")
            typical_price = (bar.high + bar.low + bar.close) / Decimal("3")
            numerator += typical_price * volume
            denominator += volume
            if denominator > 0:
                vwap_rth = numerator / denominator

        yield FuturesBar(
            timestamp=bar.timestamp,
            symbol=bar.symbol,
            open=bar.open,
            high=bar.high,
            low=bar.low,
            close=bar.close,
            volume=bar.volume,
            vwap_rth=vwap_rth,
            vwap_eth=bar.vwap_eth,
        )


def evaluate_candidate(
    args: argparse.Namespace,
    candidate: StrategyCandidate,
    *,
    start: Optional[datetime],
    end: Optional[datetime],
) -> dict[str, Any]:
    runner = OpeningRangeBreakoutRunner(candidate.to_config())
    result = runner.run_bars(
        iter_research_bars(args, start=start, end=end),
        max_rows=args.max_rows,
    )
    return normalize_summary(candidate, result.summary())


def normalize_summary(
    candidate: StrategyCandidate,
    summary: dict[str, Any],
) -> dict[str, Any]:
    fields = {
        "rows_processed": summary["rows_processed"],
        "attempts": summary["attempts"],
        "completed_attempts": summary["completed_attempts"],
        "passes": summary["passes"],
        "failures": summary["failures"],
        "pass_rate": summary["pass_rate"],
        "trade_count": summary["trade_count"],
        "pre_lock_trade_count": summary["pre_lock_trade_count"],
        "post_lock_trade_count": summary["post_lock_trade_count"],
        "win_rate": summary["win_rate"],
        "avg_trade_pnl": summary["avg_trade_pnl"],
        "total_trade_pnl": summary["total_trade_pnl"],
        "avg_mae_pnl": summary["avg_mae_pnl"],
        "avg_mfe_pnl": summary["avg_mfe_pnl"],
        "avg_time_underwater_minutes": summary["avg_time_underwater_minutes"],
        "avg_time_to_target_minutes": summary["avg_time_to_target_minutes"],
        "winners_with_adverse_excursion_rate": summary[
            "winners_with_adverse_excursion_rate"
        ],
        "avg_winner_mae_pnl": summary["avg_winner_mae_pnl"],
        "avg_days_to_pass": summary["avg_days_to_pass"],
        "dll_breaches": summary["dll_breaches"],
        "mll_lock_reached": summary["mll_lock_reached"],
        "failure_reasons": json.dumps(summary["failure_reasons"], sort_keys=True),
        "risk_filter_skips": json.dumps(summary["risk_filter_skips"], sort_keys=True),
    }
    return {**candidate.param_row(), **fields}


def candidate_rank(args: argparse.Namespace, row: dict[str, Any]) -> tuple[Any, ...]:
    completed_attempts = int(row["completed_attempts"])
    trade_count = int(row["trade_count"])
    pass_rate = Decimal(str(row["pass_rate"]))
    win_rate = Decimal(str(row["win_rate"]))
    avg_trade_pnl = Decimal(str(row["avg_trade_pnl"]))
    avg_mae_pnl = Decimal(str(row["avg_mae_pnl"]))
    dll_breaches = int(row["dll_breaches"])

    enough_data = (
        completed_attempts >= args.min_completed_attempts
        and trade_count >= args.min_trades
    )
    positive_ev = avg_trade_pnl >= money(args.min_avg_trade_pnl)

    if args.objective == "pass_rate":
        return (
            int(enough_data),
            pass_rate,
            avg_trade_pnl,
            win_rate,
            -dll_breaches,
            avg_mae_pnl,
        )
    if args.objective == "positive_ev_pass_rate":
        return (
            int(enough_data),
            int(positive_ev),
            pass_rate,
            avg_trade_pnl,
            win_rate,
            -dll_breaches,
            avg_mae_pnl,
        )
    if args.objective == "survival_score":
        mll_failure_rate = _safe_decimal_ratio(
            int(row["failures"]), max(completed_attempts, 1)
        )
        score = (
            pass_rate * Decimal("10000")
            + win_rate * Decimal("100")
            + avg_trade_pnl
            - Decimal(dll_breaches) * Decimal("25")
            + avg_mae_pnl.copy_abs() * Decimal("-0.02")
            - mll_failure_rate * Decimal("250")
        )
        return (int(enough_data), int(positive_ev), score, pass_rate, avg_trade_pnl)
    raise ValueError(f"unsupported objective: {args.objective}")


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run_walk_forward(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    global_start = parse_timestamp(args.start) if args.start else None
    global_end = parse_timestamp(args.end) if args.end else None
    sessions = load_session_dates(
        args.data,
        args.symbol,
        start=global_start,
        end=global_end,
    )
    chunks = contiguous_chunks(sessions, args.folds)
    if args.train_folds >= args.folds:
        raise ValueError("train_folds must be less than folds")

    candidates = build_candidates(args)
    fold_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    oos_rows: list[dict[str, Any]] = []

    for fold_start in range(0, args.folds - args.train_folds):
        fold_number = fold_start + 1
        train_sessions = [
            session
            for chunk in chunks[fold_start : fold_start + args.train_folds]
            for session in chunk
        ]
        test_sessions = chunks[fold_start + args.train_folds]
        train_start, train_end = session_range_bounds(train_sessions)
        test_start, test_end = session_range_bounds(test_sessions)

        rankings = []
        for candidate in candidates:
            row = evaluate_candidate(
                args,
                candidate,
                start=train_start,
                end=train_end,
            )
            row["fold"] = fold_number
            row["sample"] = "train"
            row["train_start"] = train_sessions[0].isoformat()
            row["train_end"] = train_sessions[-1].isoformat()
            row["test_start"] = test_sessions[0].isoformat()
            row["test_end"] = test_sessions[-1].isoformat()
            row["rank_key"] = json.dumps(
                [str(item) for item in candidate_rank(args, row)]
            )
            rankings.append(row)

        rankings.sort(key=lambda row: candidate_rank(args, row), reverse=True)
        for rank, row in enumerate(rankings, start=1):
            ranked = dict(row)
            ranked["train_rank"] = rank
            train_rows.append(ranked)

        selected_train_rows = rankings[: args.oos_top_n]
        selected_test_rows = []
        for train_rank, train_row in enumerate(selected_train_rows, start=1):
            candidate = candidates[int(train_row["candidate_id"]) - 1]
            test_row = evaluate_candidate(
                args,
                candidate,
                start=test_start,
                end=test_end,
            )
            test_row["fold"] = fold_number
            test_row["sample"] = "oos"
            test_row["train_rank"] = train_rank
            test_row["train_start"] = train_sessions[0].isoformat()
            test_row["train_end"] = train_sessions[-1].isoformat()
            test_row["test_start"] = test_sessions[0].isoformat()
            test_row["test_end"] = test_sessions[-1].isoformat()
            selected_test_rows.append(test_row)
            oos_rows.append(test_row)

        best_train = rankings[0]
        best_test = selected_test_rows[0]
        fold_rows.append(
            {
                "fold": fold_number,
                "train_start": train_sessions[0].isoformat(),
                "train_end": train_sessions[-1].isoformat(),
                "test_start": test_sessions[0].isoformat(),
                "test_end": test_sessions[-1].isoformat(),
                "train_sessions": len(train_sessions),
                "test_sessions": len(test_sessions),
                **prefixed("selected_train", best_train),
                **prefixed("selected_oos", best_test),
            }
        )

    leaderboard = aggregate_oos_leaderboard(oos_rows)
    summary = build_summary(args, sessions, candidates, fold_rows, oos_rows, leaderboard)
    return fold_rows, train_rows, oos_rows, leaderboard, summary


def aggregate_oos_leaderboard(oos_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in oos_rows:
        grouped.setdefault(str(row["candidate_id"]), []).append(row)

    leaderboard: list[dict[str, Any]] = []
    for candidate_id, rows in grouped.items():
        first = rows[0]
        pass_rates = [Decimal(str(row["pass_rate"])) for row in rows]
        win_rates = [Decimal(str(row["win_rate"])) for row in rows]
        avg_trade_pnls = [Decimal(str(row["avg_trade_pnl"])) for row in rows]
        completed = sum(int(row["completed_attempts"]) for row in rows)
        passes = sum(int(row["passes"]) for row in rows)
        failures = sum(int(row["failures"]) for row in rows)
        trades = sum(int(row["trade_count"]) for row in rows)
        aggregate = {
            **{
                key: first[key]
                for key in first
                if key
                in {
                    "candidate_id",
                    "strategy_family",
                    "account_tier",
                    "data_symbol",
                    "contract_symbol",
                    "quantity",
                    "opening_range_minutes",
                    "stop_points",
                    "target_points",
                    "reward_risk_ratio",
                    "breakout_buffer_points",
                    "max_hold_minutes",
                    "max_trades_per_session",
                    "last_entry_time",
                    "force_exit_time",
                    "commission_per_side",
                    "max_opening_range_points",
                    "max_opening_gap_points",
                    "pre_lock_filter_mode",
                    "pre_lock_min_mll_buffer",
                    "pre_lock_max_opening_range_points",
                }
            },
            "oos_evaluated_folds": len(rows),
            "oos_completed_attempts": completed,
            "oos_passes": passes,
            "oos_failures": failures,
            "oos_trade_count": trades,
            "oos_aggregate_pass_rate": str(_safe_decimal_ratio(passes, completed)),
            "oos_avg_fold_pass_rate": str(_mean_decimal(pass_rates)),
            "oos_avg_win_rate": str(_mean_decimal(win_rates)),
            "oos_avg_trade_pnl": str(money(sum(avg_trade_pnls) / Decimal(len(rows)))),
            "oos_positive_ev_folds": sum(
                1 for value in avg_trade_pnls if value >= Decimal("0")
            ),
            "oos_positive_pass_folds": sum(1 for value in pass_rates if value > 0),
        }
        leaderboard.append(aggregate)

    leaderboard.sort(
        key=lambda row: (
            int(row["oos_positive_ev_folds"]),
            Decimal(str(row["oos_aggregate_pass_rate"])),
            Decimal(str(row["oos_avg_trade_pnl"])),
            int(row["oos_trade_count"]),
        ),
        reverse=True,
    )
    return leaderboard


def build_summary(
    args: argparse.Namespace,
    sessions: Sequence[date],
    candidates: Sequence[StrategyCandidate],
    fold_rows: Sequence[dict[str, Any]],
    oos_rows: Sequence[dict[str, Any]],
    leaderboard: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    selected_oos_pass_rates = [
        Decimal(str(row["selected_oos_pass_rate"])) for row in fold_rows
    ]
    selected_oos_avg_trade_pnl = [
        Decimal(str(row["selected_oos_avg_trade_pnl"])) for row in fold_rows
    ]
    return {
        "folds_run": len(fold_rows),
        "sessions": len(sessions),
        "first_session": sessions[0].isoformat(),
        "last_session": sessions[-1].isoformat(),
        "candidate_count": len(candidates),
        "oos_evaluations": len(oos_rows),
        "selected_avg_oos_pass_rate": str(_mean_decimal(selected_oos_pass_rates)),
        "selected_worst_oos_pass_rate": str(min(selected_oos_pass_rates)),
        "selected_best_oos_pass_rate": str(max(selected_oos_pass_rates)),
        "selected_positive_ev_folds": sum(
            1 for value in selected_oos_avg_trade_pnl if value >= Decimal("0")
        ),
        "best_oos_candidate": leaderboard[0] if leaderboard else None,
        "metadata": {
            "data": args.data,
            "symbol": args.symbol,
            "contract": args.contract,
            "strategy_family": args.strategy_family,
            "account": args.account,
            "folds": args.folds,
            "train_folds": args.train_folds,
            "oos_top_n": args.oos_top_n,
            "objective": args.objective,
            "min_completed_attempts": args.min_completed_attempts,
            "min_trades": args.min_trades,
            "min_avg_trade_pnl": str(money(args.min_avg_trade_pnl)),
            "vwap_mode": args.vwap_mode,
            "max_rows": args.max_rows,
        },
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    args: argparse.Namespace,
    fold_rows: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, Any]],
    oos_rows: Sequence[dict[str, Any]],
    leaderboard: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _write_csv(out / "folds.csv", fold_rows)
    if not args.skip_train_rankings:
        _write_csv(out / "train_rankings.csv", train_rows)
    _write_csv(out / "oos_evaluations.csv", oos_rows)
    _write_csv(out / "leaderboard.csv", leaderboard)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def round_to_tick(value: Decimal, tick: Decimal) -> Decimal:
    ticks = (Decimal(value) / tick).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return ticks * tick


def _mean_decimal(values: Sequence[Decimal]) -> Decimal:
    if not values:
        return Decimal("0.0000")
    return (sum(values, Decimal("0")) / Decimal(len(values))).quantize(
        Decimal("0.0001")
    )


def _safe_decimal_ratio(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def _optional_decimal_str(value: Optional[Decimal]) -> str:
    return "" if value is None else str(value)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", default="topstep_nq_scalp_walk_forward")
    parser.add_argument("--symbol", default="NQ")
    parser.add_argument("--contract", default="NQ", choices=["NQ", "MNQ"])
    parser.add_argument("--account", default="50K")
    parser.add_argument(
        "--strategy-family",
        default="scalp_reversion",
        choices=["scalp_reversion", "vwap_reversion", "orb_fade", "orb_breakout"],
    )
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--train-folds", type=int, default=3)
    parser.add_argument("--oos-top-n", type=int, default=5)
    parser.add_argument(
        "--objective",
        default="positive_ev_pass_rate",
        choices=["positive_ev_pass_rate", "pass_rate", "survival_score"],
    )
    parser.add_argument("--min-completed-attempts", type=int, default=3)
    parser.add_argument("--min-trades", type=int, default=20)
    parser.add_argument("--min-avg-trade-pnl", default="0")
    parser.add_argument("--quantities", default="1")
    parser.add_argument("--opening-range-minutes", default="5")
    parser.add_argument("--stop-points", default="16,20,24")
    parser.add_argument(
        "--reward-risk-ratios",
        default="0.5,0.6,0.7",
        help="Used to derive targets when --target-points is omitted.",
    )
    parser.add_argument(
        "--target-points",
        help="Explicit comma grid. If set, overrides --reward-risk-ratios.",
    )
    parser.add_argument("--breakout-buffer-points", default="8,10,12,15")
    parser.add_argument("--max-hold-minutes", default="5,10")
    parser.add_argument("--max-trades-per-session", default="1,2,3")
    parser.add_argument("--last-entry-times", default="11:30")
    parser.add_argument("--force-exit-time", default="16:00")
    parser.add_argument(
        "--max-opening-range-points",
        default="none",
        help="Comma grid. Use none to disable.",
    )
    parser.add_argument(
        "--max-opening-gap-points",
        default="none",
        help="Comma grid. Use none to disable.",
    )
    parser.add_argument(
        "--pre-lock-filter-mode",
        default="any",
        choices=["any", "combined"],
    )
    parser.add_argument(
        "--pre-lock-min-mll-buffer",
        default="none",
        help="Comma grid. Use none to disable.",
    )
    parser.add_argument(
        "--pre-lock-max-opening-range-points",
        default="none",
        help="Comma grid. Use none to disable.",
    )
    parser.add_argument(
        "--commission-per-side",
        help="Override per-side cost. Default uses Topstep round turn split per side.",
    )
    parser.add_argument(
        "--vwap-mode",
        default="auto",
        choices=["auto", "csv", "compute"],
        help="auto uses CSV Vwap_RTH when present, otherwise computes RTH VWAP.",
    )
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--skip-train-rankings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fold_rows, train_rows, oos_rows, leaderboard, summary = run_walk_forward(args)
    write_outputs(args, fold_rows, train_rows, oos_rows, leaderboard, summary)
    print(json.dumps({"output_dir": args.output_dir, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
