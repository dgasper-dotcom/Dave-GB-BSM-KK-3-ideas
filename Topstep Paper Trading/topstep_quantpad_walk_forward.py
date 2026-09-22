"""Walk-forward validate QuantPad strategy sweeps.

This runner optimizes a controlled parameter grid on a rolling training window
and evaluates the selected candidate on the next out-of-sample window. It is
designed to reduce dependence on one full-sample leaderboard.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional, Sequence

import topstep_quantpad_backtest as qpbt
from topstep_rule_simulator import money


def decimal(value: Any) -> Decimal:
    if value in {None, ""}:
        return Decimal("0")
    return Decimal(str(value))


def ratio(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0.0000")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.0001"))


def q4(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal("0.0001"))


def wilson_lower_bound(successes: int, total: int, z: float) -> Decimal:
    if total <= 0:
        return Decimal("0.0000")
    phat = successes / total
    z2 = z * z
    denominator = 1 + z2 / total
    center = phat + z2 / (2 * total)
    margin = z * math.sqrt(phat * (1 - phat) / total + z2 / (4 * total * total))
    return q4(max(0.0, (center - margin) / denominator))


def parse_local_date(value: str) -> date:
    if "T" in value:
        return datetime.fromisoformat(value).date()
    return date.fromisoformat(value)


def date_windows(start: date, end: date, folds: int) -> list[tuple[date, date]]:
    if end <= start:
        raise ValueError("end must be after start")
    if folds <= 1:
        raise ValueError("folds must be greater than 1")
    total_days = (end - start).days
    if total_days < folds:
        raise ValueError("date range is too short for requested folds")
    windows = []
    for index in range(folds):
        window_start = start + timedelta(days=total_days * index // folds)
        window_end = start + timedelta(days=total_days * (index + 1) // folds)
        if window_end <= window_start:
            raise ValueError("fold window collapsed; use fewer folds")
        windows.append((window_start, window_end))
    return windows


def backtest_args(args: argparse.Namespace, start: date, end: date) -> argparse.Namespace:
    argv = [
        "--start",
        start.isoformat(),
        "--end",
        end.isoformat(),
        "--timeframe",
        args.timeframe,
        "--chunk-days",
        str(args.chunk_days),
        "--account",
        args.account,
        "--sweep",
        "--skip-xfa",
        "--quick-pass-days",
        str(args.quick_pass_days),
        "--slippage-ticks-per-side",
        args.slippage_ticks_per_side,
        "--sweep-symbols",
        args.sweep_symbols,
        "--sweep-families",
        args.sweep_families,
        "--sweep-quantities",
        args.sweep_quantities,
        "--sweep-max-trades-per-day",
        args.sweep_max_trades_per_day,
        "--sweep-max-hold-minutes",
        args.sweep_max_hold_minutes,
        "--sweep-opening-range-minutes",
        args.sweep_opening_range_minutes,
        "--sweep-last-entry-times",
        args.sweep_last_entry_times,
        "--sweep-max-opening-range-points",
        args.sweep_max_opening_range_points,
        "--sweep-max-entry-bar-range-points",
        args.sweep_max_entry_bar_range_points,
        "--sweep-nq-stop-points",
        args.sweep_nq_stop_points,
        "--sweep-nq-target-points",
        args.sweep_nq_target_points,
        "--sweep-nq-threshold-points",
        args.sweep_nq_threshold_points,
        "--sweep-nq-confirm-points",
        args.sweep_nq_confirm_points,
        "--sweep-es-stop-points",
        args.sweep_es_stop_points,
        "--sweep-es-target-points",
        args.sweep_es_target_points,
        "--sweep-es-threshold-points",
        args.sweep_es_threshold_points,
        "--sweep-es-confirm-points",
        args.sweep_es_confirm_points,
    ]
    if args.round_turn_cost:
        argv.extend(["--round-turn-cost", args.round_turn_cost])
    return qpbt.parse_args(argv)


def annotate_row(args: argparse.Namespace, row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    completed = int(out["completed_attempts"])
    passes = int(out["passes"])
    quick_passes = int(
        decimal(out.get("quick_passes", "0")).to_integral_value(
            rounding=ROUND_HALF_UP
        )
    )
    if not quick_passes and completed:
        quick_passes = int(
            (decimal(out["quick_pass_rate"]) * completed).to_integral_value(
                rounding=ROUND_HALF_UP
            )
        )
    out["quick_passes"] = quick_passes
    out["pass_lcb"] = str(wilson_lower_bound(passes, completed, args.confidence_z))
    out["quick_pass_lcb"] = str(
        wilson_lower_bound(quick_passes, completed, args.confidence_z)
    )
    out["eligible"] = int(
        completed >= args.min_completed_attempts
        and int(out["trade_count"]) >= args.min_trades
        and decimal(out["avg_trade_pnl"]) >= decimal(args.min_avg_trade_pnl)
    )
    out["selection_score"] = json.dumps([str(item) for item in rank_key(out)])
    return out


def rank_key(row: dict[str, Any]) -> tuple[Any, ...]:
    mae = abs(decimal(row["avg_mae_pnl"]))
    return (
        int(row["eligible"]),
        decimal(row["quick_pass_lcb"]),
        decimal(row["pass_lcb"]),
        decimal(row["quick_pass_rate"]),
        decimal(row["pass_rate"]),
        decimal(row["avg_trade_pnl"]),
        -mae,
        int(row["completed_attempts"]),
    )


def run_period(
    args: argparse.Namespace,
    *,
    fold: int,
    sample: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    print(
        json.dumps(
            {
                "event": "walk_forward_period",
                "fold": fold,
                "sample": sample,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
        ),
        flush=True,
    )
    _, summaries = qpbt.run_backtest(backtest_args(args, start, end))
    rows = []
    for row in summaries:
        annotated = annotate_row(args, row)
        annotated["fold"] = fold
        annotated["sample"] = sample
        annotated["period_start"] = start.isoformat()
        annotated["period_end"] = end.isoformat()
        rows.append(annotated)
    rows.sort(key=rank_key, reverse=True)
    return rows


def prefixed(prefix: str, row: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in row.items()}


def aggregate_candidate_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["strategy"]].append(row)

    leaderboard = []
    for strategy, group in grouped.items():
        first = group[0]
        completed = sum(int(row["completed_attempts"]) for row in group)
        passes = sum(int(row["passes"]) for row in group)
        quick_passes = sum(int(row["quick_passes"]) for row in group)
        trades = sum(int(row["trade_count"]) for row in group)
        total_pnl = sum(decimal(row["total_trade_pnl"]) for row in group)
        mae_weighted = sum(
            decimal(row["avg_mae_pnl"]) * int(row["trade_count"]) for row in group
        )
        mfe_weighted = sum(
            decimal(row["avg_mfe_pnl"]) * int(row["trade_count"]) for row in group
        )
        avg_trade_pnl = money(total_pnl / Decimal(trades)) if trades else money(0)
        avg_mae_pnl = money(mae_weighted / Decimal(trades)) if trades else money(0)
        avg_mfe_pnl = money(mfe_weighted / Decimal(trades)) if trades else money(0)
        quick_rate = ratio(quick_passes, completed)
        pass_rate = ratio(passes, completed)
        row = {
            "strategy": strategy,
            "symbol": first["symbol"],
            "family": first["family"],
            "quantity": first["quantity"],
            "stop_points": first["stop_points"],
            "target_points": first["target_points"],
            "reward_risk_ratio": first["reward_risk_ratio"],
            "threshold_points": first["threshold_points"],
            "max_opening_range_points": first.get("max_opening_range_points", ""),
            "max_entry_bar_range_points": first.get("max_entry_bar_range_points", ""),
            "oos_evaluated_folds": len(group),
            "oos_completed_attempts": completed,
            "oos_passes": passes,
            "oos_quick_passes": quick_passes,
            "oos_failures": sum(int(row["failures"]) for row in group),
            "oos_pass_rate": str(pass_rate),
            "oos_quick_pass_rate": str(quick_rate),
            "oos_pass_lcb": str(
                wilson_lower_bound(passes, completed, 1.0)
            ),
            "oos_quick_pass_lcb": str(
                wilson_lower_bound(quick_passes, completed, 1.0)
            ),
            "oos_trade_count": trades,
            "oos_avg_trade_pnl": str(avg_trade_pnl),
            "oos_avg_mae_pnl": str(avg_mae_pnl),
            "oos_avg_mfe_pnl": str(avg_mfe_pnl),
            "oos_positive_trade_pnl_folds": sum(
                1 for row in group if decimal(row["avg_trade_pnl"]) >= 0
            ),
            "oos_positive_pass_folds": sum(
                1 for row in group if decimal(row["pass_rate"]) > 0
            ),
            "oos_positive_quick_pass_folds": sum(
                1 for row in group if decimal(row["quick_pass_rate"]) > 0
            ),
        }
        leaderboard.append(row)

    leaderboard.sort(
        key=lambda row: (
            decimal(row["oos_quick_pass_lcb"]),
            decimal(row["oos_pass_lcb"]),
            decimal(row["oos_quick_pass_rate"]),
            decimal(row["oos_avg_trade_pnl"]),
            int(row["oos_positive_trade_pnl_folds"]),
        ),
        reverse=True,
    )
    return leaderboard


def aggregate_rows(label: str, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = sum(int(row["completed_attempts"]) for row in rows)
    passes = sum(int(row["passes"]) for row in rows)
    quick_passes = sum(int(row["quick_passes"]) for row in rows)
    trades = sum(int(row["trade_count"]) for row in rows)
    total_pnl = sum(decimal(row["total_trade_pnl"]) for row in rows)
    avg_trade_pnl = money(total_pnl / Decimal(trades)) if trades else money(0)
    return {
        "label": label,
        "fold_rows": len(rows),
        "completed_attempts": completed,
        "passes": passes,
        "quick_passes": quick_passes,
        "pass_rate": str(ratio(passes, completed)),
        "quick_pass_rate": str(ratio(quick_passes, completed)),
        "trade_count": trades,
        "avg_trade_pnl": str(avg_trade_pnl),
        "positive_avg_trade_folds": sum(
            1 for row in rows if decimal(row["avg_trade_pnl"]) >= 0
        ),
        "positive_pass_folds": sum(
            1 for row in rows if decimal(row["pass_rate"]) > 0
        ),
    }


def neighbor_summary(best: Optional[dict[str, Any]], leaderboard: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not best:
        return {}
    neighbors = []
    for row in leaderboard:
        if row["strategy"] == best["strategy"]:
            continue
        if row["symbol"] != best["symbol"] or row["family"] != best["family"]:
            continue
        if str(row["quantity"]) != str(best["quantity"]):
            continue
        if row.get("max_opening_range_points", "") != best.get("max_opening_range_points", ""):
            continue
        stop_distance = abs(decimal(row["stop_points"]) - decimal(best["stop_points"]))
        target_distance = abs(decimal(row["target_points"]) - decimal(best["target_points"]))
        threshold_distance = abs(
            decimal(row["threshold_points"]) - decimal(best["threshold_points"])
        )
        if (
            stop_distance <= Decimal("2")
            and target_distance <= Decimal("1")
            and threshold_distance <= Decimal("1")
        ):
            neighbors.append(row)
    positive = [
        row
        for row in neighbors
        if decimal(row["oos_avg_trade_pnl"]) >= 0 and decimal(row["oos_pass_rate"]) > 0
    ]
    return {
        "best_strategy": best["strategy"],
        "neighbor_count": len(neighbors),
        "positive_neighbor_count": len(positive),
        "positive_neighbor_rate": str(ratio(len(positive), len(neighbors))),
    }


def run_walk_forward(args: argparse.Namespace) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    if args.api_key_stdin:
        os.environ["QUANTPAD_API_KEY"] = sys.stdin.readline().strip()
    elif not os.environ.get("QUANTPAD_API_KEY"):
        raise ValueError("set QUANTPAD_API_KEY or pass --api-key-stdin")

    start = parse_local_date(args.start)
    end = parse_local_date(args.end)
    windows = date_windows(start, end, args.folds)
    if args.train_folds >= args.folds:
        raise ValueError("train_folds must be less than folds")

    fold_rows: list[dict[str, Any]] = []
    train_rows: list[dict[str, Any]] = []
    selected_oos_rows: list[dict[str, Any]] = []
    all_oos_rows: list[dict[str, Any]] = []

    for fold_start in range(args.folds - args.train_folds):
        fold = fold_start + 1
        train_start = windows[fold_start][0]
        train_end = windows[fold_start + args.train_folds - 1][1]
        test_start, test_end = windows[fold_start + args.train_folds]

        train_rankings = run_period(
            args,
            fold=fold,
            sample="train",
            start=train_start,
            end=train_end,
        )
        for rank, row in enumerate(train_rankings, start=1):
            ranked = dict(row)
            ranked["train_rank"] = rank
            train_rows.append(ranked)

        test_rankings = run_period(
            args,
            fold=fold,
            sample="oos_all",
            start=test_start,
            end=test_end,
        )
        all_oos_rows.extend(test_rankings)
        test_by_strategy = {row["strategy"]: row for row in test_rankings}

        selected_rows = []
        for train_rank, selected_train in enumerate(
            train_rankings[: args.oos_top_n], start=1
        ):
            selected_test = dict(test_by_strategy[selected_train["strategy"]])
            selected_test["train_rank"] = train_rank
            selected_test["selected_by_train_strategy"] = selected_train["strategy"]
            selected_test.update(
                {
                    "train_pass_rate": selected_train["pass_rate"],
                    "train_quick_pass_rate": selected_train["quick_pass_rate"],
                    "train_quick_pass_lcb": selected_train["quick_pass_lcb"],
                    "train_avg_trade_pnl": selected_train["avg_trade_pnl"],
                    "train_completed_attempts": selected_train["completed_attempts"],
                }
            )
            selected_rows.append(selected_test)
            selected_oos_rows.append(selected_test)

        best_train = train_rankings[0]
        best_test = selected_rows[0]
        fold_rows.append(
            {
                "fold": fold,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "test_start": test_start.isoformat(),
                "test_end": test_end.isoformat(),
                **prefixed("selected_train", best_train),
                **prefixed("selected_oos", best_test),
            }
        )

    all_oos_leaderboard = aggregate_candidate_rows(all_oos_rows)
    selected_leaderboard = aggregate_candidate_rows(selected_oos_rows)
    selected_top1 = [row for row in selected_oos_rows if int(row["train_rank"]) == 1]
    selected_top1_leaderboard = aggregate_candidate_rows(selected_top1)
    best_all = all_oos_leaderboard[0] if all_oos_leaderboard else None
    best_selected = selected_top1_leaderboard[0] if selected_top1_leaderboard else None
    summary = {
        "folds_run": len(fold_rows),
        "fold_windows": [
            {"start": start.isoformat(), "end": end.isoformat()}
            for start, end in windows
        ],
        "candidate_count": len(all_oos_rows) // max(1, len(fold_rows)),
        "oos_top_n": args.oos_top_n,
        "best_all_oos_candidate": best_all,
        "best_selected_top1_candidate": best_selected,
        "selected_top1_overall": aggregate_rows("selected_top1", selected_top1),
        "selected_top_n_overall": aggregate_rows("selected_top_n", selected_oos_rows),
        "best_all_neighbor_summary": neighbor_summary(best_all, all_oos_leaderboard),
        "metadata": {
            "start": args.start,
            "end": args.end,
            "timeframe": args.timeframe,
            "folds": args.folds,
            "train_folds": args.train_folds,
            "quick_pass_days": args.quick_pass_days,
            "confidence_z": args.confidence_z,
            "min_completed_attempts": args.min_completed_attempts,
            "min_trades": args.min_trades,
            "min_avg_trade_pnl": str(money(args.min_avg_trade_pnl)),
            "sweep_symbols": args.sweep_symbols,
            "sweep_families": args.sweep_families,
            "sweep_quantities": args.sweep_quantities,
            "sweep_es_stop_points": args.sweep_es_stop_points,
            "sweep_es_target_points": args.sweep_es_target_points,
            "sweep_es_threshold_points": args.sweep_es_threshold_points,
            "sweep_max_opening_range_points": args.sweep_max_opening_range_points,
            "slippage_ticks_per_side": args.slippage_ticks_per_side,
        },
    }
    return fold_rows, train_rows, selected_oos_rows, all_oos_leaderboard, summary


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
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


def write_outputs(
    args: argparse.Namespace,
    fold_rows: Sequence[dict[str, Any]],
    train_rows: Sequence[dict[str, Any]],
    selected_oos_rows: Sequence[dict[str, Any]],
    all_oos_leaderboard: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "folds.csv", fold_rows)
    if not args.skip_train_rankings:
        write_csv(output_dir / "train_rankings.csv", train_rows)
    write_csv(output_dir / "selected_oos.csv", selected_oos_rows)
    write_csv(output_dir / "all_oos_leaderboard.csv", all_oos_leaderboard)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-stdin", action="store_true")
    parser.add_argument("--output-dir", default="topstep_quantpad_walk_forward")
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--timeframe", default="1s", choices=["1s", "1m"])
    parser.add_argument("--chunk-days", type=int, default=14)
    parser.add_argument("--account", default="50K")
    parser.add_argument("--round-turn-cost")
    parser.add_argument("--slippage-ticks-per-side", default="0.5")
    parser.add_argument("--quick-pass-days", type=int, default=21)
    parser.add_argument("--folds", type=int, default=8)
    parser.add_argument("--train-folds", type=int, default=2)
    parser.add_argument("--oos-top-n", type=int, default=3)
    parser.add_argument("--confidence-z", type=float, default=1.0)
    parser.add_argument("--min-completed-attempts", type=int, default=2)
    parser.add_argument("--min-trades", type=int, default=10)
    parser.add_argument("--min-avg-trade-pnl", default="0")
    parser.add_argument("--sweep-symbols", default="ES")
    parser.add_argument("--sweep-families", default="vwap_reversion")
    parser.add_argument("--sweep-quantities", default="2")
    parser.add_argument("--sweep-max-trades-per-day", default="3")
    parser.add_argument("--sweep-max-hold-minutes", default="10")
    parser.add_argument("--sweep-opening-range-minutes", default="5")
    parser.add_argument("--sweep-last-entry-times", default="11:30")
    parser.add_argument("--sweep-max-opening-range-points", default="none,12")
    parser.add_argument("--sweep-max-entry-bar-range-points", default="none")
    parser.add_argument("--sweep-nq-stop-points", default="10,15,20")
    parser.add_argument("--sweep-nq-target-points", default="8,10,12")
    parser.add_argument("--sweep-nq-threshold-points", default="10,15")
    parser.add_argument("--sweep-nq-confirm-points", default="0")
    parser.add_argument("--sweep-es-stop-points", default="6,8,10")
    parser.add_argument("--sweep-es-target-points", default="2,3")
    parser.add_argument("--sweep-es-threshold-points", default="3,4")
    parser.add_argument("--sweep-es-confirm-points", default="0")
    parser.add_argument("--skip-train-rankings", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    fold_rows, train_rows, selected_oos_rows, all_oos_leaderboard, summary = (
        run_walk_forward(args)
    )
    write_outputs(
        args,
        fold_rows,
        train_rows,
        selected_oos_rows,
        all_oos_leaderboard,
        summary,
    )
    print(json.dumps({"output_dir": args.output_dir, "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
