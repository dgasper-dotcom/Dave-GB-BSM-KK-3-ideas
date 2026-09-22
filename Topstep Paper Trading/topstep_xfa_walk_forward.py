"""Walk-forward validation for XFA payout/risk policies."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Any, Optional, Sequence

from topstep_rule_simulator import money
from topstep_xfa_policy_sweep import parse_grid
from topstep_xfa_simulator import (
    TradingDay,
    XFAAccountConfig,
    XFAPayoutPath,
    XFAPayoutPolicy,
    XFARiskPolicy,
    load_trade_days,
    run_xfa_bootstrap,
    summarize_xfa_trials,
)


def contiguous_chunks(items: Sequence[TradingDay], chunks: int) -> list[list[TradingDay]]:
    if chunks <= 1:
        raise ValueError("chunks must be greater than 1")
    if len(items) < chunks:
        raise ValueError("not enough trading days for requested chunks")
    out: list[list[TradingDay]] = []
    for index in range(chunks):
        start = index * len(items) // chunks
        end = (index + 1) * len(items) // chunks
        out.append(list(items[start:end]))
    return out


def policy_candidates(args: argparse.Namespace) -> list[tuple[XFAPayoutPolicy, XFARiskPolicy]]:
    candidates = []
    for retained_buffer in parse_grid(args.retained_buffers):
        for min_payout in parse_grid(args.min_payouts):
            for danger_buffer in parse_grid(args.danger_buffers, allow_none=True):
                for danger_scale in parse_grid(args.danger_scales):
                    for protected_profit in parse_grid(args.protected_profits, allow_none=True):
                        for protected_scale in parse_grid(args.protected_scales):
                            candidates.append(
                                (
                                    XFAPayoutPolicy(
                                        path=XFAPayoutPath(args.payout_path),
                                        retained_buffer=retained_buffer,
                                        min_payout=min_payout,
                                    ),
                                    XFARiskPolicy(
                                        danger_buffer=danger_buffer,
                                        danger_scale=danger_scale,
                                        protected_profit=protected_profit,
                                        protected_scale=protected_scale,
                                    ),
                                )
                            )
    if not candidates:
        raise ValueError("policy grid produced no candidates")
    return candidates


def policy_fields(
    payout_policy: XFAPayoutPolicy,
    risk_policy: XFARiskPolicy,
) -> dict[str, str]:
    return {
        "retained_buffer": str(payout_policy.retained_buffer),
        "min_payout": str(payout_policy.min_payout),
        "danger_buffer": ""
        if risk_policy.danger_buffer is None
        else str(risk_policy.danger_buffer),
        "danger_scale": str(risk_policy.danger_scale),
        "protected_profit": ""
        if risk_policy.protected_profit is None
        else str(risk_policy.protected_profit),
        "protected_scale": str(risk_policy.protected_scale),
    }


def evaluate_policy(
    days: Sequence[TradingDay],
    account_config: XFAAccountConfig,
    payout_policy: XFAPayoutPolicy,
    risk_policy: XFARiskPolicy,
    *,
    trials: int,
    max_days: int,
    seed: int,
    pnl_haircut_per_trade: str,
    liquidation_slippage: str,
    challenge_cost: str,
    challenge_pass_rate: Decimal,
    activation_fee: str,
) -> dict[str, Any]:
    results = run_xfa_bootstrap(
        days=days,
        trials=trials,
        account_config=account_config,
        payout_policy=payout_policy,
        risk_policy=risk_policy,
        max_days=max_days,
        seed=seed,
        pnl_haircut_per_trade=pnl_haircut_per_trade,
        liquidation_slippage=liquidation_slippage,
    )
    return summarize_xfa_trials(
        results,
        challenge_cost=challenge_cost,
        challenge_pass_rate=challenge_pass_rate,
        activation_fee=activation_fee,
    )


def prefixed(prefix: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def run_walk_forward(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    days = load_trade_days(args.trades)
    chunks = contiguous_chunks(days, args.folds)
    if args.train_folds >= args.folds:
        raise ValueError("train_folds must be less than folds")
    candidates = policy_candidates(args)
    account_config = XFAAccountConfig(
        daily_loss_limit=Decimal(args.daily_loss_limit)
        if args.daily_loss_limit
        else None
    )
    rows = []
    challenge_pass_rate = Decimal(args.challenge_pass_rate)

    for fold_start in range(0, args.folds - args.train_folds):
        train_days = [
            day
            for chunk in chunks[fold_start : fold_start + args.train_folds]
            for day in chunk
        ]
        test_days = chunks[fold_start + args.train_folds]

        train_rankings = []
        for candidate_index, (payout_policy, risk_policy) in enumerate(candidates, start=1):
            summary = evaluate_policy(
                train_days,
                account_config,
                payout_policy,
                risk_policy,
                trials=args.train_trials,
                max_days=args.max_days,
                seed=args.seed + fold_start * 10_000 + candidate_index,
                pnl_haircut_per_trade=args.pnl_haircut_per_trade,
                liquidation_slippage=args.liquidation_slippage,
                challenge_cost=args.challenge_cost,
                challenge_pass_rate=challenge_pass_rate,
                activation_fee=args.activation_fee,
            )
            train_rankings.append((summary, payout_policy, risk_policy))

        train_rankings.sort(
            key=lambda item: Decimal(str(item[0]["ev_per_challenge_attempt"])),
            reverse=True,
        )
        best_train, payout_policy, risk_policy = train_rankings[0]
        test_summary = evaluate_policy(
            test_days,
            account_config,
            payout_policy,
            risk_policy,
            trials=args.test_trials,
            max_days=args.max_days,
            seed=args.seed + 500_000 + fold_start,
            pnl_haircut_per_trade=args.pnl_haircut_per_trade,
            liquidation_slippage=args.liquidation_slippage,
            challenge_cost=args.challenge_cost,
            challenge_pass_rate=challenge_pass_rate,
            activation_fee=args.activation_fee,
        )
        rows.append(
            {
                "fold": fold_start + 1,
                "train_start": train_days[0].session_date.isoformat(),
                "train_end": train_days[-1].session_date.isoformat(),
                "test_start": test_days[0].session_date.isoformat(),
                "test_end": test_days[-1].session_date.isoformat(),
                "train_days": len(train_days),
                "test_days": len(test_days),
                **policy_fields(payout_policy, risk_policy),
                **prefixed("train", best_train),
                **prefixed("test", test_summary),
            }
        )

    test_evs = [Decimal(str(row["test_ev_per_challenge_attempt"])) for row in rows]
    summary = {
        "folds": len(rows),
        "avg_test_ev_per_challenge_attempt": str(
            money(sum(test_evs, Decimal("0")) / Decimal(len(test_evs)))
        ),
        "positive_test_ev_folds": sum(1 for ev in test_evs if ev > 0),
        "worst_test_ev_per_challenge_attempt": str(min(test_evs)),
        "best_test_ev_per_challenge_attempt": str(max(test_evs)),
        "metadata": {
            "trades": args.trades,
            "trading_days": len(days),
            "folds": args.folds,
            "train_folds": args.train_folds,
            "train_trials": args.train_trials,
            "test_trials": args.test_trials,
            "max_days": args.max_days,
            "payout_path": args.payout_path,
            "pnl_haircut_per_trade": str(money(args.pnl_haircut_per_trade)),
            "challenge_cost": str(money(args.challenge_cost)),
            "challenge_pass_rate": str(challenge_pass_rate),
            "activation_fee": str(money(args.activation_fee)),
            "candidate_count": len(candidates),
        },
    }
    return rows, summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output-dir", default="topstep_xfa_walk_forward")
    parser.add_argument("--folds", type=int, default=7)
    parser.add_argument("--train-folds", type=int, default=3)
    parser.add_argument("--train-trials", type=int, default=500)
    parser.add_argument("--test-trials", type=int, default=1000)
    parser.add_argument("--max-days", type=int, default=504)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--payout-path", choices=["standard", "consistency"], default="consistency")
    parser.add_argument("--retained-buffers", default="0,1000,2000")
    parser.add_argument("--min-payouts", default="125")
    parser.add_argument("--danger-buffers", default="none,1000,1500")
    parser.add_argument("--danger-scales", default="0.25,0.5,1")
    parser.add_argument("--protected-profits", default="none")
    parser.add_argument("--protected-scales", default="1")
    parser.add_argument("--pnl-haircut-per-trade", default="0")
    parser.add_argument("--liquidation-slippage", default="0")
    parser.add_argument("--daily-loss-limit")
    parser.add_argument("--challenge-cost", default="85")
    parser.add_argument("--challenge-pass-rate", default="0.33")
    parser.add_argument("--activation-fee", default="0")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    rows, summary = run_walk_forward(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "folds.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
