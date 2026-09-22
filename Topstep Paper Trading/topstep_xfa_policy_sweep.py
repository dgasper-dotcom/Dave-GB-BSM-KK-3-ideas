"""Sweep XFA payout/risk policies and rank by expected value."""

from __future__ import annotations

import argparse
import csv
import json
from decimal import Decimal
from pathlib import Path
from typing import Optional, Sequence

from topstep_rule_simulator import money
from topstep_xfa_simulator import (
    XFAAccountConfig,
    XFAPayoutPath,
    XFAPayoutPolicy,
    XFARiskPolicy,
    load_trade_days,
    run_xfa_bootstrap,
    summarize_xfa_trials,
)


def parse_grid(value: str, *, allow_none: bool = False) -> list[Optional[Decimal]]:
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
        raise ValueError("grid cannot be empty")
    return items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trades", required=True)
    parser.add_argument("--output-dir", default="topstep_xfa_policy_sweep")
    parser.add_argument("--trials", type=int, default=1_000)
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
    days = load_trade_days(args.trades)
    account_config = XFAAccountConfig(
        daily_loss_limit=Decimal(args.daily_loss_limit)
        if args.daily_loss_limit
        else None
    )
    rows = []
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    retained_buffers = parse_grid(args.retained_buffers)
    min_payouts = parse_grid(args.min_payouts)
    danger_buffers = parse_grid(args.danger_buffers, allow_none=True)
    danger_scales = parse_grid(args.danger_scales)
    protected_profits = parse_grid(args.protected_profits, allow_none=True)
    protected_scales = parse_grid(args.protected_scales)

    run_id = 0
    for retained_buffer in retained_buffers:
        for min_payout in min_payouts:
            for danger_buffer in danger_buffers:
                for danger_scale in danger_scales:
                    for protected_profit in protected_profits:
                        for protected_scale in protected_scales:
                            run_id += 1
                            payout_policy = XFAPayoutPolicy(
                                path=XFAPayoutPath(args.payout_path),
                                retained_buffer=retained_buffer,
                                min_payout=min_payout,
                            )
                            risk_policy = XFARiskPolicy(
                                danger_buffer=danger_buffer,
                                danger_scale=danger_scale,
                                protected_profit=protected_profit,
                                protected_scale=protected_scale,
                            )
                            results = run_xfa_bootstrap(
                                days=days,
                                trials=args.trials,
                                account_config=account_config,
                                payout_policy=payout_policy,
                                risk_policy=risk_policy,
                                max_days=args.max_days,
                                seed=args.seed + run_id,
                                pnl_haircut_per_trade=args.pnl_haircut_per_trade,
                                liquidation_slippage=args.liquidation_slippage,
                            )
                            summary = summarize_xfa_trials(
                                results,
                                challenge_cost=args.challenge_cost,
                                challenge_pass_rate=Decimal(args.challenge_pass_rate),
                                activation_fee=args.activation_fee,
                            )
                            rows.append(
                                {
                                    "retained_buffer": str(money(retained_buffer)),
                                    "min_payout": str(money(min_payout)),
                                    "danger_buffer": ""
                                    if danger_buffer is None
                                    else str(money(danger_buffer)),
                                    "danger_scale": str(danger_scale),
                                    "protected_profit": ""
                                    if protected_profit is None
                                    else str(money(protected_profit)),
                                    "protected_scale": str(protected_scale),
                                    **summary,
                                }
                            )

    rows.sort(key=lambda row: Decimal(str(row["ev_per_challenge_attempt"])), reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    with (out / "leaderboard.csv").open("w", newline="") as f:
        fieldnames = ["rank"] + [key for key in rows[0] if key != "rank"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_doc = {
        "metadata": {
            "trades": args.trades,
            "trading_days": len(days),
            "trials": args.trials,
            "max_days": args.max_days,
            "payout_path": args.payout_path,
            "pnl_haircut_per_trade": str(money(args.pnl_haircut_per_trade)),
            "challenge_cost": str(money(args.challenge_cost)),
            "challenge_pass_rate": str(Decimal(args.challenge_pass_rate)),
            "activation_fee": str(money(args.activation_fee)),
            "rows": len(rows),
        },
        "best": rows[0],
    }
    (out / "summary.json").write_text(json.dumps(summary_doc, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), **summary_doc}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
