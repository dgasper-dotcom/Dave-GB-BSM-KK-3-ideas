import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from topstep_rule_simulator import money
from topstep_xfa_simulator import (
    TradePath,
    TradingDay,
    XFAAccountConfig,
    XFAAccountSimulator,
    XFAEventType,
    XFAPayoutPath,
    XFAPayoutPolicy,
    XFAStatus,
    load_trade_days,
    run_xfa_bootstrap,
    summarize_xfa_trials,
)
from topstep_xfa_walk_forward import contiguous_chunks


class TopstepXFASimulatorTest(unittest.TestCase):
    def assert_money(self, actual, expected):
        self.assertEqual(actual, money(expected))

    def finish_day(self, sim, pnl):
        sim.close_position(pnl)
        sim.end_session()

    def test_default_50k_xfa_starts_from_zero_pnl_with_negative_mll(self):
        sim = XFAAccountSimulator()

        self.assert_money(sim.closed_balance, 0)
        self.assert_money(sim.active_mll, -2_000)
        self.assertFalse(sim.mll_locked)
        self.assertEqual(sim.status, XFAStatus.ACTIVE)
        self.assertIsNone(sim.active_dll_threshold)

    def test_exact_mll_touch_fails_funded_account(self):
        sim = XFAAccountSimulator()

        event = sim.mark_to_market(-2_000)

        self.assertEqual(event.event_type, XFAEventType.MLL_BREACH)
        self.assertEqual(sim.status, XFAStatus.FAILED)
        self.assert_money(sim.closed_balance, -2_000)

    def test_balance_reaching_2000_locks_mll_at_zero(self):
        sim = XFAAccountSimulator()

        event = sim.close_position(2_000)

        self.assertEqual(event.event_type, XFAEventType.MLL_LOCKED)
        self.assertTrue(sim.mll_locked)
        self.assert_money(sim.active_mll, 0)
        self.assert_money(sim.closed_balance, 2_000)

    def test_optional_daily_loss_limit_locks_day_without_permanent_failure(self):
        sim = XFAAccountSimulator(XFAAccountConfig(daily_loss_limit=500))

        event = sim.mark_to_market(-500)

        self.assertEqual(event.event_type, XFAEventType.DLL_BREACH)
        self.assertEqual(sim.status, XFAStatus.ACTIVE)
        self.assertTrue(sim.day_locked)
        self.assert_money(sim.closed_balance, -500)
        self.assertEqual(sim.dll_breach_count, 1)

    def test_standard_path_requires_five_150_winning_days(self):
        sim = XFAAccountSimulator()
        policy = XFAPayoutPolicy(path=XFAPayoutPath.STANDARD)

        for _ in range(4):
            self.finish_day(sim, 200)
            self.assertFalse(sim.payout_eligible(policy))
            sim.start_next_session()

        self.finish_day(sim, 200)

        self.assertTrue(sim.payout_eligible(policy))

    def test_consistency_path_allows_three_balanced_trading_days(self):
        sim = XFAAccountSimulator()
        policy = XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY)

        for day in range(3):
            self.finish_day(sim, 796)
            if day < 2:
                self.assertFalse(sim.payout_eligible(policy))
                sim.start_next_session()

        self.assertTrue(sim.payout_eligible(policy))

    def test_consistency_path_blocks_one_day_dominating_profit(self):
        sim = XFAAccountSimulator()
        policy = XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY)

        for pnl in (2_000, 200, 200):
            self.finish_day(sim, pnl)
            if pnl != 200 or sim.period_trading_days < 3:
                sim.start_next_session()

        self.assertFalse(sim.payout_eligible(policy))
        event = sim.request_payout(policy)
        self.assertEqual(event.event_type, XFAEventType.PAYOUT_SKIPPED)

    def test_payout_caps_at_50_percent_and_uses_90_percent_split(self):
        sim = XFAAccountSimulator()
        policy = XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY)
        for day in range(3):
            self.finish_day(sim, 796)
            if day < 2:
                sim.start_next_session()

        event = sim.request_payout(policy)

        self.assertEqual(event.event_type, XFAEventType.PAYOUT)
        self.assert_money(sim.gross_payouts, 1_194)
        self.assert_money(sim.trader_payouts, 1_074.60)
        self.assert_money(sim.closed_balance, 1_194)
        self.assertTrue(sim.mll_locked)
        self.assert_money(sim.active_mll, 0)

    def test_consistency_payout_uses_50k_path_cap(self):
        sim = XFAAccountSimulator()
        policy = XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY)
        for day in range(3):
            self.finish_day(sim, 4_000)
            if day < 2:
                sim.start_next_session()

        event = sim.request_payout(policy)

        self.assertEqual(event.event_type, XFAEventType.PAYOUT)
        self.assert_money(sim.gross_payouts, 3_000)
        self.assert_money(sim.trader_payouts, 2_700)
        self.assert_money(sim.closed_balance, 9_000)

    def test_consistency_payout_cap_doubles_when_dll_is_configured(self):
        sim = XFAAccountSimulator(XFAAccountConfig(daily_loss_limit=1_000))
        policy = XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY)
        for day in range(3):
            self.finish_day(sim, 4_000)
            if day < 2:
                sim.start_next_session()

        event = sim.request_payout(policy)

        self.assertEqual(event.event_type, XFAEventType.PAYOUT)
        self.assert_money(sim.gross_payouts, 6_000)
        self.assert_money(sim.trader_payouts, 5_400)
        self.assert_money(sim.closed_balance, 6_000)

    def test_load_trade_days_reads_required_trade_path_columns(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "trades.csv"
        path.write_text(
            "session_date,realized_pnl,mae_pnl,mfe_pnl,exit_reason\n"
            "2024-01-02,796.20,-100.00,796.20,target\n",
            encoding="utf-8",
        )

        days = load_trade_days(path)

        self.assertEqual(len(days), 1)
        self.assertEqual(days[0].session_date, date(2024, 1, 2))
        self.assert_money(days[0].trades[0].realized_pnl, 796.20)

    def test_bootstrap_summary_includes_challenge_ev(self):
        days = [
            TradingDay(
                session_date=date(2024, 1, 2),
                trades=(
                    TradePath(
                        session_date=date(2024, 1, 2),
                        realized_pnl=money(796),
                        mae_pnl=money(-50),
                        mfe_pnl=money(796),
                    ),
                ),
            )
        ]
        results = run_xfa_bootstrap(
            days,
            trials=1,
            account_config=XFAAccountConfig(),
            payout_policy=XFAPayoutPolicy(path=XFAPayoutPath.CONSISTENCY),
            max_days=3,
            seed=1,
        )

        summary = summarize_xfa_trials(
            results,
            challenge_cost=85,
            challenge_pass_rate=Decimal("0.33"),
            activation_fee=0,
        )

        self.assertEqual(results[0].payout_count, 1)
        self.assert_money(results[0].trader_payouts, 1_074.60)
        self.assertEqual(summary["probability_of_any_payout"], "1.0000")
        self.assertEqual(summary["expected_challenge_cost_per_passed_account"], "257.58")
        self.assertEqual(summary["ev_per_challenge_attempt"], "269.62")

    def test_walk_forward_chunks_keep_chronological_order(self):
        days = [
            TradingDay(
                session_date=date(2024, 1, day),
                trades=(
                    TradePath(
                        session_date=date(2024, 1, day),
                        realized_pnl=money(100),
                        mae_pnl=money(-10),
                        mfe_pnl=money(100),
                    ),
                ),
            )
            for day in range(1, 8)
        ]

        chunks = contiguous_chunks(days, 3)

        self.assertEqual([d.session_date.day for d in chunks[0]], [1, 2])
        self.assertEqual([d.session_date.day for d in chunks[1]], [3, 4])
        self.assertEqual([d.session_date.day for d in chunks[2]], [5, 6, 7])


if __name__ == "__main__":
    unittest.main()
