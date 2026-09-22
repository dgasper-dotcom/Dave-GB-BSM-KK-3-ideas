import unittest

from topstep_rule_simulator import (
    AccountConfig,
    ChallengeStatus,
    RuleEventType,
    RuleStateError,
    TopstepRuleSimulator,
    get_account_config,
    money,
)


class TopstepRuleSimulatorTest(unittest.TestCase):
    def assert_money(self, actual, expected):
        self.assertEqual(actual, money(expected))

    def test_default_tiers_have_expected_initial_risk_parameters(self):
        cases = [
            ("50K", 50_000, 3_000, 2_000, 1_000, 48_000),
            ("100K", 100_000, 6_000, 3_000, 2_000, 97_000),
            ("150K", 150_000, 9_000, 4_500, 3_000, 145_500),
        ]

        for tier, start, target, mll_distance, dll, initial_mll in cases:
            with self.subTest(tier=tier):
                config = get_account_config(tier)
                sim = TopstepRuleSimulator(config)

                self.assert_money(config.starting_balance, start)
                self.assert_money(config.profit_target, target)
                self.assert_money(config.mll_distance, mll_distance)
                self.assert_money(config.daily_loss_limit, dll)
                self.assert_money(sim.closed_balance, start)
                self.assert_money(sim.active_mll, initial_mll)
                self.assert_money(sim.active_dll_threshold, start - dll)
                self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
                self.assertFalse(sim.mll_locked)

    def test_tier_aliases_are_supported(self):
        self.assertEqual(get_account_config("$50K").name, "50K")
        self.assertEqual(get_account_config("50000").name, "50K")
        self.assertEqual(get_account_config("100").name, "100K")
        self.assertEqual(get_account_config("150k").name, "150K")

    def test_unknown_tier_raises_clear_error(self):
        with self.assertRaises(ValueError):
            get_account_config("25K")

    def test_mll_trails_only_after_new_eod_closed_balance_high(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.close_position(800)
        self.assertEqual(event.event_type, RuleEventType.REALIZED_OK)
        self.assert_money(sim.closed_balance, 50_800)
        self.assert_money(sim.active_mll, 48_000)

        event = sim.end_session()
        self.assertEqual(event.event_type, RuleEventType.EOD_MLL_UPDATED)
        self.assert_money(sim.active_mll, 48_800)
        self.assert_money(sim.highest_eod_closed_balance, 50_800)

    def test_mll_never_moves_down_after_lower_eod_closed_balance(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.close_position(800)
        sim.end_session()
        sim.start_next_session()

        sim.close_position(-300)
        event = sim.end_session()

        self.assertEqual(event.event_type, RuleEventType.EOD_NO_CHANGE)
        self.assert_money(sim.closed_balance, 50_500)
        self.assert_money(sim.highest_eod_closed_balance, 50_800)
        self.assert_money(sim.active_mll, 48_800)

    def test_mll_uses_new_eod_high_not_intraday_closed_high(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        sim.close_position(1_000)
        self.assert_money(sim.active_mll, 48_000)
        sim.close_position(-700)
        event = sim.end_session()

        self.assertEqual(event.event_type, RuleEventType.EOD_MLL_UPDATED)
        self.assert_money(sim.closed_balance, 50_300)
        self.assert_money(sim.highest_eod_closed_balance, 50_300)
        self.assert_money(sim.active_mll, 48_300)

    def test_mll_locks_at_start_balance_and_stops_trailing(self):
        config = AccountConfig(
            name="50K-custom-target",
            starting_balance=50_000,
            profit_target=10_000,
            mll_distance=2_000,
            daily_loss_limit=1_000,
            max_contracts=5,
            max_micro_contracts=50,
        )
        sim = TopstepRuleSimulator(config)

        sim.close_position(2_000)
        event = sim.end_session()
        self.assertEqual(event.event_type, RuleEventType.EOD_MLL_LOCKED)
        self.assert_money(sim.active_mll, 50_000)
        self.assertTrue(sim.mll_locked)

        sim.start_next_session()
        sim.close_position(900)
        event = sim.end_session()

        self.assertEqual(event.event_type, RuleEventType.EOD_NO_CHANGE)
        self.assert_money(sim.highest_eod_closed_balance, 52_900)
        self.assert_money(sim.active_mll, 50_000)
        self.assertTrue(sim.mll_locked)

    def test_eod_without_new_high_leaves_mll_unchanged(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        event = sim.end_session()

        self.assertEqual(event.event_type, RuleEventType.EOD_NO_CHANGE)
        self.assert_money(sim.active_mll, 48_000)
        self.assert_money(sim.highest_eod_closed_balance, 50_000)

    def test_eod_requires_closed_positions_so_unrealized_pnl_cannot_trail_mll(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.mark_to_market(3_000)

        with self.assertRaises(RuleStateError):
            sim.end_session()

        self.assert_money(sim.closed_balance, 50_000)
        self.assert_money(sim.active_mll, 48_000)

    def test_mll_breach_on_exact_threshold_fails_immediately(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(-2_000)

        self.assertEqual(event.event_type, RuleEventType.MLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)
        self.assert_money(sim.closed_balance, 48_000)
        self.assert_money(sim.open_pnl, 0)
        self.assertFalse(sim.session_active)

    def test_mll_breach_precedes_dll_when_both_thresholds_are_hit(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(-2_500)

        self.assertEqual(event.event_type, RuleEventType.MLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)
        self.assert_money(sim.closed_balance, 47_500)
        self.assertEqual(sim.dll_breach_count, 0)

    def test_mll_liquidation_slippage_reduces_final_closed_balance(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(-2_000, liquidation_slippage=50)

        self.assertEqual(event.event_type, RuleEventType.MLL_BREACH)
        self.assert_money(sim.closed_balance, 47_950)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)

    def test_dll_breach_on_exact_threshold_locks_day_without_permanent_failure(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(-1_000)

        self.assertEqual(event.event_type, RuleEventType.DLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assertTrue(sim.day_locked)
        self.assertTrue(sim.session_active)
        self.assert_money(sim.closed_balance, 49_000)
        self.assert_money(sim.open_pnl, 0)
        self.assertEqual(sim.dll_breach_count, 1)

    def test_trading_is_blocked_for_remainder_of_day_after_dll_breach(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.mark_to_market(-1_000)

        with self.assertRaises(RuleStateError):
            sim.mark_to_market(100)
        with self.assertRaises(RuleStateError):
            sim.close_position(100)

    def test_dll_resets_next_session_from_post_liquidation_balance(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.mark_to_market(-1_000)
        sim.end_session()
        event = sim.start_next_session()

        self.assertEqual(event.event_type, RuleEventType.SESSION_STARTED)
        self.assertFalse(sim.day_locked)
        self.assert_money(sim.session_start_closed_balance, 49_000)
        self.assert_money(sim.active_dll_threshold, 48_000)
        self.assert_money(sim.active_mll, 48_000)

    def test_dll_includes_realized_and_unrealized_pnl(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.close_position(-500)
        self.assertEqual(event.event_type, RuleEventType.REALIZED_OK)
        self.assert_money(sim.closed_balance, 49_500)

        event = sim.mark_to_market(-500)

        self.assertEqual(event.event_type, RuleEventType.DLL_BREACH)
        self.assert_money(sim.closed_balance, 49_000)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)

    def test_realized_loss_can_trigger_dll(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.close_position(-1_000)

        self.assertEqual(event.event_type, RuleEventType.DLL_BREACH)
        self.assert_money(sim.closed_balance, 49_000)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)

    def test_daily_loss_limit_can_be_disabled_for_custom_accounts(self):
        config = AccountConfig(
            name="no-dll",
            starting_balance=50_000,
            profit_target=3_000,
            mll_distance=2_000,
            daily_loss_limit=None,
            max_contracts=5,
            max_micro_contracts=50,
        )
        sim = TopstepRuleSimulator(config)

        event = sim.mark_to_market(-1_000)

        self.assertEqual(event.event_type, RuleEventType.MARK_OK)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assertFalse(sim.day_locked)
        self.assertIsNone(sim.active_dll_threshold)

    def test_intrabar_mark_path_stops_at_first_dll_breach(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market_path([-200, -800, -1_000, -1_500])

        self.assertEqual(event.event_type, RuleEventType.DLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assert_money(sim.closed_balance, 49_000)
        self.assertEqual(sim.dll_breach_count, 1)

    def test_liquidation_slippage_can_convert_dll_breach_into_mll_failure(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(-1_000, liquidation_slippage=1_200)

        self.assertEqual(event.event_type, RuleEventType.MLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)
        self.assert_money(sim.closed_balance, 47_800)
        self.assertEqual(sim.dll_breach_count, 1)

    def test_post_lock_mll_failure_takes_precedence_over_dll(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.close_position(2_000)
        sim.end_session()
        sim.start_next_session()
        sim.close_position(-500)
        sim.end_session()
        sim.start_next_session()

        event = sim.mark_to_market(-1_500)

        self.assertEqual(event.event_type, RuleEventType.MLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)
        self.assert_money(sim.active_mll, 50_000)
        self.assert_money(sim.active_dll_threshold, 50_500)
        self.assert_money(sim.closed_balance, 50_000)
        self.assertEqual(sim.dll_breach_count, 0)

    def test_profit_target_passes_on_closed_balance(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.close_position(3_000)

        self.assertEqual(event.event_type, RuleEventType.PASSED)
        self.assertEqual(sim.status, ChallengeStatus.PASSED)
        self.assert_money(sim.closed_balance, 53_000)
        self.assertFalse(sim.session_active)
        self.assertTrue(sim.day_locked)

    def test_unrealized_profit_does_not_pass_challenge(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        event = sim.mark_to_market(3_000)

        self.assertEqual(event.event_type, RuleEventType.MARK_OK)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assert_money(sim.closed_balance, 50_000)
        self.assert_money(sim.valuation, 53_000)

    def test_terminal_states_reject_more_events(self):
        passed = TopstepRuleSimulator.from_tier("50K")
        passed.close_position(3_000)
        with self.assertRaises(RuleStateError):
            passed.start_next_session()
        with self.assertRaises(RuleStateError):
            passed.mark_to_market(0)

        failed = TopstepRuleSimulator.from_tier("50K")
        failed.mark_to_market(-2_000)
        with self.assertRaises(RuleStateError):
            failed.start_next_session()
        with self.assertRaises(RuleStateError):
            failed.close_position(0)

    def test_session_boundaries_are_explicit(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        with self.assertRaises(RuleStateError):
            sim.start_next_session()

        sim.end_session()
        with self.assertRaises(RuleStateError):
            sim.end_session()

    def test_custom_account_config_controls_all_thresholds(self):
        config = AccountConfig(
            name="custom",
            starting_balance=25_000,
            profit_target=1_500,
            mll_distance=1_250,
            daily_loss_limit=500,
            max_contracts=2,
            max_micro_contracts=20,
        )
        sim = TopstepRuleSimulator(config)

        self.assert_money(sim.active_mll, 23_750)
        self.assert_money(sim.active_dll_threshold, 24_500)

        sim.close_position(750)
        sim.end_session()

        self.assert_money(sim.active_mll, 24_500)
        self.assertFalse(sim.mll_locked)

    def test_snapshot_exposes_complete_rule_state(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        sim.close_position(500)
        snap = sim.snapshot()

        self.assertEqual(snap.day_number, 1)
        self.assert_money(snap.closed_balance, 50_500)
        self.assert_money(snap.open_pnl, 0)
        self.assert_money(snap.valuation, 50_500)
        self.assert_money(snap.session_start_closed_balance, 50_000)
        self.assert_money(snap.highest_eod_closed_balance, 50_000)
        self.assert_money(snap.active_mll, 48_000)
        self.assert_money(snap.active_dll_threshold, 49_000)
        self.assert_money(snap.target_balance, 53_000)
        self.assertEqual(snap.status, ChallengeStatus.ACTIVE)
        self.assertTrue(snap.session_active)
        self.assertFalse(snap.day_locked)
        self.assertFalse(snap.mll_locked)
        self.assertEqual(snap.dll_breach_count, 0)

    def test_negative_liquidation_slippage_is_rejected(self):
        sim = TopstepRuleSimulator.from_tier("50K")

        with self.assertRaises(ValueError):
            sim.mark_to_market(-100, liquidation_slippage=-1)

        self.assert_money(sim.open_pnl, 0)
        self.assert_money(sim.closed_balance, 50_000)

        with self.assertRaises(ValueError):
            sim.close_position(-100, liquidation_slippage=-1)

        self.assert_money(sim.open_pnl, 0)
        self.assert_money(sim.closed_balance, 50_000)


if __name__ == "__main__":
    unittest.main()
