import unittest

from topstep_execution_adapter import (
    BarPathMode,
    ContractSpec,
    ExecutionError,
    Fill,
    FillSide,
    FuturesExecutionAdapter,
    PriceBar,
)
from topstep_rule_simulator import (
    ChallengeStatus,
    RuleEventType,
    TopstepRuleSimulator,
    money,
)


class FuturesExecutionAdapterTest(unittest.TestCase):
    def assert_money(self, actual, expected):
        self.assertEqual(actual, money(expected))

    def adapter(self, tier="50K", commission=0):
        sim = TopstepRuleSimulator.from_tier(tier)
        adapter = FuturesExecutionAdapter(
            sim,
            contract_specs={
                "ES": ContractSpec(
                    "ES",
                    tick_size="0.25",
                    tick_value="12.50",
                    commission_per_contract=commission,
                )
            },
        )
        return sim, adapter

    def test_long_tick_path_feeds_unrealized_pnl_to_rule_simulator(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        result = adapter.mark_price_path("ES", [4001, 4002, 4010])

        self.assertEqual(result.rule_event.event_type, RuleEventType.MARK_OK)
        self.assert_money(result.open_pnl, 500)
        self.assert_money(sim.open_pnl, 500)
        self.assertTrue(adapter.has_open_position("ES"))

    def test_short_tick_path_uses_inverse_pnl(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.SELL, 1, 4000))

        result = adapter.mark_price("ES", 3990)

        self.assertEqual(result.rule_event.event_type, RuleEventType.MARK_OK)
        self.assert_money(result.open_pnl, 500)
        self.assert_money(sim.valuation, 50_500)

    def test_entry_and_exit_commissions_are_counted_once(self):
        sim, adapter = self.adapter(commission="2.50")

        entry = adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))
        self.assertEqual(entry.rule_event.event_type, RuleEventType.MARK_OK)
        self.assert_money(entry.open_pnl, -2.50)
        self.assert_money(sim.closed_balance, 50_000)

        exit_result = adapter.process_fill(Fill("ES", FillSide.SELL, 1, 4010))

        self.assertEqual(exit_result.rule_event.event_type, RuleEventType.REALIZED_OK)
        self.assert_money(exit_result.realized_pnl, 495)
        self.assert_money(sim.closed_balance, 50_495)
        self.assert_money(sim.open_pnl, 0)
        self.assertFalse(adapter.has_open_position())

    def test_realized_exit_can_pass_challenge(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 2, 4000))

        result = adapter.process_fill(Fill("ES", FillSide.SELL, 2, 4030))

        self.assertEqual(result.rule_event.event_type, RuleEventType.PASSED)
        self.assertEqual(sim.status, ChallengeStatus.PASSED)
        self.assert_money(sim.closed_balance, 53_000)
        self.assertFalse(adapter.has_open_position())

    def test_conservative_long_bar_checks_low_before_recovery(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        result = adapter.mark_bar(
            PriceBar("ES", open=4000, high=4050, low=3979, close=4050),
            path_mode=BarPathMode.CONSERVATIVE,
        )

        self.assertEqual(result.rule_event.event_type, RuleEventType.DLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assertTrue(sim.day_locked)
        self.assert_money(sim.closed_balance, 48_950)
        self.assertFalse(adapter.has_open_position())

    def test_close_only_bar_can_survive_same_ohlc_that_conservative_bar_fails(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        result = adapter.mark_bar(
            PriceBar("ES", open=4000, high=4050, low=3979, close=4050),
            path_mode=BarPathMode.CLOSE_ONLY,
        )

        self.assertEqual(result.rule_event.event_type, RuleEventType.MARK_OK)
        self.assert_money(result.open_pnl, 2_500)
        self.assert_money(sim.closed_balance, 50_000)
        self.assertTrue(adapter.has_open_position("ES"))

    def test_conservative_short_bar_checks_high_before_recovery(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.SELL, 1, 4000))

        result = adapter.mark_bar(
            PriceBar("ES", open=4000, high=4021, low=3950, close=3950),
            path_mode=BarPathMode.CONSERVATIVE,
        )

        self.assertEqual(result.rule_event.event_type, RuleEventType.DLL_BREACH)
        self.assert_money(sim.closed_balance, 48_950)
        self.assertFalse(adapter.has_open_position())

    def test_price_path_can_fail_mll_and_clear_adapter_positions(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        result = adapter.mark_price_path("ES", [3990, 3960])

        self.assertEqual(result.rule_event.event_type, RuleEventType.MLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.FAILED_MLL)
        self.assert_money(sim.closed_balance, 48_000)
        self.assertFalse(adapter.has_open_position())

    def test_partial_exit_realizes_closed_contract_and_keeps_remaining_open_pnl(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 2, 4000))

        result = adapter.process_fill(Fill("ES", FillSide.SELL, 1, 4010))

        self.assertEqual(result.rule_event.event_type, RuleEventType.REALIZED_OK)
        self.assert_money(result.realized_pnl, 500)
        self.assert_money(sim.closed_balance, 50_500)
        self.assert_money(sim.open_pnl, 500)
        position = adapter.position("ES")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 1)
        self.assertEqual(position.average_price, money(4000))

    def test_partial_exit_can_trigger_dll_from_remaining_open_loss(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 2, 4000))

        result = adapter.process_fill(Fill("ES", FillSide.SELL, 1, 3980.25))

        self.assertEqual(result.rule_event.event_type, RuleEventType.DLL_BREACH)
        self.assertEqual(sim.status, ChallengeStatus.ACTIVE)
        self.assert_money(result.realized_pnl, -987.50)
        self.assert_money(sim.closed_balance, 48_025)
        self.assertFalse(adapter.has_open_position())

    def test_reversal_realizes_old_position_and_opens_new_direction(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        result = adapter.process_fill(Fill("ES", FillSide.SELL, 2, 4010))

        self.assertEqual(result.rule_event.event_type, RuleEventType.REALIZED_OK)
        self.assert_money(result.realized_pnl, 500)
        self.assert_money(sim.closed_balance, 50_500)
        self.assert_money(sim.open_pnl, 0)
        position = adapter.position("ES")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, -1)
        self.assertEqual(position.average_price, money(4010))

        result = adapter.mark_price("ES", 4000)
        self.assert_money(result.open_pnl, 500)

    def test_scale_in_recalculates_weighted_average_price(self):
        sim, adapter = self.adapter()

        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))
        result = adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4020))

        position = adapter.position("ES")
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 2)
        self.assertEqual(position.average_price, money(4010))
        self.assert_money(result.open_pnl, 1_000)
        self.assert_money(sim.valuation, 51_000)

    def test_multi_symbol_open_pnl_is_account_wide(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        adapter = FuturesExecutionAdapter(sim)

        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))
        adapter.process_fill(Fill("NQ", FillSide.BUY, 1, 15_000))
        adapter.mark_price("ES", 3990)
        result = adapter.mark_price("NQ", 14_990)

        self.assertEqual(result.rule_event.event_type, RuleEventType.MARK_OK)
        self.assert_money(result.open_pnl, -700)
        self.assert_money(sim.valuation, 49_300)

    def test_market_fill_applies_half_spread_and_slippage_adversely(self):
        _, adapter = self.adapter()

        buy = adapter.market_fill(
            "ES",
            FillSide.BUY,
            quantity=1,
            mid_price=4000,
            spread_ticks=2,
            slippage_ticks=1,
        )
        sell = adapter.market_fill(
            "ES",
            FillSide.SELL,
            quantity=1,
            mid_price=4000,
            spread_ticks=2,
            slippage_ticks=1,
        )

        self.assertEqual(buy.fill_price, money("4000.50"))
        self.assertEqual(sell.fill_price, money("3999.50"))

    def test_max_regular_contract_limit_is_enforced_without_mutating_state(self):
        sim, adapter = self.adapter()

        with self.assertRaises(ExecutionError):
            adapter.process_fill(Fill("ES", FillSide.BUY, 6, 4000))

        self.assert_money(sim.closed_balance, 50_000)
        self.assertFalse(adapter.has_open_position())
        self.assertNotIn("ES", adapter.last_prices)

    def test_max_micro_contract_limit_is_enforced(self):
        sim = TopstepRuleSimulator.from_tier("50K")
        adapter = FuturesExecutionAdapter(sim)

        with self.assertRaises(ExecutionError):
            adapter.process_fill(Fill("MNQ", FillSide.BUY, 51, 15_000))

        self.assertFalse(adapter.has_open_position())

    def test_unknown_contract_spec_is_rejected(self):
        _, adapter = self.adapter()

        with self.assertRaises(ExecutionError):
            adapter.process_fill(Fill("ZN", FillSide.BUY, 1, 110))

    def test_adapter_rejects_new_events_after_dll_lockout_until_next_session(self):
        sim, adapter = self.adapter()
        adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))
        adapter.mark_price("ES", 3979)

        with self.assertRaises(ExecutionError):
            adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        sim.end_session()
        sim.start_next_session()
        result = adapter.process_fill(Fill("ES", FillSide.BUY, 1, 4000))

        self.assertEqual(result.rule_event.event_type, RuleEventType.MARK_OK)
        self.assertTrue(adapter.has_open_position("ES"))

    def test_fill_constructor_validates_quantity_and_side(self):
        with self.assertRaises(ValueError):
            Fill("ES", FillSide.BUY, 0, 4000)
        with self.assertRaises(ValueError):
            Fill("ES", "hold", 1, 4000)

    def test_price_bar_constructor_validates_ohlc_bounds(self):
        with self.assertRaises(ValueError):
            PriceBar("ES", open=4000, high=3999, low=3998, close=4000)
        with self.assertRaises(ValueError):
            PriceBar("ES", open=4000, high=4001, low=4002, close=4000)


if __name__ == "__main__":
    unittest.main()
