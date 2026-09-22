import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from topstep_data_loader import FuturesBar
from topstep_prop_challenge_paper import (
    PropChallengePaperEngine,
    SeparatePortfolioPaperMonitor,
    create_app,
)
from topstep_rule_simulator import money


def bar(symbol, ts, open_, high, low, close, volume="100"):
    return FuturesBar(
        timestamp=datetime.strptime(ts, "%m/%d/%Y %H:%M"),
        symbol=symbol,
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
    )


class TopstepPropChallengePaperTest(unittest.TestCase):
    def new_engine(self, symbols=("NQ",)):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return PropChallengePaperEngine(symbols=symbols, output_dir=Path(tmp.name))

    def feed_nq_opening_range(self, engine):
        for minute in range(30, 35):
            engine.process_bar(
                bar("NQ", f"01/02/2024 09:{minute:02d}", "95", "100", "90", "95")
            )

    def feed_symbol_opening_range(self, engine, symbol):
        for minute in range(30, 35):
            engine.process_bar(
                bar(symbol, f"01/02/2024 09:{minute:02d}", "95", "100", "90", "95")
            )

    def test_shared_account_records_trade_with_topstep_fee(self):
        engine = self.new_engine()
        self.feed_nq_opening_range(engine)

        engine.process_bar(bar("NQ", "01/02/2024 09:35", "111", "111", "111", "111"))
        engine.process_bar(bar("NQ", "01/02/2024 09:36", "111", "111", "71", "71"))

        self.assertIsNone(engine.active_trade)
        self.assertEqual(len(engine.trades), 1)
        self.assertEqual(engine.trades[0]["symbol"], "NQ")
        self.assertEqual(engine.trades[0]["exit_reason"], "target")
        self.assertEqual(Decimal(engine.trades[0]["realized_pnl"]), money("796.20"))
        self.assertEqual(engine.sim.closed_balance, money("50796.20"))

    def test_custom_cost_override_keeps_stress_haircut_available(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = PropChallengePaperEngine(
            symbols=("NQ",),
            output_dir=Path(tmp.name),
            commission_per_contract=Decimal("12.50"),
        )
        self.feed_nq_opening_range(engine)

        engine.process_bar(bar("NQ", "01/02/2024 09:35", "111", "111", "111", "111"))
        engine.process_bar(bar("NQ", "01/02/2024 09:36", "111", "111", "71", "71"))

        self.assertEqual(Decimal(engine.trades[0]["realized_pnl"]), money("775"))

    def test_one_position_policy_skips_competing_symbol_signal(self):
        engine = self.new_engine(symbols=("NQ", "ES"))
        self.feed_symbol_opening_range(engine, "NQ")
        self.feed_symbol_opening_range(engine, "ES")

        engine.process_bars(
            [
                (bar("NQ", "01/02/2024 09:35", "111", "111", "111", "111"), True),
                (bar("ES", "01/02/2024 09:35", "105", "105", "105", "105"), True),
            ]
        )

        self.assertIsNotNone(engine.active_trade)
        self.assertEqual(engine.active_trade.symbol, "NQ")
        self.assertEqual(engine.skipped_entry_conflicts, 1)
        self.assertEqual(len(engine.trades), 0)

    def test_flask_snapshot_reports_shared_prop_mode(self):
        engine = self.new_engine(symbols=("NQ", "ES"))
        app = create_app(engine)
        client = app.test_client()

        response = client.get("/api/snapshot")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["mode"], "shared_prop_challenge")
        self.assertEqual(payload["symbols"], ["NQ", "ES"])
        self.assertIn("account", payload)
        self.assertIn("symbols_state", payload)

    def test_separate_portfolios_allow_independent_symbol_trades(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        monitor = SeparatePortfolioPaperMonitor(symbols=("NQ", "ES"), output_dir=Path(tmp.name))
        self.feed_symbol_opening_range(monitor, "NQ")
        self.feed_symbol_opening_range(monitor, "ES")

        monitor.process_bars(
            [
                (bar("NQ", "01/02/2024 09:35", "111", "111", "111", "111"), True),
                (bar("ES", "01/02/2024 09:35", "105", "105", "105", "105"), True),
            ]
        )

        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["portfolio_mode"], "separate")
        self.assertIn("NQ", snapshot["active_trades"])
        self.assertIn("ES", snapshot["active_trades"])
        self.assertEqual(snapshot["account"]["status"], "2/2 active portfolios")

    def test_non_tradable_warmup_bars_do_not_advance_challenge_day(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = PropChallengePaperEngine(
            symbols=("ES",),
            output_dir=Path(tmp.name),
            account_tier="150K",
            strategy_family="vwap_reversion",
            quantity=6,
            stop_points=Decimal("8"),
            target_points=Decimal("3"),
            breakout_buffer_points=Decimal("3"),
            disable_session_filters=True,
            max_trades_per_session=3,
            max_hold_minutes=10,
        )

        engine.process_bar(bar("ES", "01/02/2024 09:30", "5000", "5001", "4999", "5000"), allow_entries=False)
        engine.process_bar(bar("ES", "01/03/2024 09:30", "5000", "5001", "4999", "5000"), allow_entries=False)
        self.assertEqual(engine.sim.day_number, 1)
        self.assertIsNone(engine.current_session_date)
        self.assertEqual(engine.sim.closed_balance, money("150000"))
        self.assertIsNotNone(engine.runtimes["ES"].rth_vwap)

        engine.process_bar(bar("ES", "01/03/2024 09:35", "5000", "5001", "4999", "5000"), allow_entries=True)
        self.assertEqual(engine.sim.day_number, 1)
        self.assertEqual(engine.current_session_date.isoformat(), "2024-01-03")


if __name__ == "__main__":
    unittest.main()
