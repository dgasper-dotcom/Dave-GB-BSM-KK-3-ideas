import tempfile
import unittest
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from topstep_data_loader import FuturesBar
from topstep_nq_paper_monitor import (
    MultiSymbolPaperMonitor,
    NQPaperEngine,
    create_app,
    default_strategy_config,
    parse_yahoo_chart_payload,
)
from topstep_rule_simulator import money


def bar(ts, open_, high, low, close, volume="100"):
    return FuturesBar(
        timestamp=datetime.strptime(ts, "%m/%d/%Y %H:%M"),
        symbol="NQ",
        open=Decimal(str(open_)),
        high=Decimal(str(high)),
        low=Decimal(str(low)),
        close=Decimal(str(close)),
        volume=Decimal(str(volume)),
    )


class TopstepNQPaperMonitorTest(unittest.TestCase):
    def new_engine(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return NQPaperEngine(config=default_strategy_config(), output_dir=Path(tmp.name))

    def feed_opening_range(self, engine):
        for minute in range(30, 35):
            engine.process_bar(
                bar(f"01/02/2024 09:{minute:02d}", "95", "100", "90", "95")
            )

    def test_or_fade_enters_and_records_target_with_topstep_fee(self):
        engine = self.new_engine()
        self.feed_opening_range(engine)

        engine.process_bar(bar("01/02/2024 09:35", "111", "111", "111", "111"))
        self.assertIsNotNone(engine.active_trade)
        self.assertEqual(engine.active_trade.side.value, "sell")
        self.assertEqual(engine.active_trade.entry_price, Decimal("111"))

        engine.process_bar(bar("01/02/2024 09:36", "111", "111", "71", "71"))

        self.assertIsNone(engine.active_trade)
        self.assertEqual(len(engine.trades), 1)
        trade = engine.trades[0]
        self.assertEqual(trade.exit_reason, "target")
        self.assertEqual(trade.realized_pnl, money("796.20"))
        self.assertEqual(engine.snapshot()["stats"]["realized_pnl"], 796.2)

    def test_gap_filter_skips_session_once(self):
        engine = self.new_engine()
        engine.process_bar(bar("01/01/2024 15:59", "100", "100", "100", "100"))
        for minute in range(30, 35):
            engine.process_bar(
                bar(f"01/02/2024 09:{minute:02d}", "250", "260", "245", "250")
            )

        engine.process_bar(bar("01/02/2024 09:35", "271", "271", "271", "271"))

        self.assertIsNone(engine.active_trade)
        self.assertEqual(len(engine.trades), 0)
        self.assertEqual(engine.risk_filter_skips, {"opening_gap": 1})
        self.assertEqual(engine.snapshot()["session"]["skip_reason"], "opening_gap")

    def test_warmup_bars_build_opening_range_without_entering_trade(self):
        engine = self.new_engine()
        for minute in range(30, 35):
            engine.process_bar(
                bar(f"01/02/2024 09:{minute:02d}", "95", "100", "90", "95"),
                allow_entries=False,
            )
        engine.process_bar(
            bar("01/02/2024 09:35", "111", "111", "111", "111"),
            allow_entries=False,
        )

        snapshot = engine.snapshot()
        self.assertTrue(snapshot["session"]["ready"])
        self.assertEqual(snapshot["session"]["opening_range_size"], 10.0)
        self.assertIsNone(engine.active_trade)
        self.assertEqual(len(engine.trades), 0)

        engine.process_bar(bar("01/02/2024 09:36", "112", "112", "112", "112"))

        self.assertIsNotNone(engine.active_trade)
        self.assertEqual(engine.active_trade.entry_price, Decimal("112"))

    def test_flask_manual_bar_endpoint_accepts_bar(self):
        engine = self.new_engine()
        app = create_app(engine)
        client = app.test_client()
        response = client.post(
            "/api/bar",
            json={
                "timestamp": "01/02/2024 09:30",
                "open": "95",
                "high": "100",
                "low": "90",
                "close": "95",
                "volume": "100",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["accepted"])
        self.assertEqual(engine.snapshot()["bars_processed"], 1)

    def test_yahoo_chart_parser_skips_incomplete_current_bar(self):
        payload = {
            "chart": {
                "result": [
                    {
                        "meta": {"exchangeTimezoneName": "America/New_York"},
                        "timestamp": [1710000000, 1710000060, 1710000119],
                        "indicators": {
                            "quote": [
                                {
                                    "open": [100, 101, 102],
                                    "high": [105, 106, 107],
                                    "low": [99, 100, 101],
                                    "close": [104, 105, 106],
                                    "volume": [10, 20, 30],
                                }
                            ]
                        },
                    }
                ],
                "error": None,
            }
        }

        bars = parse_yahoo_chart_payload(payload, data_symbol="NQ")

        self.assertEqual(len(bars), 2)
        self.assertEqual(bars[0].symbol, "NQ")
        self.assertEqual(bars[0].open, Decimal("100"))
        self.assertEqual(bars[1].close, Decimal("105"))

    def test_non_nq_contract_uses_own_tick_size_and_geometry(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = NQPaperEngine(
            config=default_strategy_config("CL"),
            output_dir=Path(tmp.name),
        )
        for minute in range(30, 35):
            engine.process_bar(
                FuturesBar(
                    timestamp=datetime.strptime(f"01/02/2024 09:{minute:02d}", "%m/%d/%Y %H:%M"),
                    symbol="CL",
                    open=Decimal("75.00"),
                    high=Decimal("75.10"),
                    low=Decimal("74.90"),
                    close=Decimal("75.00"),
                    volume=Decimal("100"),
                )
            )

        engine.process_bar(
            FuturesBar(
                timestamp=datetime.strptime("01/02/2024 09:35", "%m/%d/%Y %H:%M"),
                symbol="CL",
                open=Decimal("75.31"),
                high=Decimal("75.32"),
                low=Decimal("75.31"),
                close=Decimal("75.314"),
                volume=Decimal("100"),
            )
        )

        self.assertIsNotNone(engine.active_trade)
        self.assertEqual(engine.active_trade.entry_price, Decimal("75.31"))
        self.assertEqual(engine.active_trade.stop_price, Decimal("75.71"))
        self.assertEqual(engine.active_trade.target_price, Decimal("74.51"))

    def test_multi_symbol_snapshot_exposes_each_engine(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        monitor = MultiSymbolPaperMonitor(["NQ", "ES"], output_dir=Path(tmp.name))
        app = create_app(monitor)
        client = app.test_client()

        response = client.get("/api/snapshot")

        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["multi_symbol"])
        self.assertEqual(payload["symbols"], ["NQ", "ES"])
        self.assertIn("NQ", payload["engines"])
        self.assertIn("ES", payload["engines"])


if __name__ == "__main__":
    unittest.main()
