import csv
import tempfile
import unittest
from datetime import date, datetime, time
from decimal import Decimal
from pathlib import Path

from topstep_nq_strategy_runner import (
    OpeningRangeBreakoutConfig,
    OpeningRangeBreakoutRunner,
    session_date_for_timestamp,
)
from topstep_rule_simulator import ChallengeStatus, RuleEventType, money


class TopstepNQStrategyRunnerTest(unittest.TestCase):
    def write_bars(self, directory, bars):
        path = Path(directory) / "nq.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp ET", "open", "high", "low", "close", "volume"])
            for row in bars:
                writer.writerow(row)
        return path

    def write_vwap_bars(self, directory, bars):
        path = Path(directory) / "nq_vwap.csv"
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp ET",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "Vwap_RTH",
                    "Vwap_ETH",
                ]
            )
            for row in bars:
                writer.writerow(row)
        return path

    def opening_range_rows(self):
        rows = []
        for minute in range(30, 45):
            rows.append(
                [
                    f"01/02/2024 09:{minute:02d}",
                    "95",
                    "100",
                    "90",
                    "95",
                    "100",
                ]
            )
        return rows

    def run_rows(self, rows, **config_kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_bars(tmp, rows)
            config = OpeningRangeBreakoutConfig(**config_kwargs)
            return OpeningRangeBreakoutRunner(config).run(str(path))

    def test_session_date_rolls_evening_globex_to_next_trade_date(self):
        self.assertEqual(
            session_date_for_timestamp(datetime(2024, 1, 1, 18, 1)),
            date(2024, 1, 2),
        )
        self.assertEqual(
            session_date_for_timestamp(datetime(2024, 1, 2, 9, 30)),
            date(2024, 1, 2),
        )

    def test_opening_range_breakout_target_records_positive_trade(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "161", "101", "161", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=30, target_points=60)

        self.assertEqual(len(result.trades), 1)
        trade = result.trades[0]
        self.assertEqual(trade.side, "buy")
        self.assertEqual(trade.exit_reason, "target")
        self.assertEqual(trade.rule_event, RuleEventType.REALIZED_OK.value)
        self.assertEqual(trade.entry_price, money(101))
        self.assertEqual(trade.exit_price, money(161))
        self.assertEqual(trade.realized_pnl, money(1200))
        self.assertEqual(result.attempts[0].outcome, "incomplete")
        self.assertEqual(result.summary()["trade_count"], 1)
        self.assertEqual(result.summary()["avg_trade_pnl"], "1200.00")

    def test_target_winner_records_adverse_excursion_metrics(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "161", "91", "161", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=30, target_points=60)
        trade = result.trades[0]

        self.assertEqual(trade.exit_reason, "target")
        self.assertEqual(trade.mae_points, Decimal("10"))
        self.assertEqual(trade.mfe_points, Decimal("60"))
        self.assertEqual(trade.mae_pnl, money(-200))
        self.assertEqual(trade.mfe_pnl, money(1200))
        self.assertEqual(trade.time_to_target_minutes, 1)
        self.assertEqual(trade.time_underwater_minutes, 1)
        self.assertEqual(trade.bars_held, 1)
        self.assertEqual(result.summary()["winners_with_adverse_excursion_rate"], "1.0000")
        self.assertEqual(result.summary()["avg_winner_mae_pnl"], "-200.00")

    def test_target_can_pass_challenge(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "251", "101", "251", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=30, target_points=150)

        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(result.attempts[0].outcome, ChallengeStatus.PASSED.value)
        self.assertEqual(result.attempts[0].net_pnl, money(3000))
        self.assertEqual(result.attempts[0].terminal_rule_event, RuleEventType.PASSED.value)
        self.assertEqual(result.attempts[0].failure_reason, "")
        self.assertEqual(result.summary()["passes"], 1)
        self.assertEqual(result.summary()["pass_rate"], "1.0000")

    def test_stop_can_fail_mll(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "101", "1", "1", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=100, target_points=200)

        self.assertEqual(result.trades[0].exit_reason, "stop")
        self.assertEqual(result.trades[0].realized_pnl, money(-2000))
        self.assertEqual(result.trades[0].rule_event, RuleEventType.MLL_BREACH.value)
        self.assertEqual(result.attempts[0].outcome, ChallengeStatus.FAILED_MLL.value)
        self.assertEqual(result.attempts[0].terminal_rule_event, RuleEventType.MLL_BREACH.value)
        self.assertEqual(result.attempts[0].terminal_detail, "active MLL breached")
        self.assertEqual(result.attempts[0].failure_reason, "stop_exit_breached_mll")
        self.assertEqual(result.summary()["failures"], 1)
        self.assertEqual(
            result.summary()["failure_reasons"],
            {"stop_exit_breached_mll": 1},
        )

    def test_unrealized_drawdown_can_trigger_dll_before_stop(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "101", "50", "101", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=200, target_points=400)

        self.assertEqual(result.trades[0].exit_reason, "dll_breach")
        self.assertEqual(result.trades[0].rule_event, RuleEventType.DLL_BREACH.value)
        self.assertEqual(result.trades[0].realized_pnl, money(-1020))
        self.assertEqual(result.attempts[0].outcome, "incomplete")
        self.assertEqual(result.attempts[0].dll_breaches, 1)
        self.assertEqual(result.attempts[0].failure_reason, "")

    def test_short_breakout_profit(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "89", "89", "89", "89", "100"],
                ["01/02/2024 09:46", "89", "89", "29", "29", "100"],
            ]
        )

        result = self.run_rows(rows, stop_points=30, target_points=60)

        self.assertEqual(result.trades[0].side, "sell")
        self.assertEqual(result.trades[0].exit_reason, "target")
        self.assertEqual(result.trades[0].entry_price, money(89))
        self.assertEqual(result.trades[0].exit_price, money(29))
        self.assertEqual(result.trades[0].realized_pnl, money(1200))

    def test_opening_range_fade_enters_against_upside_break(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "101", "41", "41", "100"],
            ]
        )

        result = self.run_rows(
            rows,
            strategy_family="orb_fade",
            stop_points=30,
            target_points=60,
        )

        self.assertEqual(result.trades[0].side, "sell")
        self.assertEqual(result.trades[0].exit_reason, "target")
        self.assertEqual(result.trades[0].entry_price, money(101))
        self.assertEqual(result.trades[0].exit_price, money(41))
        self.assertEqual(result.trades[0].realized_pnl, money(1200))

    def test_vwap_reversion_fades_extension_from_rth_vwap(self):
        rows = []
        for minute in range(30, 45):
            rows.append(
                [
                    f"01/02/2024 09:{minute:02d}",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                ]
            )
        rows.extend(
            [
                ["01/02/2024 09:45", "130", "130", "130", "130", "100", "100", "100"],
                ["01/02/2024 09:46", "130", "130", "70", "70", "100", "100", "100"],
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_vwap_bars(tmp, rows)
            config = OpeningRangeBreakoutConfig(
                strategy_family="vwap_reversion",
                breakout_buffer_points=Decimal("20"),
                stop_points=30,
                target_points=60,
            )
            result = OpeningRangeBreakoutRunner(config).run(str(path))

        self.assertEqual(result.trades[0].side, "sell")
        self.assertEqual(result.trades[0].exit_reason, "target")
        self.assertEqual(result.trades[0].entry_price, money(130))
        self.assertEqual(result.trades[0].exit_price, money(70))
        self.assertEqual(result.trades[0].realized_pnl, money(1200))

    def test_scalp_reversion_fades_vwap_extension_with_small_target(self):
        rows = []
        for minute in range(30, 35):
            rows.append(
                [
                    f"01/02/2024 09:{minute:02d}",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                    "100",
                ]
            )
        rows.extend(
            [
                ["01/02/2024 09:35", "112", "112", "112", "112", "100", "100", "100"],
                ["01/02/2024 09:36", "112", "112", "107", "107", "100", "100", "100"],
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_vwap_bars(tmp, rows)
            config = OpeningRangeBreakoutConfig(
                strategy_family="scalp_reversion",
                opening_range_minutes=5,
                breakout_buffer_points=Decimal("10"),
                stop_points=20,
                target_points=5,
            )
            result = OpeningRangeBreakoutRunner(config).run(str(path))

        self.assertEqual(result.trades[0].side, "sell")
        self.assertEqual(result.trades[0].exit_reason, "target")
        self.assertEqual(result.trades[0].entry_price, money(112))
        self.assertEqual(result.trades[0].exit_price, money(107))
        self.assertEqual(result.trades[0].realized_pnl, money(100))

    def test_last_entry_time_blocks_late_breakouts(self):
        rows = self.opening_range_rows()
        rows.append(["01/02/2024 15:01", "101", "101", "101", "101", "100"])

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=60,
            last_entry_time=time(15, 0),
        )

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.attempts[0].outcome, "incomplete")

    def test_pre_lock_opening_range_filter_skips_entry(self):
        rows = self.opening_range_rows()
        rows.append(["01/02/2024 09:45", "101", "101", "101", "101", "100"])

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=60,
            pre_lock_max_opening_range_points=Decimal("5"),
        )

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.summary()["risk_filter_skips"], {"pre_lock_opening_range": 1})

    def test_general_opening_range_filter_skips_entry(self):
        rows = self.opening_range_rows()
        rows.append(["01/02/2024 09:45", "101", "101", "101", "101", "100"])

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=60,
            max_opening_range_points=Decimal("5"),
        )

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.summary()["risk_filter_skips"], {"opening_range": 1})
        self.assertEqual(result.summary()["max_opening_range_points"], "5")

    def test_opening_gap_filter_skips_entry(self):
        rows = self.opening_range_rows()
        rows.append(["01/02/2024 15:59", "95", "95", "95", "95", "100"])
        for minute in range(30, 45):
            rows.append(
                [
                    f"01/03/2024 09:{minute:02d}",
                    "200",
                    "205",
                    "195",
                    "200",
                    "100",
                ]
            )
        rows.append(["01/03/2024 09:45", "206", "206", "206", "206", "100"])

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=60,
            max_opening_gap_points=Decimal("50"),
        )

        self.assertEqual(len(result.trades), 0)
        self.assertEqual(result.summary()["risk_filter_skips"], {"opening_gap": 1})
        self.assertEqual(result.summary()["max_opening_gap_points"], "50")

    def test_combined_pre_lock_filter_requires_buffer_and_opening_range_trigger(self):
        rows = self.opening_range_rows()
        rows.append(["01/02/2024 09:45", "101", "101", "101", "101", "100"])

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=60,
            pre_lock_filter_mode="combined",
            pre_lock_min_mll_buffer=Decimal("500"),
            pre_lock_max_opening_range_points=Decimal("5"),
        )

        self.assertEqual(len(result.trades), 1)
        self.assertEqual(result.trades[0].risk_state, "pre_lock")
        self.assertEqual(result.summary()["risk_filter_skips"], {})

    def test_post_lock_quantity_applies_after_mll_locks(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "201", "101", "201", "100"],
            ]
        )
        for minute in range(30, 45):
            rows.append(
                [
                    f"01/03/2024 09:{minute:02d}",
                    "95",
                    "100",
                    "90",
                    "95",
                    "100",
                ]
            )
        rows.extend(
            [
                ["01/03/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/03/2024 09:46", "101", "101", "101", "101", "100"],
            ]
        )

        result = self.run_rows(
            rows,
            stop_points=30,
            target_points=100,
            post_lock_quantity=2,
        )

        self.assertEqual(result.trades[0].risk_state, "pre_lock")
        self.assertEqual(result.trades[0].quantity, 1)
        self.assertEqual(result.trades[1].risk_state, "post_lock")
        self.assertEqual(result.trades[1].quantity, 2)

    def test_max_rows_limits_processing(self):
        rows = self.opening_range_rows()
        rows.extend(
            [
                ["01/02/2024 09:45", "101", "101", "101", "101", "100"],
                ["01/02/2024 09:46", "101", "161", "101", "161", "100"],
            ]
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_bars(tmp, rows)
            config = OpeningRangeBreakoutConfig(stop_points=30, target_points=60)
            result = OpeningRangeBreakoutRunner(config).run(str(path), max_rows=10)

        self.assertEqual(result.rows_processed, 10)
        self.assertEqual(len(result.trades), 0)


if __name__ == "__main__":
    unittest.main()
