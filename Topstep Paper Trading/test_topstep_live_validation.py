import tempfile
import unittest
from pathlib import Path

from topstep_live_validation import load_live_trades, summarize_live_trades


class TopstepLiveValidationTest(unittest.TestCase):
    def test_loads_separate_portfolio_trade_logs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        nq = root / "NQ"
        es = root / "ES"
        nq.mkdir()
        es.mkdir()
        header = (
            "symbol,session_date,realized_pnl,exit_reason,rule_event,mae_pnl,mfe_pnl\n"
        )
        (nq / "trades.csv").write_text(
            header + "NQ,2024-01-02,796.20,target,realized_ok,-100.00,796.20\n",
            encoding="utf-8",
        )
        (es / "trades.csv").write_text(
            header + "ES,2024-01-03,-403.80,stop,realized_ok,-403.80,200.00\n",
            encoding="utf-8",
        )

        trades = load_live_trades(root)
        summary = summarize_live_trades(trades)

        self.assertEqual(summary["total"]["trades"], 2)
        self.assertEqual(summary["total"]["active_days"], 2)
        self.assertEqual(summary["total"]["win_rate"], "0.5000")
        self.assertEqual(summary["symbols"]["NQ"]["realized_pnl"], "796.20")
        self.assertEqual(summary["symbols"]["ES"]["realized_pnl"], "-403.80")


if __name__ == "__main__":
    unittest.main()
