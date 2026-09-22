#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kalshi_mlb_backtest.dashboard import DashboardConfig, serve_dashboard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the Kalshi scanner dashboard.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--db", type=Path, default=None, help="Optional SQLite DB path for MLB panels.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--refresh-seconds", type=int, default=5)
    parser.add_argument("--live-limit", type=int, default=300)
    args = parser.parse_args(argv)

    output_dir = args.output_dir.resolve()
    db_path = args.db.resolve() if args.db else output_dir / "kalshi_mlb_backtest.db"
    serve_dashboard(
        DashboardConfig(
            db_path=db_path,
            output_dir=output_dir,
            host=args.host,
            port=args.port,
            refresh_seconds=args.refresh_seconds,
            live_limit=args.live_limit,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
