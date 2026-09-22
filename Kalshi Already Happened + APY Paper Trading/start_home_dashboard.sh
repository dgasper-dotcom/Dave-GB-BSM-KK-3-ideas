#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8766}"
CONTRACTS="${CONTRACTS:-100}"
POLL_SECONDS="${POLL_SECONDS:-600}"
ORDERBOOK_WORKERS="${ORDERBOOK_WORKERS:-1}"

cd "$ROOT"

screen -dmS kalshi_known_outcome_loop bash -lc \
  "cd '$ROOT' && CONTRACTS='$CONTRACTS' POLL_SECONDS='$POLL_SECONDS' ORDERBOOK_WORKERS='$ORDERBOOK_WORKERS' ./known_outcome_loop.sh >> known_outcome_loop.log 2>&1"

screen -dmS kalshi_dashboard bash -lc \
  "cd '$ROOT' && python3 run_dashboard.py --host '$HOST' --port '$PORT' --output-dir '$ROOT' >> dashboard.log 2>&1"

echo "Known-outcome scanner: screen session kalshi_known_outcome_loop"
echo "Dashboard: http://localhost:$PORT"
echo "Dashboard screen session: kalshi_dashboard"
