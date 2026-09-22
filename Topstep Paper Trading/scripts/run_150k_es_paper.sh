#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:a:h}
REPO_DIR=${SCRIPT_DIR:h}
PYTHON="${REPO_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON}" ]]; then
  PYTHON="$(command -v python3)"
fi

cd "${REPO_DIR}"
exec "${PYTHON}" topstep_prop_challenge_paper.py \
  --host 127.0.0.1 \
  --port "${PORT:-8787}" \
  --source yahoo-chart \
  --symbols ES \
  --portfolio-mode separate \
  --account 150K \
  --strategy-family vwap_reversion \
  --quantity 6 \
  --stop-points 8 \
  --target-points 3 \
  --breakout-buffer-points 3 \
  --disable-session-filters \
  --max-trades-per-session 3 \
  --max-hold-minutes 10 \
  --last-entry-time 11:30 \
  --slippage-ticks-per-side 0.5 \
  --poll-seconds 60 \
  --output-dir topstep_150k_es_paper
