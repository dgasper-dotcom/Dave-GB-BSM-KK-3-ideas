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
  --port 8787 \
  --source yahoo-chart \
  --symbol-set six \
  --portfolio-mode separate \
  --strategy-family scalp_reversion \
  --scalp-reward-risk-ratio 0.6 \
  --max-trades-per-session 3 \
  --max-hold-minutes 10 \
  --poll-seconds 60 \
  --output-dir topstep_prop_challenge_paper
