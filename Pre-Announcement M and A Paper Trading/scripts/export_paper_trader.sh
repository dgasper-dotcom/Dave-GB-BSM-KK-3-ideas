#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-full}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${2:-exports}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

mkdir -p "${ROOT_DIR}/${OUT_DIR}"

COMMON_PATHS=(
  PAPER_TRADER_TRANSFER.md
  README.md
  requirements.txt
  pytest.ini
  config
  scripts/export_paper_trader.sh
  src
  tests
  reports/paper_trading_model
  reports/REAL_SLIPPAGE_DATA_PLAN.md
  reports/slippage_reality_check
)

FULL_EXTRA_PATHS=(
  reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv
  reports/backfill_2018_2026_sec_fusion_cap8/fusion_summary.json
)

case "${MODE}" in
  minimal)
    ARCHIVE="${ROOT_DIR}/${OUT_DIR}/paper_trader_minimal_${STAMP}.tar.gz"
    PATHS=("${COMMON_PATHS[@]}")
    ;;
  full)
    ARCHIVE="${ROOT_DIR}/${OUT_DIR}/paper_trader_full_${STAMP}.tar.gz"
    PATHS=("${COMMON_PATHS[@]}" "${FULL_EXTRA_PATHS[@]}")
    ;;
  *)
    echo "Usage: $0 [minimal|full] [out_dir]" >&2
    echo "  minimal: code, requirements, trained model outputs, paper journal, slippage tools" >&2
    echo "  full: minimal plus fused panel needed to retrain/rescore" >&2
    exit 2
    ;;
esac

cd "${ROOT_DIR}"
tar -czf "${ARCHIVE}" "${PATHS[@]}"

echo "${ARCHIVE}"
du -sh "${ARCHIVE}"
