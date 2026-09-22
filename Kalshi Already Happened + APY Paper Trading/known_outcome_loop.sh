#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${OUT:-$ROOT}"
DURATION_MINUTES="${DURATION_MINUTES:-20160}"
POLL_SECONDS="${POLL_SECONDS:-600}"
ORDERBOOK_WORKERS="${ORDERBOOK_WORKERS:-1}"
CONTRACTS="${CONTRACTS:-100}"
MIN_CONTRACTS="${MIN_CONTRACTS:-1}"
MAX_ASK="${MAX_ASK:-0.99}"
KNOWN_OUTCOME_APY="${KNOWN_OUTCOME_APY:-0.0325}"
KNOWN_OUTCOME_LOOKBACK_DAYS="${KNOWN_OUTCOME_LOOKBACK_DAYS:-3}"
KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT="${KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT:-0.001}"
KNOWN_OUTCOME_SERIES_TICKER="${KNOWN_OUTCOME_SERIES_TICKER:-}"
MAX_MARKET_PAGES="${MAX_MARKET_PAGES:-}"

started_at="$(date +%s)"
end_at=$((started_at + DURATION_MINUTES * 60))

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"
}

scan_once() {
  cmd=(
    python3 "$ROOT/run_known_outcome.py" scan
    --output-dir "$OUT"
    --contracts "$CONTRACTS"
    --min-contracts "$MIN_CONTRACTS"
    --max-ask "$MAX_ASK"
    --known-outcome-apy "$KNOWN_OUTCOME_APY"
    --known-outcome-lookback-days "$KNOWN_OUTCOME_LOOKBACK_DAYS"
    --known-outcome-min-net-profit-per-contract "$KNOWN_OUTCOME_MIN_NET_PROFIT_PER_CONTRACT"
    --known-outcome-orderbook-workers "$ORDERBOOK_WORKERS"
  )
  if [[ -n "$KNOWN_OUTCOME_SERIES_TICKER" ]]; then
    cmd+=(--known-outcome-series-ticker "$KNOWN_OUTCOME_SERIES_TICKER")
  fi
  if [[ -n "$MAX_MARKET_PAGES" ]]; then
    cmd+=(--max-market-pages "$MAX_MARKET_PAGES")
  fi
  "${cmd[@]}"
}

log "known outcome loop start duration_minutes=$DURATION_MINUTES poll_seconds=$POLL_SECONDS orderbook_workers=$ORDERBOOK_WORKERS contracts=$CONTRACTS min_contracts=$MIN_CONTRACTS apy=$KNOWN_OUTCOME_APY"
while [[ "$(date +%s)" -lt "$end_at" ]]; do
  log "known-outcome-scan start"
  if scan_once; then
    log "known-outcome-scan complete"
  else
    log "known-outcome-scan failed"
  fi
  sleep "$POLL_SECONDS"
done
log "known outcome loop complete"
