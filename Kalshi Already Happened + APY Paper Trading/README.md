# Kalshi Known-Outcome Scanner

This scanner looks for Kalshi markets where an external verifier indicates the outcome is already known, then records APY-adjusted paper entries using real executable orderbook liquidity.

It is separate from the MLB fair-value model. It does not model future baseball outcomes; it only attempts to trade markets where the result has already happened or can already be verified.

## What It Scans

- MLB player props: already-hit YES, final known YES, final known NO
- MLB, NFL, NCAAF, NBA, NHL, WNBA game winner markets
- Final-score spread, total, and team-total markets where the threshold can be parsed
- Hourly and daily high-temperature markets using Weather.com verifier endpoints
- Index close/range markets using Yahoo Finance chart closes

## Economics

The default paper settings are:

- `contracts=100`
- `min_contracts=1`
- `max_ask=0.99`
- `known_outcome_apy=0.0325`
- `known_outcome_min_net_profit_per_contract=0.001`

Capital used is execution cost plus fees. APY-adjusted PnL subtracts a 3.25% annualized cash hurdle for the expected settlement wait.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional Kalshi authentication:

```bash
export KALSHI_ACCESS_KEY="your-key-id"
export KALSHI_PRIVATE_KEY_FILE="/path/to/private_key.pem"
```

The scanner can read public orderbooks without auth when Kalshi allows it, but authenticated requests are more reliable.

## Run One Scan

```bash
python3 run_known_outcome.py scan \
  --contracts 100 \
  --min-contracts 1 \
  --max-ask 0.99 \
  --known-outcome-apy 0.0325
```

## Run Continuously

```bash
./known_outcome_loop.sh
```

Environment overrides:

```bash
CONTRACTS=100 POLL_SECONDS=90 ./known_outcome_loop.sh
```

## Run Scanner + Dashboard On a Home Computer

On Windows, run:

```powershell
.\start_home_dashboard.bat
```

If Kalshi returns `429 Too Many Requests`, wait 5-10 minutes, then restart with a slower/light scan:

```powershell
$env:POLL_SECONDS="600"
$env:ORDERBOOK_WORKERS="1"
$env:MAX_MARKET_PAGES="1"
$env:KNOWN_OUTCOME_SERIES_TICKER="KXMLBHIT,KXMLBHR,KXTEMPNYCH,KXTEMPCHIH,KXTEMPDCH,KXTEMPLAXH,KXTEMPMIAH,KXTEMPAUSH,KXHIGHNY,KXHIGHCHI,KXHIGHDEN,KXHIGHLAX,KXHIGHMIA,KXHIGHPHIL,KXHIGHAUS"
.\start_home_dashboard.ps1
```

That scans MLB player props and weather first, with one market page per series and one orderbook worker. After it runs cleanly, remove `MAX_MARKET_PAGES` and broaden `KNOWN_OUTCOME_SERIES_TICKER`.

On macOS/Linux:

Install `screen` if it is not already available, then run:

```bash
./start_home_dashboard.sh
```

This starts two detached sessions:

- `kalshi_known_outcome_loop`: scans every 90 seconds with `CONTRACTS=100`
- `kalshi_dashboard`: serves the dashboard on `http://localhost:8766`

To check the sessions:

```bash
screen -ls
tail -f known_outcome_loop.log
tail -f dashboard.log
```

To stop them:

```bash
screen -S kalshi_known_outcome_loop -X quit
screen -S kalshi_dashboard -X quit
```

To view the dashboard from another device on your home network, start it with:

```bash
HOST=0.0.0.0 ./start_home_dashboard.sh
```

Then open `http://HOME_COMPUTER_LOCAL_IP:8766`.

For access while away from home, use a private tunnel such as Tailscale and open:

```text
http://HOME_COMPUTER_TAILSCALE_IP:8766
```

## Outputs

Runtime files are intentionally ignored by git.

- `known_outcome_candidates.csv`: verified known outcomes and liquidity status
- `known_outcome_opportunities.csv`: currently fillable entries passing the APY filter
- `known_outcome_trades.csv`: cumulative deduped paper ledger
- `known_outcome_pnl.csv`: APY-adjusted PnL summary by market family and side

## Notes

The ledger is deduped by `kalshi_market_ticker|known_side`. If a market was previously entered at a smaller cap, raising `--contracts` does not automatically top up that existing paper entry.
