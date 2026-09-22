# Moving The Paper Trader To Another Computer

Use the full bundle if you want the other computer to retrain/rescore the paper model. Use the minimal bundle only if you want the current trained artifacts, watchlist, journal template, and slippage tools.

## Create The Bundle

```bash
cd /Users/davidgasper/__pycache__/transformative_tx_ml

# Recommended: code + trained artifacts + fused panel for retraining/rescoring.
./scripts/export_paper_trader.sh full

# Smaller: code + current paper artifacts only.
./scripts/export_paper_trader.sh minimal
```

Current generated bundles:

- `exports/paper_trader_full_20260920_094325.tar.gz`
- `exports/paper_trader_minimal_20260920_094325.tar.gz`

## Transfer It

Any of these are fine:

```bash
# Local network copy.
scp exports/paper_trader_full_20260920_094325.tar.gz user@OTHER_COMPUTER:~/

# Or use rsync if both machines are reachable.
rsync -av exports/paper_trader_full_20260920_094325.tar.gz user@OTHER_COMPUTER:~/
```

AirDrop, Dropbox, iCloud Drive, or a USB drive are also fine. The full bundle is currently about 288 MB compressed.

## Set Up On The Other Computer

```bash
mkdir -p ~/transformative_tx_ml
tar -xzf ~/paper_trader_full_20260920_094325.tar.gz -C ~/transformative_tx_ml
cd ~/transformative_tx_ml

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m pytest
```

## Run The Paper Model

```bash
source .venv/bin/activate

python -m src.paper_trading_model \
  --panel reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv \
  --out-dir reports/paper_trading_model \
  --max-iter 30 \
  --max-event-train-rows 100000 \
  --max-return-train-rows 100000 \
  --min-predicted-net-return 0.0 \
  --watchlist-size 50
```

The important output files are:

- `reports/paper_trading_model/MODEL_CARD.md`
- `reports/paper_trading_model/paper_orders.csv`
- `reports/paper_trading_model/paper_watchlist.csv`
- `reports/paper_trading_model/paper_trade_journal_template.csv`

## Slippage Follow-Up

After paper fills exist on the new computer:

```bash
python -m src.slippage_reality_check \
  --fills reports/paper_trading_model/paper_trade_journal_template.csv \
  --panel reports/paper_trading_model/latest_scores.csv \
  --out-dir reports/slippage_reality_check
```

Do not put API keys in the bundle. If you use QuantPad or broker data later, set keys locally on the new machine with environment variables.
