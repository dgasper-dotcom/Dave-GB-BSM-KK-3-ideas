# topstep

Live paper trading tools for Topstep-style prop challenge research.

The main app is `topstep_prop_challenge_paper.py`. By default it runs each monitored futures symbol as its own independent paper challenge portfolio, with separate DLL/MLL state, PnL, trade log, and localhost dashboard row.

## What It Does

- Runs separate `50K` Topstep-style paper challenge portfolios by default.
- Monitors `NQ`, `ES`, `RTY`, `CL`, `GC`, and `6E` by default.
- Pulls delayed 1-minute OHLC bars from Yahoo's direct chart endpoint.
- Trades an opening-range fade model with per-market normalized stop/target/filter settings.
- Applies Topstep's published product-specific round-turn costs by default, split half on entry and half on exit.
- Allows at most one open position per symbol portfolio.
- Tracks account valuation, DLL buffer, MLL buffer, active trade, closed trades, skips, MAE, and MFE.

This is for paper trading and research. It is not wired to a broker and does not place real orders.

## Install

```bash
git clone https://github.com/dgasper-dotcom/topstep.git
cd topstep
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Live Paper Trading

```bash
./scripts/run_live_paper.sh
```

Open:

```text
http://127.0.0.1:8787
```

The default command runs:

```bash
python topstep_prop_challenge_paper.py \
  --host 127.0.0.1 \
  --port 8787 \
  --source yahoo-chart \
  --symbol-set six \
  --portfolio-mode separate \
  --poll-seconds 60 \
  --output-dir topstep_prop_challenge_paper
```

Use `--portfolio-mode shared` only when you intentionally want one account shared across all symbols.

## Install As A macOS Background Service

From the repo root:

```bash
./scripts/install_launch_agent.sh
```

Then open:

```text
http://127.0.0.1:8787
```

Logs and state are written to:

```text
topstep_prop_challenge_paper/
```

In separate portfolio mode, each symbol writes its own state and trades under:

```text
topstep_prop_challenge_paper/NQ/
topstep_prop_challenge_paper/ES/
...
```

To stop and remove the background service:

```bash
./scripts/uninstall_launch_agent.sh
```

## Check Status

```bash
curl -s http://127.0.0.1:8787/api/snapshot | python -m json.tool | head -120
```

Closed trades, once they exist, are written to:

```text
topstep_prop_challenge_paper/trades.csv
```

For 30-60 day live paper validation, summarize closed paper trades with:

```bash
python topstep_live_validation.py \
  --output-dir topstep_prop_challenge_paper \
  --json
```

Use this to track whether live paper behavior matches the 1-minute research assumptions:

```text
trade count
active trading days
win rate
average trade PnL
average MAE/MFE
rule exits
per-symbol contribution
```

## Run Tests

```bash
python -m unittest discover -p 'test_topstep*.py'
```

## Simulate The Funded XFA Phase

After a challenge pass, use `topstep_xfa_simulator.py` to model the Express Funded Account separately from the challenge. The simulator starts from `$0` XFA PnL, uses a `-$2,000` MLL for the 50K account, locks the MLL at `$0` after `+$2,000` or the first payout, and models Standard or Consistency payout qualification.

## Walk-Forward Optimize The Challenge Scalp

To optimize the NQ high-win/low-RR scalp without trusting one full-sample leaderboard:

```bash
python topstep_nq_scalp_walk_forward.py \
  --data /Users/davidgasper/Downloads/Dataset_NQ_1min_2022_2025.csv \
  --output-dir topstep_nq_scalp_walk_forward \
  --folds 8 \
  --train-folds 3 \
  --oos-top-n 5 \
  --stop-points 16,20,24 \
  --reward-risk-ratios 0.5,0.6,0.7 \
  --breakout-buffer-points 8,10,12,15 \
  --max-hold-minutes 5,10 \
  --max-trades-per-session 1,2,3
```

The optimizer writes:

```text
topstep_nq_scalp_walk_forward/folds.csv
topstep_nq_scalp_walk_forward/train_rankings.csv
topstep_nq_scalp_walk_forward/oos_evaluations.csv
topstep_nq_scalp_walk_forward/leaderboard.csv
topstep_nq_scalp_walk_forward/summary.json
```

The default cost model uses Topstep's NQ `$3.80` round-turn cost as `$1.90` per side. The script uses CSV `Vwap_RTH` when present and computes RTH VWAP when it is missing.

For the 50K XFA, the default payout model uses:

```text
Minimum payout: $125
Standard path: 5 winning days of $150+
Consistency path: 3 trading days, largest day <= 40% of total profit
Payout request: 50% of balance
50K XFA Standard cap: $2,000, or $4,000 with DLL
50K XFA Consistency cap: $3,000, or $6,000 with DLL
Trader split: 90%
```

The input should be a strategy-runner `trades.csv` with `session_date`, `realized_pnl`, `mae_pnl`, and `mfe_pnl` columns. Example using an NQ research trade stream, Topstep's `$3.80` NQ round-turn cost, an `$85` challenge cost, and a `33%` challenge pass rate:

```bash
python topstep_xfa_simulator.py \
  --trades path/to/trades.csv \
  --output-dir topstep_nq_xfa_consistency_topstep_fee_mc_10000 \
  --trials 10000 \
  --max-days 504 \
  --payout-path consistency \
  --pnl-haircut-per-trade 3.80 \
  --challenge-cost 85 \
  --challenge-pass-rate 0.33
```

To sweep payout buffers and danger-zone risk settings:

```bash
python topstep_xfa_policy_sweep.py \
  --trades path/to/trades.csv \
  --output-dir topstep_nq_xfa_policy_sweep \
  --trials 1000 \
  --payout-path consistency \
  --retained-buffers 0,1000,2000 \
  --danger-buffers none,1000,1500 \
  --danger-scales 0.25,0.5,1 \
  --pnl-haircut-per-trade 3.80 \
  --challenge-cost 85 \
  --challenge-pass-rate 0.33
```

The sweep writes:

```text
topstep_nq_xfa_policy_sweep/leaderboard.csv
topstep_nq_xfa_policy_sweep/summary.json
```

To walk-forward validate XFA policy selection instead of trusting one full-sample leaderboard:

```bash
python topstep_xfa_walk_forward.py \
  --trades path/to/trades.csv \
  --output-dir topstep_nq_xfa_walk_forward \
  --folds 7 \
  --train-folds 3 \
  --train-trials 500 \
  --test-trials 1000 \
  --payout-path consistency \
  --retained-buffers 0,1000,2000 \
  --danger-buffers none,1000,1500 \
  --danger-scales 0.25,0.5,1 \
  --pnl-haircut-per-trade 3.80 \
  --challenge-cost 85 \
  --challenge-pass-rate 0.33
```

The walk-forward run writes:

```text
topstep_nq_xfa_walk_forward/folds.csv
topstep_nq_xfa_walk_forward/summary.json
```

## Current Caveats

- Yahoo data is delayed and unofficial. It is good enough for live paper mechanics, not execution-quality fills.
- Only NQ has been researched deeply. The non-NQ settings are normalized starting assumptions, not validated production strategies.
- The monitor starts with a warmup backfill. Backfilled bars initialize state but do not create fake historical trades.
- The default cost model uses Topstep's published product-specific round-turn costs. It does not add extra slippage beyond those costs.
- The XFA simulator models the funded rules as configurable research assumptions. Re-check current Topstep payout rules before using outputs as operational instructions.
- This does not replace a real broker or licensed futures feed.
