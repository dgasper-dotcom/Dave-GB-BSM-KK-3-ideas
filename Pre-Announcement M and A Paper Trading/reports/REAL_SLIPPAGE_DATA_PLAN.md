# Real Slippage Data Plan

Bottom line: yes, we can find and use real slippage data, but there are three different data qualities.

## 1. Actual Fill Logs

This is the best source for this strategy because it measures the exact thing we care about: our order, our size, our broker, our order type, our timestamp, and our realized fill.

Required fields for the new analyzer:

- `ticker`
- `signal_date`
- `planned_entry_date`
- `planned_position_dollars`
- `actual_entry_time`
- `actual_entry_price`
- `actual_entry_bid`
- `actual_entry_ask`
- `actual_entry_mid`
- `actual_entry_shares`
- `actual_entry_status`
- `actual_exit_date`
- `actual_exit_price`

The paper model now writes a richer `paper_trade_journal_template.csv` so these fields are captured during paper testing.

Run after paper fills exist:

```bash
python3 -m src.slippage_reality_check \
  --fills reports/paper_trading_model/paper_trade_journal_template.csv \
  --panel reports/paper_trading_model/latest_scores.csv \
  --out-dir reports/slippage_reality_check
```

## 2. NBBO, Tick, Or L2 Quote Data

This is the best source before we have our own fills. It answers: what spread and displayed liquidity were available at the time we would have entered?

Use it to estimate:

- half-spread cost versus midpoint
- marketable order cost versus midpoint
- whether a 1% account position would fit in top-of-book or need multiple levels
- whether signal names are too illiquid for the model's expected edge

The analyzer accepts quote snapshots with `bid`/`ask`, or L2-style fields like `ask_px_00`, `ask_sz_00`, `bid_px_00`, `bid_sz_00`.

QuantPad's direct API is appropriate for this job because the MCP guide says the REST/Python SDK supports tick/quote data and `mbp-10` 10-level order-book depth for US equities where coverage is available. Set `QUANTPAD_API_KEY` locally; do not put it in code or chat.

## 3. Public SEC Rule 605 / 606 Data

This is real execution-quality data, but it is aggregate, not order-level. It helps benchmark brokers and venues; it will not tell us the exact slippage for our specific ticker, timestamp, order size, and order type.

What it is good for:

- broker/venue execution quality benchmarking
- price improvement, effective spread, realized spread, and speed context
- choosing brokers/order handling rules for paper/live testing

What it is not good for:

- replacing actual fill logs
- measuring our exact strategy's next-open fill quality
- validating thin small-cap names where order timing and quote depth dominate

Useful public references:

- FINRA Rule 605 directory: https://www.finra.org/filing-reporting/regulation-nms/sec-rule-605-reports
- SEC Rule 605 FAQ: https://www.sec.gov/rules-regulations/staff-guidance/trading-markets-frequently-asked-questions/frequently-asked-questions-rule-605-regulation-nms
- CFR text: https://www.law.cornell.edu/cfr/text/17/242.605

## How To Use This In The Model

After at least 30 to 50 paper fills, run `src.slippage_reality_check` and replace the paper model's bps constants with the recommendation:

```bash
python3 -m src.paper_trading_model \
  --panel reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv \
  --out-dir reports/paper_trading_model_measured_slippage \
  --entry-slippage-bps <measured_entry_bps> \
  --exit-slippage-bps <measured_exit_bps> \
  --fixed-cost-bps <measured_fixed_bps>
```

Use p75 slippage as the main paper model and p90 as the kill-switch stress model.
