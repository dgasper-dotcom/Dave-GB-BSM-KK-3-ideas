# Transformative Transaction ML Research System

This project is a point-in-time research platform for testing whether public information can identify U.S. small-cap and micro-cap companies with elevated historical probability of publicly announced transformative transactions.

It is deliberately built to answer "is the signal real?" rather than to make a single case look good. AEMD is supported as a case study, but the feature code and labels are general.

## What Is Implemented

- SEC EDGAR ingestion with local caching, accession-level metadata, parsed text, and explicit `information_available_timestamp`.
- Point-in-time joins that reject future source data.
- Filing NLP features for strategic review, investment bank, change-of-control, special committee, unsolicited interest, multiple-party interest, transaction, confidentiality, and financing language.
- Market, fundamental, insider/corporate-structure hooks with timestamp-aware feature timestamps.
- Multi-horizon labels for 5, 10, 20, 40, 60, and 90 trading-day event windows.
- Chronological and walk-forward model training utilities.
- Baselines: logistic regression, random forest, gradient boosting fallback, optional XGBoost/LightGBM/CatBoost, MLP, text-only, and grouped ablations.
- Evaluation metrics for rare events: base rate, PR-AUC, ROC-AUC, Brier score, precision/recall/F1, precision@K, recall@K, and lift.
- Backtest utilities with liquidity, spread, slippage, max percent of ADV, and event-day exit modes.
- Placebo/random signal comparison and ablation helpers.
- AEMD case-study script that shows features at 90/60/40/20/10/5/1 trading days before an event date without using later information.
- Leakage/data-quality tests.

## First Principles

Every feature row has:

- `ticker`
- `cik`
- `date`
- `feature_timestamp`
- source-specific timestamps such as `filing_available_ts`
- label event date, when applicable

The pipeline enforces:

```text
information_available_timestamp <= prediction_timestamp
```

If an SEC filing is accepted after 4:00 p.m. ET and the prediction timestamp is end-of-day, the filing is treated as available on the next trading session unless a custom prediction timestamp explicitly permits after-hours data.

## Suggested Workflow

1. Configure `config/config.yaml`, especially the SEC `user_agent`.
2. Build or import a point-in-time universe, including delisted names. A seed cohort can be created with:

```bash
python -m src.build_training_cohort \
  --start-year 2021 \
  --end-year 2026 \
  --target-positives 250 \
  --controls-per-positive 3 \
  --out-dir data/processed/cohort
```

This creates positive acquisition/merger seeds plus non-event controls. Treat StockAnalysis acquisition dates as provisional action/closing-style dates until SEC filings or press releases establish the first public announcement timestamp.

3. Ingest SEC filings:

```bash
python -m src.sec_ingestion --cik 0000882291 --out data/raw/sec --forms 8-K 10-Q 10-K DEF\ 14A S-1 S-3 S-4 424B
```

To re-run the SEC fusion model after changing phrase dictionaries, rebuild from cached filing HTML:

```bash
python -m src.rebuild_sec_fusion_from_cache \
  --market-panel reports/profitability_full_stockanalysis/market_feature_panel.csv \
  --filing-index reports/profitability_sec_market_fusion_full/sec_filings_index.csv \
  --out-dir reports/profitability_sec_market_fusion_full_strategy_review
```

4. Build event labels from a manually audited first-announcement table:

```text
ticker,cik,event_type,announcement_ts,announcement_source,announcement_url
```

5. Build feature panels, run leakage checks, then train baselines before complex models.
6. Use chronological validation and keep a final out-of-sample period untouched.

## AEMD Case Study

Prepare:

- `data/processed/feature_panel.parquet`
- `data/processed/events.csv`
- `data/processed/trading_calendar.csv`

Then run:

```bash
python -m src.aemd_case_study \
  --features data/processed/feature_panel.parquet \
  --events data/processed/events.csv \
  --calendar data/processed/trading_calendar.csv \
  --out reports/aemd_case_study.md
```

The report only displays features whose timestamps existed as of each pre-announcement date.

## Important Limits

This repository is the research system and anti-leakage harness. It does not ship a fully backfilled commercial-grade historical universe, short-interest history, institutional ownership history, bid/ask history, or delisting database. Those sources must be added through timestamped adapters before serious claims are made.

The correct final answer may be:

```text
No robust evidence of a predictive signal.
```

That outcome is treated as a valid research result.
