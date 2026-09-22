# Research Plan

1. Build point-in-time universe with delisted companies and security-type exclusions.
2. Ingest SEC filings and normalize `information_available_timestamp`.
3. Manually audit first public announcement dates for transformative transactions.
4. Create company-date observations and multi-horizon labels.
5. Build market, SEC-text, fundamental, capital-structure, insider, short-interest, and corporate-structure features.
6. Run leakage and data-quality checks before modeling.
7. Train simple baselines first.
8. Run walk-forward validation, ablations, placebo tests, and realistic execution backtests.
9. Produce the AEMD case study using the same feature pipeline as all other companies.
10. Only create a live scanner if out-of-sample evidence survives placebo and cost tests.
