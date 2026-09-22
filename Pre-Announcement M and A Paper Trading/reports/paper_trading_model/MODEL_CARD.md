# Paper Trading Model

Panel: `reports/backfill_2018_2026_sec_fusion_cap8/fused_market_sec_panel.csv`
Signal date: `2026-09-16`
Training cutoff: outcomes known before `2026-09-16`

This is a paper-testing scanner, not a live trading recommendation. It is designed to abstain unless the return-aware model expects nonnegative harsh net return after adverse entry, exit, and fixed-cost assumptions.

## Decision

- Paper orders: `0`
- Watchlist rows: `50`
- Decision reason: Selected calibration strategy passed; no current names met all gates.

## Paper Orders
_No rows._

## Watchlist
| ticker   | date                |    close |   next_open | paper_action   |   event_score |   expected_net_return_20d |   positive_return_score |   execution_score | paper_entry_eligible   | no_order_reason                                                       |
|:---------|:--------------------|---------:|------------:|:---------------|--------------:|--------------------------:|------------------------:|------------------:|:-----------------------|:----------------------------------------------------------------------|
| AIRO     | 2026-09-16 00:00:00 |   7.1900 |      7.3200 | WATCH_ONLY     |        0.0122 |                    0.0525 |                  0.5312 |            0.0031 | True                   | Selected calibration strategy passed; no current names met all gates. |
| VENU     | 2026-09-16 00:00:00 |   1.6900 |      1.7300 | WATCH_ONLY     |        0.0093 |                    0.0418 |                  0.4819 |            0.0019 | True                   | Selected calibration strategy passed; no current names met all gates. |
| FLY      | 2026-09-16 00:00:00 |  20.2800 |     20.9800 | WATCH_ONLY     |        0.0089 |                    0.0426 |                  0.4649 |            0.0019 | True                   | Selected calibration strategy passed; no current names met all gates. |
| QNCX     | 2026-09-16 00:00:00 |  27.1700 |     27.1700 | WATCH_ONLY     |        0.0207 |                    0.0260 |                  0.4491 |            0.0017 | True                   | Selected calibration strategy passed; no current names met all gates. |
| SHAZ     | 2026-09-16 00:00:00 |  52.2100 |     55.1700 | WATCH_ONLY     |        0.0172 |                    0.0268 |                  0.4639 |            0.0016 | True                   | Selected calibration strategy passed; no current names met all gates. |
| GPRO     | 2026-09-16 00:00:00 |   1.2700 |      1.2900 | WATCH_ONLY     |        0.0205 |                    0.0243 |                  0.3942 |            0.0014 | True                   | Selected calibration strategy passed; no current names met all gates. |
| LVLU     | 2026-09-16 00:00:00 |   9.9000 |      9.8700 | WATCH_ONLY     |        0.0218 |                    0.0201 |                  0.4431 |            0.0013 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ADVB     | 2026-09-16 00:00:00 |   7.7200 |      7.7600 | WATCH_ONLY     |        0.0171 |                    0.0267 |                  0.2911 |            0.0010 | True                   | Selected calibration strategy passed; no current names met all gates. |
| MSS      | 2026-09-16 00:00:00 |   1.5200 |      1.5400 | WATCH_ONLY     |        0.0093 |                    0.0239 |                  0.4115 |            0.0009 | True                   | Selected calibration strategy passed; no current names met all gates. |
| PZG      | 2026-09-16 00:00:00 |   1.2800 |      1.3200 | WATCH_ONLY     |        0.0238 |                    0.0070 |                  0.4270 |            0.0005 | True                   | Selected calibration strategy passed; no current names met all gates. |
| OPTU     | 2026-09-16 00:00:00 |   1.0100 |      1.0200 | WATCH_ONLY     |        0.0165 |                    0.0082 |                  0.4047 |            0.0004 | True                   | Selected calibration strategy passed; no current names met all gates. |
| CDXS     | 2026-09-16 00:00:00 |   1.3400 |      1.3800 | WATCH_ONLY     |        0.0114 |                    0.0069 |                  0.4476 |            0.0003 | True                   | Selected calibration strategy passed; no current names met all gates. |
| CAVA     | 2026-09-16 00:00:00 |  49.7600 |     51.3100 | WATCH_ONLY     |        0.0101 |                    0.0062 |                  0.4672 |            0.0003 | True                   | Selected calibration strategy passed; no current names met all gates. |
| NAK      | 2026-09-16 00:00:00 |   1.2400 |      1.2900 | WATCH_ONLY     |        0.0041 |                    0.0090 |                  0.4390 |            0.0003 | True                   | Selected calibration strategy passed; no current names met all gates. |
| MAIR     | 2026-09-16 00:00:00 |  23.4800 |     23.9800 | WATCH_ONLY     |        0.0115 |                    0.0048 |                  0.3540 |            0.0002 | True                   | Selected calibration strategy passed; no current names met all gates. |
| CAI      | 2026-09-16 00:00:00 |  28.2100 |     28.5950 | WATCH_ONLY     |        0.0203 |                    0.0025 |                  0.4345 |            0.0002 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACVA     | 2026-09-16 00:00:00 |  10.4200 |     10.4600 | WATCH_ONLY     |        0.0267 |                    0.0023 |                  0.3680 |            0.0001 | True                   | Selected calibration strategy passed; no current names met all gates. |
| AMBQ     | 2026-09-16 00:00:00 |  59.9000 |     61.7600 | WATCH_ONLY     |        0.0111 |                    0.0023 |                  0.4364 |            0.0001 | True                   | Selected calibration strategy passed; no current names met all gates. |
| AAOI     | 2026-09-16 00:00:00 |  96.7400 |    100.7250 | WATCH_ONLY     |        0.0068 |                   -0.0127 |                  0.4404 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| AB       | 2026-09-16 00:00:00 |  35.5000 |     35.8800 | WATCH_ONLY     |        0.0090 |                   -0.0247 |                  0.2885 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACB      | 2026-09-16 00:00:00 |   3.8300 |      3.8700 | WATCH_ONLY     |        0.0041 |                   -0.0117 |                  0.4334 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACFN     | 2026-09-16 00:00:00 |  20.7800 |     21.1900 | WATCH_ONLY     |        0.0208 |                   -0.0158 |                  0.4090 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACGL     | 2026-09-16 00:00:00 |  96.9700 |     97.0000 | WATCH_ONLY     |        0.0065 |                   -0.0247 |                  0.3134 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACH      | 2026-09-16 00:00:00 |   1.0700 |      1.0800 | WATCH_ONLY     |        0.0102 |                   -0.0112 |                  0.4276 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |
| ACLS     | 2026-09-16 00:00:00 | 103.7500 |    106.1600 | WATCH_ONLY     |        0.0095 |                   -0.0068 |                  0.4476 |            0.0000 | True                   | Selected calibration strategy passed; no current names met all gates. |

## Operating Rules

- Score after the signal-date close.
- Paper-buy at the next open only if gap, liquidity, price, and spread-proxy checks still pass.
- Position size is 1% of the paper account by default.
- Exit at the 20-trading-day close for apples-to-apples evaluation.
- Log actual entry/exit prices, bid/ask/mid snapshots, order type, limit price, partial fills, rejects, borrow/halts/news, and actual exit price.
- Do not promote to live trading until forward paper results are positive after costs and robust after removing top winners.
