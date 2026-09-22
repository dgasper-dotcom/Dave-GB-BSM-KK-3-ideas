# Real Slippage Reality Check

This report converts actual fills and/or quote snapshots into bps assumptions that can replace the fixed slippage constants in the paper model.

## Recommended Cost Inputs

| method            | entry_slippage_bps   | exit_slippage_bps   |   fixed_cost_bps |   conservative_quantile |
|:------------------|:---------------------|:--------------------|-----------------:|------------------------:|
| insufficient_data |                      |                     |           0.0000 |                  0.7500 |

## Fill Summary

_No rows._

## Quote Summary

No quote observations were provided.

## Data Caveats

- Actual broker fills are the highest-quality slippage data because they include routing, order type, price improvement, partial fills, and rejects.
- NBBO/L2 quote snapshots estimate available liquidity and spread cost, but they do not prove where a broker would have routed or filled an order.
- SEC Rule 605 data is useful for public execution-quality benchmarking by broker/venue, but it is monthly aggregate data rather than our exact order-level slippage.