from __future__ import annotations

import numpy as np
import pandas as pd


def market_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"])
    group = df.groupby("ticker", group_keys=False)

    for n in (1, 3, 5, 10, 20, 60):
        df[f"return_{n}d"] = group["close"].pct_change(n)
    returns = group["close"].pct_change()
    for n in (5, 20, 60):
        df[f"volatility_{n}d"] = returns.groupby(df["ticker"]).rolling(n).std().reset_index(level=0, drop=True)
        df[f"volume_to_{n}d_avg"] = df["volume"] / group["volume"].transform(lambda s: s.rolling(n).mean())

    df["gap"] = df["open"] / group["close"].shift(1) - 1
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["high_252d"] = group["high"].transform(lambda s: s.rolling(252, min_periods=20).max())
    df["low_252d"] = group["low"].transform(lambda s: s.rolling(252, min_periods=20).min())
    df["distance_from_52w_high"] = df["close"] / df["high_252d"] - 1
    df["distance_from_52w_low"] = df["close"] / df["low_252d"] - 1
    df["dollar_volume"] = df["close"] * df["volume"]
    df["adv_20d_dollars"] = group["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["feature_timestamp"] = pd.to_datetime(df["date"]).dt.tz_localize("America/New_York") + pd.Timedelta(hours=16)
    df["feature_timestamp"] = df["feature_timestamp"].dt.tz_convert("UTC")
    return df


def abnormal_returns(features: pd.DataFrame, benchmark: pd.DataFrame, benchmark_ticker: str = "SPY") -> pd.DataFrame:
    left = features.copy()
    bench = benchmark[benchmark["ticker"].eq(benchmark_ticker)].copy()
    bench = market_features(bench)[["date", "return_1d", "return_5d", "return_20d", "return_60d"]]
    bench = bench.rename(columns={c: f"{benchmark_ticker.lower()}_{c}" for c in bench.columns if c != "date"})
    out = left.merge(bench, on="date", how="left")
    for n in (1, 5, 20, 60):
        out[f"abnormal_return_{n}d_vs_{benchmark_ticker.lower()}"] = out[f"return_{n}d"] - out[
            f"{benchmark_ticker.lower()}_return_{n}d"
        ]
    return out
