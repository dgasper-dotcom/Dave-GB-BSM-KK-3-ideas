from __future__ import annotations

import numpy as np
import pandas as pd


def apply_execution_costs(
    trades: pd.DataFrame,
    spread_bps_col: str = "spread_bps",
    slippage_bps: float = 75,
    commissions_bps: float = 0,
) -> pd.DataFrame:
    out = trades.copy()
    spread = out.get(spread_bps_col, pd.Series(np.nan, index=out.index)).fillna(150)
    round_trip_cost = (spread + 2 * slippage_bps + commissions_bps) / 10000.0
    out["net_return"] = out["gross_return"] - round_trip_cost
    out["execution_cost"] = round_trip_cost
    return out


def rank_strategy(
    predictions: pd.DataFrame,
    prices: pd.DataFrame,
    score_col: str,
    holding_days: int = 20,
    top_k: int = 10,
    max_pct_adv: float = 0.05,
    min_dollar_volume: float = 100000,
) -> pd.DataFrame:
    pred = predictions.copy()
    pred["date"] = pd.to_datetime(pred["date"])
    px = prices.copy()
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["ticker", "date"])
    px["exit_close"] = px.groupby("ticker")["close"].shift(-holding_days)
    px["gross_return"] = px["exit_close"] / px["close"] - 1
    merged = pred.merge(px[["ticker", "date", "close", "exit_close", "gross_return", "adv_20d_dollars"]], on=["ticker", "date"], how="left")
    eligible = merged[
        merged["adv_20d_dollars"].fillna(0).ge(min_dollar_volume)
        & merged["close"].notna()
        & merged["exit_close"].notna()
    ].copy()
    eligible["rank"] = eligible.groupby("date")[score_col].rank(ascending=False, method="first")
    trades = eligible[eligible["rank"] <= top_k].copy()
    trades["max_position_dollars"] = trades["adv_20d_dollars"] * max_pct_adv
    return apply_execution_costs(trades)


def return_distribution(trades: pd.DataFrame, return_col: str = "net_return") -> dict[str, float]:
    r = trades[return_col].dropna()
    if r.empty:
        return {}
    equity = (1 + r).cumprod()
    dd = equity / equity.cummax() - 1
    return {
        "trades": float(len(r)),
        "mean": float(r.mean()),
        "median": float(r.median()),
        "p25": float(r.quantile(0.25)),
        "p75": float(r.quantile(0.75)),
        "p90": float(r.quantile(0.90)),
        "p95": float(r.quantile(0.95)),
        "max": float(r.max()),
        "max_drawdown": float(dd.min()),
        "pct_losing": float((r < 0).mean()),
        "pct_gt_50": float((r > 0.50).mean()),
        "pct_gt_100": float((r > 1.00).mean()),
        "pct_gt_200": float((r > 2.00).mean()),
        "pct_gt_400": float((r > 4.00).mean()),
    }


def placebo_scores(predictions: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    out = predictions.copy()
    out["random_score"] = rng.random(len(out))
    if "market_cap" in out.columns:
        out["market_cap_only_score"] = -out["market_cap"].rank(pct=True)
    if "volume_to_20d_avg" in out.columns:
        out["volume_only_score"] = out["volume_to_20d_avg"].rank(pct=True)
    keyword_cols = [c for c in out.columns if c.endswith("_hit_90d") or c.endswith("_hit_30d")]
    if keyword_cols:
        out["simple_keyword_score"] = out[keyword_cols].fillna(0).sum(axis=1)
    return out
