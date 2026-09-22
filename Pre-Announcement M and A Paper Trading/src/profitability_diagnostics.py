from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def nonoverlap_20d_backtest(
    predictions: pd.DataFrame,
    top_k: int,
    score_threshold: float,
    min_adv: float = 100_000,
    cost_bps: float = 250,
    hold_col: str = "fwd_return_20d",
    holding_days: int = 20,
) -> dict[str, float]:
    df = predictions.copy()
    df = df[df["adv_20d_dollars"].fillna(0).ge(min_adv)]
    df = df[df["score"].ge(score_threshold)]
    df = df.dropna(subset=[hold_col])
    if df.empty:
        return {"trades": 0}

    all_dates = pd.Index(sorted(pd.to_datetime(predictions["date"]).dropna().unique()))
    rebalance_dates = set(all_dates[::holding_days])
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"].isin(rebalance_dates)].copy()
    if df.empty:
        return {"trades": 0}

    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="first")
    trades = df[df["rank"].le(top_k)].copy()
    if trades.empty:
        return {"trades": 0}

    trades["net_return"] = trades[hold_col] - cost_bps / 10000.0
    period_returns = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + period_returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    years = max((period_returns.index.max() - period_returns.index.min()).days / 365.25, 1 / 365.25)
    return {
        "trades": int(len(trades)),
        "active_periods": int(len(period_returns)),
        "mean_trade_return": float(trades["net_return"].mean()),
        "median_trade_return": float(trades["net_return"].median()),
        "win_rate": float(trades["net_return"].gt(0).mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": float(equity.iloc[-1] ** (1 / years) - 1),
        "max_drawdown": float(drawdown.min()),
        "top_k": top_k,
        "threshold": score_threshold,
        "min_adv": min_adv,
        "cost_bps": cost_bps,
    }


def build_grid(run_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for sample in ("validation", "test"):
        for model in ("logistic", "rf", "hgb"):
            path = run_dir / f"{model}_{sample}_predictions.csv"
            if not path.exists():
                continue
            predictions = pd.read_csv(path, parse_dates=["date"])
            for top_k in (1, 3, 5, 10):
                for quantile in (0.90, 0.95, 0.98, 0.99):
                    threshold = float(predictions["score"].quantile(quantile))
                    result = nonoverlap_20d_backtest(predictions, top_k=top_k, score_threshold=threshold)
                    result.update({"model": model, "sample": sample, "quantile": quantile})
                    rows.append(result)
    grid = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(out_path, index=False)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="reports/profitability_full_stockanalysis")
    parser.add_argument("--out", default="reports/profitability_full_stockanalysis/nonoverlap_20d_backtest_grid.csv")
    args = parser.parse_args()
    grid = build_grid(Path(args.run_dir), Path(args.out))
    eligible = grid[(grid["sample"].eq("validation")) & (grid["trades"].ge(10))]
    if eligible.empty:
        print("No validation strategy met the minimum trade count.")
        return
    best = eligible.sort_values("total_return", ascending=False).head(1)
    print(best.to_string(index=False))


if __name__ == "__main__":
    main()
