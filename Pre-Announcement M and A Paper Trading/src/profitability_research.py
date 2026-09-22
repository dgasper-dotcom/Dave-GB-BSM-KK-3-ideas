from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .evaluation import rare_event_metrics


MARKET_FEATURES = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "volatility_5d",
    "volatility_20d",
    "volatility_60d",
    "volume_to_20d_avg",
    "volume_to_60d_avg",
    "gap",
    "intraday_range",
    "distance_from_52w_high",
    "distance_from_52w_low",
    "dollar_volume",
    "adv_20d_dollars",
    "price",
]


STOCKANALYSIS_HISTORY_URL = "https://stockanalysis.com/api/symbol/s/{symbol}/history"
STOCKANALYSIS_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://stockanalysis.com/",
}


def normalize_price_frame(data: pd.DataFrame, symbol: str, start: str, end: str) -> pd.DataFrame:
    if data.empty:
        return data
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    data = data.dropna(subset=["date"])
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    data = data[(data["date"] >= start_ts) & (data["date"] < end_ts)]
    data["ticker"] = symbol
    cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    for col in cols:
        if col not in data.columns:
            data[col] = np.nan
    data = data[cols].sort_values("date").drop_duplicates(["ticker", "date"], keep="last")
    return data


def download_symbol_yfinance(symbol: str, start: str, end: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{symbol}.csv"
    if path.exists():
        try:
            return normalize_price_frame(pd.read_csv(path, parse_dates=["date"]), symbol, start, end)
        except Exception:
            pass
    try:
        data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False, threads=False)
        if data.empty:
            return pd.DataFrame()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [c[0].lower().replace(" ", "_") for c in data.columns]
        else:
            data.columns = [str(c).lower().replace(" ", "_") for c in data.columns]
        data = data.reset_index().rename(columns={"Date": "date", "date": "date"})
        data["date"] = pd.to_datetime(data["date"]).dt.normalize()
        cols = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
        for col in cols:
            if col not in data.columns:
                data[col] = np.nan
        data = normalize_price_frame(data, symbol, start, end)
        cache_dir.mkdir(parents=True, exist_ok=True)
        data.to_csv(path, index=False)
        return data
    except Exception:
        return pd.DataFrame()


def stockanalysis_symbol_candidates(symbol: str) -> list[str]:
    clean = re.sub(r"[^A-Z0-9.\-]", "", symbol.upper())
    candidates = [clean, clean.lower()]
    if "." in clean:
        candidates.extend([clean.replace(".", "-"), clean.lower().replace(".", "-")])
    return list(dict.fromkeys(c for c in candidates if c))


def download_symbol_stockanalysis(symbol: str, start: str, end: str, cache_dir: Path, sleep_seconds: float = 0.0) -> pd.DataFrame:
    path = cache_dir / f"{symbol}.csv"
    if path.exists():
        try:
            return normalize_price_frame(pd.read_csv(path, parse_dates=["date"]), symbol, start, end)
        except Exception:
            pass
    for candidate in stockanalysis_symbol_candidates(symbol):
        url = STOCKANALYSIS_HISTORY_URL.format(symbol=candidate)
        try:
            response = requests.get(
                url,
                params={"range": "Max", "period": "Daily"},
                headers=STOCKANALYSIS_HEADERS,
                timeout=30,
            )
            if sleep_seconds:
                time.sleep(sleep_seconds)
            if response.status_code != 200:
                continue
            payload = response.json()
            rows = payload.get("data") or []
            if not rows:
                continue
            data = pd.DataFrame(rows).rename(
                columns={"t": "date", "o": "open", "h": "high", "l": "low", "c": "close", "a": "adj_close", "v": "volume"}
            )
            data = normalize_price_frame(data, symbol, start, end)
            if data.empty:
                continue
            cache_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(path, index=False)
            return data
        except Exception:
            continue
    return pd.DataFrame()


def download_symbol(
    symbol: str,
    start: str,
    end: str,
    cache_dir: Path,
    provider: str,
    sleep_seconds: float = 0.0,
) -> pd.DataFrame:
    if provider == "stockanalysis":
        return download_symbol_stockanalysis(symbol, start, end, cache_dir, sleep_seconds=sleep_seconds)
    if provider == "combined":
        data = download_symbol_stockanalysis(symbol, start, end, cache_dir / "stockanalysis", sleep_seconds=sleep_seconds)
        if not data.empty:
            return data
        return download_symbol_yfinance(symbol, start, end, cache_dir / "yfinance")
    return download_symbol_yfinance(symbol, start, end, cache_dir)


def market_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.copy()
    df = df.sort_values(["ticker", "date"])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    g = df.groupby("ticker", group_keys=False)
    for n in (1, 3, 5, 10, 20, 60):
        df[f"return_{n}d"] = g["close"].pct_change(n)
    daily_ret = g["close"].pct_change()
    for n in (5, 20, 60):
        df[f"volatility_{n}d"] = daily_ret.groupby(df["ticker"]).rolling(n).std().reset_index(level=0, drop=True)
    df["volume_to_20d_avg"] = df["volume"] / g["volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["volume_to_60d_avg"] = df["volume"] / g["volume"].transform(lambda s: s.rolling(60, min_periods=10).mean())
    df["gap"] = df["open"] / g["close"].shift(1) - 1
    df["intraday_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)
    df["high_252d"] = g["high"].transform(lambda s: s.rolling(252, min_periods=20).max())
    df["low_252d"] = g["low"].transform(lambda s: s.rolling(252, min_periods=20).min())
    df["distance_from_52w_high"] = df["close"] / df["high_252d"] - 1
    df["distance_from_52w_low"] = df["close"] / df["low_252d"] - 1
    df["dollar_volume"] = df["close"] * df["volume"]
    df["adv_20d_dollars"] = g["dollar_volume"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    df["price"] = df["close"]
    df["next_open"] = g["open"].shift(-1)
    for h in (5, 10, 20):
        df[f"exit_close_{h}d"] = g["close"].shift(-h)
        df[f"fwd_return_{h}d"] = df[f"exit_close_{h}d"] / df["next_open"] - 1
    return df


def build_panel(
    cohort_path: Path,
    audited_events_path: Path,
    cache_dir: Path,
    start: str,
    end: str,
    max_symbols: int | None,
    provider: str,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    cohort = pd.read_csv(cohort_path, dtype={"cik": str})
    events = pd.read_csv(audited_events_path, dtype={"cik": str})
    usable_events = events[events["audited_announcement_ts"].notna()].copy()
    usable_events["announcement_date"] = pd.to_datetime(usable_events["audited_announcement_ts"], utc=True).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    event_map = dict(zip(usable_events["ticker"], usable_events["announcement_date"]))

    symbols = cohort["symbol"].dropna().astype(str).str.upper().drop_duplicates().tolist()
    if max_symbols:
        symbols = symbols[:max_symbols]
    frames = []
    failed = []
    for i, symbol in enumerate(symbols, start=1):
        df = download_symbol(symbol, start, end, cache_dir, provider=provider, sleep_seconds=sleep_seconds)
        if df.empty or len(df) < 80:
            failed.append(symbol)
        else:
            frames.append(df)
        if i % 100 == 0:
            print(f"downloaded_symbols={i} usable={len(frames)} failed={len(failed)}")
    if not frames:
        raise ValueError("No market data downloaded.")
    prices = pd.concat(frames, ignore_index=True)
    features = market_features(prices)

    features["event_date"] = features["ticker"].map(event_map)
    features["has_event"] = features["event_date"].notna().astype(int)
    features["date"] = pd.to_datetime(features["date"]).dt.normalize()
    features["days_to_event_calendar"] = (pd.to_datetime(features["event_date"]) - features["date"]).dt.days
    for h in (5, 10, 20, 40, 60, 90):
        features[f"event_{h}d"] = ((features["days_to_event_calendar"] > 0) & (features["days_to_event_calendar"] <= math.ceil(h * 1.6))).astype(int)
    # Remove post-announcement rows for event companies.
    features = features[(features["event_date"].isna()) | (features["date"] < pd.to_datetime(features["event_date"]))]
    features = features.dropna(subset=["close", "next_open", "adv_20d_dollars"])
    summary = {
        "cohort_symbols": len(symbols),
        "symbols_with_market_data": int(features["ticker"].nunique()),
        "failed_symbols": len(failed),
        "usable_event_symbols": int(features.loc[features["has_event"].eq(1), "ticker"].nunique()),
        "rows": int(len(features)),
        "date_min": str(features["date"].min().date()),
        "date_max": str(features["date"].max().date()),
        "market_data_provider": provider,
        "failed_sample": failed[:30],
    }
    return features, summary


def make_model(name: str) -> Pipeline:
    if name == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(class_weight="balanced", max_iter=2000)),
            ]
        )
    if name == "rf":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=25, class_weight="balanced_subsample", n_jobs=-1, random_state=42)),
            ]
        )
    if name == "hgb":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.03, l2_regularization=1.0, random_state=42)),
            ]
        )
    raise ValueError(name)


def chronological_splits(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = panel[panel["date"] < pd.Timestamp("2026-01-01")]
    val = panel[(panel["date"] >= pd.Timestamp("2026-01-01")) & (panel["date"] < pd.Timestamp("2026-06-01"))]
    test = panel[panel["date"] >= pd.Timestamp("2026-06-01")]
    return train, val, test


def score_model(model: Pipeline, sample: pd.DataFrame, target: str) -> pd.DataFrame:
    out = sample[["ticker", "date", "event_date", target, "next_open", "close", "adv_20d_dollars", "fwd_return_20d"]].copy()
    out["score"] = model.predict_proba(sample[MARKET_FEATURES])[:, 1]
    return out


def rank_backtest(
    predictions: pd.DataFrame,
    top_k: int,
    score_threshold: float,
    max_pct_adv: float,
    min_adv: float,
    cost_bps: float,
    hold_col: str = "fwd_return_20d",
) -> dict[str, float]:
    df = predictions.copy()
    df = df[df["adv_20d_dollars"].fillna(0) >= min_adv]
    df = df[df["score"] >= score_threshold]
    if df.empty:
        return {"trades": 0}
    df["rank"] = df.groupby("date")["score"].rank(ascending=False, method="first")
    trades = df[df["rank"] <= top_k].copy()
    trades = trades.dropna(subset=[hold_col])
    if trades.empty:
        return {"trades": 0}
    trades["net_return"] = trades[hold_col] - cost_bps / 10000.0
    daily = trades.groupby("date")["net_return"].mean().sort_index()
    equity = (1 + daily).cumprod()
    dd = equity / equity.cummax() - 1
    years = max((daily.index.max() - daily.index.min()).days / 365.25, 1 / 365.25)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if len(equity) else np.nan
    return {
        "trades": int(len(trades)),
        "active_days": int(len(daily)),
        "mean_trade_return": float(trades["net_return"].mean()),
        "median_trade_return": float(trades["net_return"].median()),
        "win_rate": float((trades["net_return"] > 0).mean()),
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "max_drawdown": float(dd.min()),
        "top_k": top_k,
        "threshold": score_threshold,
        "min_adv": min_adv,
        "cost_bps": cost_bps,
    }


def train_and_backtest(panel: pd.DataFrame, out_dir: Path, target: str = "event_20d") -> dict[str, object]:
    train, val, test = chronological_splits(panel)
    rows = []
    models = {}
    for name in ("logistic", "rf", "hgb"):
        model = make_model(name)
        fit_train = train.dropna(subset=[target]).copy()
        if fit_train[target].nunique() < 2:
            continue
        model.fit(fit_train[MARKET_FEATURES], fit_train[target].astype(int))
        models[name] = model
        for sample_name, sample in (("validation", val), ("test", test)):
            if sample.empty or sample[target].nunique() < 2:
                continue
            pred = score_model(model, sample, target)
            metrics = rare_event_metrics(pred[target].to_numpy(), pred["score"].to_numpy())
            metrics.update({"model": name, "sample": sample_name})
            rows.append(metrics)
            pred.to_csv(out_dir / f"{name}_{sample_name}_predictions.csv", index=False)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "classification_metrics.csv", index=False)

    search_rows = []
    best = None
    best_model = None
    for name in models:
        val_pred = pd.read_csv(out_dir / f"{name}_validation_predictions.csv", parse_dates=["date"])
        for top_k in (1, 3, 5, 10):
            for q in (0.90, 0.95, 0.98, 0.99):
                threshold = float(val_pred["score"].quantile(q))
                bt = rank_backtest(val_pred, top_k, threshold, max_pct_adv=0.03, min_adv=100_000, cost_bps=250)
                bt.update({"model": name, "sample": "validation", "quantile": q})
                search_rows.append(bt)
                if bt.get("trades", 0) >= 20 and (best is None or bt.get("total_return", -999) > best.get("total_return", -999)):
                    best = bt
                    best_model = name
    search = pd.DataFrame(search_rows)
    search.to_csv(out_dir / "validation_backtest_search.csv", index=False)

    test_result = {}
    if best and best_model:
        test_pred = pd.read_csv(out_dir / f"{best_model}_test_predictions.csv", parse_dates=["date"])
        test_result = rank_backtest(
            test_pred,
            int(best["top_k"]),
            float(best["threshold"]),
            max_pct_adv=0.03,
            min_adv=float(best["min_adv"]),
            cost_bps=float(best["cost_bps"]),
        )
        test_result.update({"model": best_model, "sample": "test", "selected_from_validation": best})
        pd.DataFrame([test_result]).to_csv(out_dir / "selected_test_backtest.csv", index=False)
        joblib.dump(models[best_model], out_dir / "selected_model.joblib")
    return {
        "classification_metrics": metrics.to_dict("records"),
        "best_validation": best,
        "selected_test": test_result,
        "rows": int(len(panel)),
        "train_rows": int(len(train)),
        "validation_rows": int(len(val)),
        "test_rows": int(len(test)),
        "train_events": int(train[target].sum()),
        "validation_events": int(val[target].sum()),
        "test_events": int(test[target].sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", default="data/processed/cohort/training_company_cohort.csv")
    parser.add_argument("--events", default="data/processed/cohort/events_sec_audited.csv")
    parser.add_argument("--out-dir", default="reports/profitability_research")
    parser.add_argument("--price-cache", default="data/raw/yfinance")
    parser.add_argument("--provider", choices=["yfinance", "stockanalysis", "combined"], default="yfinance")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2026-09-18")
    parser.add_argument("--max-symbols", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel, summary = build_panel(
        Path(args.cohort),
        Path(args.events),
        Path(args.price_cache),
        args.start,
        args.end,
        args.max_symbols,
        args.provider,
        args.sleep_seconds,
    )
    panel.to_csv(out_dir / "market_feature_panel.csv", index=False)
    result = train_and_backtest(panel, out_dir)
    result["data_summary"] = summary
    (out_dir / "profitability_summary.json").write_text(json.dumps(result, indent=2, default=str))
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
