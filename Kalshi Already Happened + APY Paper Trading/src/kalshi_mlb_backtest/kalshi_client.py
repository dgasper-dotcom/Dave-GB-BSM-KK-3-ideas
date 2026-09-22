from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


KALSHI_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"
DEFAULT_USER_AGENT = "kalshi-mlb-player-prop-backtest/1.0"


class KalshiApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeriesFeeInfo:
    series_ticker: str
    fee_type: str | None
    fee_multiplier: float | None
    raw: dict[str, Any]


class KalshiClient:
    def __init__(
        self,
        base_url: str = KALSHI_BASE_URL,
        timeout: float = 30.0,
        max_retries: int = 4,
        retry_sleep: float = 0.75,
        user_agent: str = DEFAULT_USER_AGENT,
        api_key_id: str | None = None,
        private_key_path: str | Path | None = None,
        private_key_pem: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.api_key_id = api_key_id or os.getenv("KALSHI_ACCESS_KEY") or os.getenv("KALSHI_API_KEY_ID")
        self.private_key = self._load_private_key(private_key_path, private_key_pem)

    def _load_private_key(
        self,
        private_key_path: str | Path | None,
        private_key_pem: str | None,
    ) -> rsa.RSAPrivateKey | None:
        raw_pem = private_key_pem or os.getenv("KALSHI_PRIVATE_KEY")
        env_key_path = os.getenv("KALSHI_PRIVATE_KEY_FILE") or os.getenv("KALSHI_PRIVATE_KEY_PATH")
        key_path = Path(private_key_path or env_key_path).expanduser() if (private_key_path or env_key_path) else None
        if raw_pem:
            payload = raw_pem.replace("\\n", "\n").encode("utf-8")
        elif key_path is not None:
            payload = key_path.read_bytes()
        else:
            return None
        key = serialization.load_pem_private_key(payload, password=None, backend=default_backend())
        if not isinstance(key, rsa.RSAPrivateKey):
            raise KalshiApiError("Kalshi private key is not an RSA private key")
        return key

    @property
    def has_auth(self) -> bool:
        return bool(self.api_key_id and self.private_key)

    def _auth_headers(self, method: str, path: str) -> dict[str, str]:
        if not self.api_key_id or self.private_key is None:
            raise KalshiApiError(
                "Kalshi authentication is required for this endpoint. Set KALSHI_ACCESS_KEY "
                "and KALSHI_PRIVATE_KEY_FILE, KALSHI_PRIVATE_KEY_PATH, or KALSHI_PRIVATE_KEY."
            )
        timestamp = str(int(time.time() * 1000))
        full_path = urlparse(self.base_url + path).path
        message = f"{timestamp}{method.upper()}{full_path.split('?')[0]}".encode("utf-8")
        signature = self.private_key.sign(
            message,
            padding.PSS(mgf=padding.MGF1(hashes.SHA256()), salt_length=padding.PSS.DIGEST_LENGTH),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.api_key_id,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode("utf-8"),
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def get_json(self, path: str, params: dict[str, Any] | None = None, auth: bool = False) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        clean_params = {k: v for k, v in (params or {}).items() if v not in (None, "")}
        headers = self._auth_headers("GET", path) if auth else None
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = self.session.get(url, params=clean_params, headers=headers, timeout=self.timeout)
                if resp.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    retry_after = None
                    if resp.status_code == 429:
                        try:
                            retry_after = float(resp.headers.get("Retry-After") or "")
                        except ValueError:
                            retry_after = None
                    time.sleep(retry_after if retry_after is not None and retry_after > 0 else self.retry_sleep * (2**attempt))
                    continue
                if resp.status_code >= 400:
                    raise KalshiApiError(f"Kalshi HTTP {resp.status_code} {resp.url}: {resp.text[:500]}")
                data = resp.json()
                if not isinstance(data, dict):
                    raise KalshiApiError(f"Kalshi returned non-object JSON from {resp.url}")
                return data
            except (requests.RequestException, ValueError, KalshiApiError) as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
                break
        raise KalshiApiError(f"Kalshi request failed for {url}: {last_exc}") from last_exc

    def iter_pages(
        self,
        path: str,
        result_key: str,
        params: dict[str, Any] | None = None,
        limit: int = 1000,
        max_pages: int | None = None,
    ) -> Iterable[dict[str, Any]]:
        cursor: str | None = None
        pages = 0
        while True:
            page_params = dict(params or {})
            page_params["limit"] = limit
            if cursor:
                page_params["cursor"] = cursor
            payload = self.get_json(path, page_params)
            for item in payload.get(result_key) or []:
                if isinstance(item, dict):
                    yield item
            pages += 1
            cursor = str(payload.get("cursor") or "")
            if not cursor:
                break
            if max_pages is not None and pages >= max_pages:
                break

    def get_series_fee_info(self, series_ticker: str) -> SeriesFeeInfo:
        payload = self.get_json(f"/series/{series_ticker}")
        series = payload.get("series") or payload
        if not isinstance(series, dict):
            series = {}
        multiplier = series.get("fee_multiplier")
        return SeriesFeeInfo(
            series_ticker=series_ticker,
            fee_type=series.get("fee_type"),
            fee_multiplier=float(multiplier) if multiplier is not None else None,
            raw=series,
        )

    def get_series_fee_changes(self, series_ticker: str | None = None) -> list[dict[str, Any]]:
        params = {"show_historical": True}
        if series_ticker:
            params["series_ticker"] = series_ticker
        payload = self.get_json("/series/fee_changes", params)
        return [item for item in payload.get("series_fee_change_arr") or [] if isinstance(item, dict)]

    def fetch_markets(
        self,
        series_ticker: str,
        status: str,
        min_settled_ts: int | None = None,
        max_settled_ts: int | None = None,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "series_ticker": series_ticker,
            "status": status,
            "mve_filter": "exclude",
        }
        if min_settled_ts is not None:
            params["min_settled_ts"] = min_settled_ts
        if max_settled_ts is not None:
            params["max_settled_ts"] = max_settled_ts
        return list(self.iter_pages("/markets", "markets", params=params, limit=1000, max_pages=max_pages))

    def fetch_candlesticks_batch(
        self,
        tickers: list[str],
        start_ts: int,
        end_ts: int,
        period_interval: int = 1,
    ) -> dict[str, list[dict[str, Any]]]:
        if not tickers:
            return {}
        payload = self.get_json(
            "/markets/candlesticks",
            {
                "market_tickers": ",".join(tickers),
                "start_ts": start_ts,
                "end_ts": end_ts,
                "period_interval": period_interval,
            },
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for market_payload in payload.get("markets") or []:
            ticker = str(market_payload.get("market_ticker") or market_payload.get("ticker") or "")
            if not ticker and len(tickers) == 1:
                ticker = tickers[0]
            out[ticker] = [c for c in market_payload.get("candlesticks") or [] if isinstance(c, dict)]
        if len(tickers) == 1 and tickers[0] not in out and "candlesticks" in payload:
            out[tickers[0]] = [c for c in payload.get("candlesticks") or [] if isinstance(c, dict)]
        return out

    def fetch_trades(
        self,
        ticker: str,
        min_ts: int,
        max_ts: int,
        max_pages: int | None = None,
    ) -> list[dict[str, Any]]:
        params = {"ticker": ticker, "min_ts": min_ts, "max_ts": max_ts}
        return list(self.iter_pages("/markets/trades", "trades", params=params, limit=1000, max_pages=max_pages))

    def get_current_orderbook(self, ticker: str, depth: int = 100) -> dict[str, Any]:
        return self.get_json(f"/markets/{ticker}/orderbook", {"depth": depth}, auth=self.has_auth)
