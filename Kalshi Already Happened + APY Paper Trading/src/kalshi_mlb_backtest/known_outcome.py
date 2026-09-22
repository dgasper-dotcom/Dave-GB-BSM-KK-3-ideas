from __future__ import annotations

import csv
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

from .kalshi_client import KalshiApiError, KalshiClient
from .market_matcher import (
    MatchedMarket,
    is_supported_player_prop_market,
    market_series_ticker,
    match_open_market_to_game_and_player,
)
from .mlb_data import GameContext, MlbClient
from .orderbook import quote_from_exact_orderbook
from .pnl import kalshi_taker_fee
from .utils import UTC, iso_utc, normalize_name, parse_float, unix_seconds


SECONDS_PER_YEAR = 365.25 * 24.0 * 60.0 * 60.0
MONTHS = {
    "JAN": 1,
    "FEB": 2,
    "MAR": 3,
    "APR": 4,
    "MAY": 5,
    "JUN": 6,
    "JUL": 7,
    "AUG": 8,
    "SEP": 9,
    "OCT": 10,
    "NOV": 11,
    "DEC": 12,
}

DEFAULT_KNOWN_OUTCOME_SERIES = ",".join(
    (
        "KXMLBHIT",
        "KXMLBHR",
        "KXMLBGAME",
        "KXMLBSPREAD",
        "KXMLBTOTAL",
        "KXMLBTEAMTOTAL",
        "KXNFLGAME",
        "KXNFLSPREAD",
        "KXNFLTOTAL",
        "KXNFLTEAMTOTAL",
        "KXNCAAFGAME",
        "KXNCAAFSPREAD",
        "KXNCAAFTOTAL",
        "KXNCAAFTEAMTOTAL",
        "KXNBAGAME",
        "KXNBASPREAD",
        "KXNBATOTAL",
        "KXNBATEAMTOTAL",
        "KXNCAAMBGAME",
        "KXNHLGAME",
        "KXNHLSPREAD",
        "KXNHLTOTAL",
        "KXWNBAGAME",
        "KXWNBASPREAD",
        "KXWNBATOTAL",
        "KXWNBATEAMTOTAL",
        "KXTEMPNYCH",
        "KXTEMPCHIH",
        "KXTEMPDCH",
        "KXTEMPLAXH",
        "KXTEMPMIAH",
        "KXTEMPAUSH",
        "KXHIGHNY",
        "KXHIGHCHI",
        "KXHIGHDEN",
        "KXHIGHLAX",
        "KXHIGHMIA",
        "KXHIGHPHIL",
        "KXHIGHAUS",
        "KXINX",
        "KXNASDAQ100",
    )
)


@dataclass(frozen=True)
class EspnLeagueConfig:
    sport_key: str
    sport_path: str
    league_path: str
    display_name: str


TEAM_GAME_SERIES: dict[str, EspnLeagueConfig] = {
    "KXMLBGAME": EspnLeagueConfig("mlb", "baseball", "mlb", "MLB"),
    "KXNFLGAME": EspnLeagueConfig("nfl", "football", "nfl", "NFL"),
    "KXNCAAFGAME": EspnLeagueConfig("ncaaf", "football", "college-football", "NCAAF"),
    "KXNBAGAME": EspnLeagueConfig("nba", "basketball", "nba", "NBA"),
    "KXNCAAMBGAME": EspnLeagueConfig("ncaamb", "basketball", "mens-college-basketball", "NCAAM"),
    "KXNHLGAME": EspnLeagueConfig("nhl", "hockey", "nhl", "NHL"),
    "KXWNBAGAME": EspnLeagueConfig("wnba", "basketball", "wnba", "WNBA"),
}

SCORE_MARKET_SERIES: dict[str, tuple[EspnLeagueConfig, str]] = {
    "KXMLBSPREAD": (EspnLeagueConfig("mlb", "baseball", "mlb", "MLB"), "spread"),
    "KXMLBTOTAL": (EspnLeagueConfig("mlb", "baseball", "mlb", "MLB"), "game_total"),
    "KXMLBTEAMTOTAL": (EspnLeagueConfig("mlb", "baseball", "mlb", "MLB"), "team_total"),
    "KXNFLSPREAD": (EspnLeagueConfig("nfl", "football", "nfl", "NFL"), "spread"),
    "KXNFLTOTAL": (EspnLeagueConfig("nfl", "football", "nfl", "NFL"), "game_total"),
    "KXNFLTEAMTOTAL": (EspnLeagueConfig("nfl", "football", "nfl", "NFL"), "team_total"),
    "KXNCAAFSPREAD": (EspnLeagueConfig("ncaaf", "football", "college-football", "NCAAF"), "spread"),
    "KXNCAAFTOTAL": (EspnLeagueConfig("ncaaf", "football", "college-football", "NCAAF"), "game_total"),
    "KXNCAAFTEAMTOTAL": (EspnLeagueConfig("ncaaf", "football", "college-football", "NCAAF"), "team_total"),
    "KXNBASPREAD": (EspnLeagueConfig("nba", "basketball", "nba", "NBA"), "spread"),
    "KXNBATOTAL": (EspnLeagueConfig("nba", "basketball", "nba", "NBA"), "game_total"),
    "KXNBATEAMTOTAL": (EspnLeagueConfig("nba", "basketball", "nba", "NBA"), "team_total"),
    "KXNHLSPREAD": (EspnLeagueConfig("nhl", "hockey", "nhl", "NHL"), "spread"),
    "KXNHLTOTAL": (EspnLeagueConfig("nhl", "hockey", "nhl", "NHL"), "game_total"),
    "KXNHLTEAMTOTAL": (EspnLeagueConfig("nhl", "hockey", "nhl", "NHL"), "team_total"),
    "KXWNBASPREAD": (EspnLeagueConfig("wnba", "basketball", "wnba", "WNBA"), "spread"),
    "KXWNBATOTAL": (EspnLeagueConfig("wnba", "basketball", "wnba", "WNBA"), "game_total"),
    "KXWNBATEAMTOTAL": (EspnLeagueConfig("wnba", "basketball", "wnba", "WNBA"), "team_total"),
}


@dataclass(frozen=True)
class FinanceIndexConfig:
    symbol: str
    display_name: str


FINANCE_INDEX_SERIES: dict[str, FinanceIndexConfig] = {
    "KXINX": FinanceIndexConfig("^GSPC", "S&P 500"),
    "KXNASDAQ100": FinanceIndexConfig("^NDX", "Nasdaq-100"),
}

WEATHER_HOURLY_STATIONS = {
    "KXTEMPAUSH": "KAUS",
    "KXTEMPCHIH": "KORD",
    "KXTEMPDCH": "KDCA",
    "KXTEMPLAXH": "KLAX",
    "KXTEMPMIAH": "KMIA",
    "KXTEMPNYCH": "KNYC",
}

WEATHER_DAILY_HIGH_STATIONS = {
    "KXHIGHAUS": "AUS",
    "KXHIGHCHI": "MDW",
    "KXHIGHDEN": "DEN",
    "KXHIGHLAX": "LAX",
    "KXHIGHMIA": "MIA",
    "KXHIGHNY": "NYC",
    "KXHIGHPHIL": "PHL",
}


@dataclass(frozen=True)
class KnownOutcomeConfig:
    output_dir: Path
    series_ticker: str = DEFAULT_KNOWN_OUTCOME_SERIES
    timezone: str = "America/New_York"
    capture_date: str | None = None
    include_previous_date: bool = True
    lookback_days: int = 3
    contracts: float = 10.0
    min_contracts: float = 1.0
    require_full_contracts: bool = False
    max_ask: float = 0.99
    annual_yield: float = 0.0325
    fallback_settlement_hours: float = 24.0
    min_net_profit_per_contract: float = 0.001
    orderbook_depth: int = 100
    orderbook_workers: int = 8
    max_markets: int | None = None
    max_market_pages: int | None = None
    include_known_no_after_final: bool = True
    include_known_yes_in_game: bool = True
    include_known_yes_after_final: bool = True
    include_team_game_winners: bool = True
    include_score_markets: bool = True
    include_weather_markets: bool = True
    include_finance_index_markets: bool = True
    espn_timeout: float = 20.0
    espn_max_retries: int = 2
    weather_timeout: float = 20.0
    finance_timeout: float = 20.0
    api_key_id: str | None = None
    private_key_path: str | None = None
    private_key_pem: str | None = None


@dataclass(frozen=True)
class KnownOutcomeResult:
    raw_markets: int
    matched_markets: int
    verified_markets: int
    priced_markets: int
    opportunities: int
    orderbook_errors: int
    no_liquidity: int
    candidates_csv: Path
    opportunities_csv: Path
    trade_count: int
    trade_ledger_csv: Path
    pnl_csv: Path


@dataclass(frozen=True)
class KnownOutcomeEvidence:
    side: str
    source: str
    reason: str
    game_status: str
    current_count: int | None
    final_count: int | None
    timestamp: int


@dataclass(frozen=True)
class TeamGameMarket:
    ticker: str
    event_ticker: str
    title: str
    selected_outcome: str
    series_ticker: str
    sport_key: str
    league_name: str
    occurrence_ts: int | None
    fee_type: str | None
    fee_multiplier: float | None
    raw_market: dict[str, Any]


@dataclass(frozen=True)
class TeamGameTeam:
    id: str
    abbreviation: str
    display_name: str
    short_display_name: str
    name: str
    location: str
    nickname: str
    home_away: str
    score: float | None
    winner: bool


@dataclass(frozen=True)
class TeamGameResult:
    sport_key: str
    league_name: str
    external_event_id: str
    external_event_name: str
    game_date_utc: int | None
    status: str
    completed: bool
    teams: tuple[TeamGameTeam, ...]


@dataclass(frozen=True)
class ScoreMarket:
    ticker: str
    event_ticker: str
    title: str
    series_ticker: str
    sport_key: str
    league_name: str
    market_type: str
    selected_outcome: str
    selected_team_name: str | None
    threshold: float
    occurrence_ts: int | None
    fee_type: str | None
    fee_multiplier: float | None
    raw_market: dict[str, Any]


@dataclass(frozen=True)
class ScalarMarket:
    ticker: str
    event_ticker: str
    title: str
    series_ticker: str
    market_family: str
    sport: str
    league: str
    selected_outcome: str
    stat_type: str
    comparator: str
    threshold: float | None
    lower_bound: float | None
    upper_bound: float | None
    source_key: str
    target_date: date
    target_hour: int | None
    occurrence_ts: int | None
    fee_type: str | None
    fee_multiplier: float | None
    raw_market: dict[str, Any]


@dataclass(frozen=True)
class ScalarObservation:
    actual_value: float
    source: str
    status: str
    timestamp: int
    source_key: str


@dataclass(frozen=True)
class VerifiedKnownMarket:
    ticker: str
    raw_market: dict[str, Any]
    side: str
    fee_type: str | None
    fee_multiplier: float | None
    settlement_ts: int
    base_row: dict[str, Any]


@dataclass(frozen=True)
class ExecutionPlan:
    side: str
    requested_contracts: float
    min_contracts: float
    filled_contracts: float
    sizing_mode: str
    fillable_contracts_at_or_below_max: float
    best_executable_ask: float
    best_executable_ask_size: float
    worst_executable_ask: float
    execution_avg_price: float
    execution_cost: float
    fee_total: float
    price_source: str
    opposing_bid_side: str
    best_opposing_bid_price: float
    worst_opposing_bid_price: float
    consumed_levels: str


@dataclass(frozen=True)
class CarryEconomics:
    gross_profit_per_contract: float
    gross_profit_total: float
    capital_per_contract: float
    capital_total: float
    settlement_seconds: float
    settlement_days: float
    annual_yield: float
    carry_cost_per_contract: float
    carry_cost_total: float
    net_profit_per_contract: float
    net_profit_total: float
    net_return_on_capital: float | None
    annualized_net_return_on_capital: float | None
    breakeven_verifier_accuracy: float


def carry_adjusted_economics(
    avg_price: float,
    contracts: float,
    fee_total: float,
    annual_yield: float,
    settlement_seconds: float,
) -> CarryEconomics:
    if contracts <= 0:
        raise ValueError("contracts must be positive")
    horizon = max(0.0, float(settlement_seconds))
    fee_per_contract = fee_total / contracts
    capital_per_contract = avg_price + fee_per_contract
    capital_total = capital_per_contract * contracts
    gross_profit_per_contract = 1.0 - avg_price - fee_per_contract
    gross_profit_total = gross_profit_per_contract * contracts
    carry_cost_per_contract = capital_per_contract * max(0.0, annual_yield) * horizon / SECONDS_PER_YEAR
    carry_cost_total = carry_cost_per_contract * contracts
    net_profit_per_contract = gross_profit_per_contract - carry_cost_per_contract
    net_profit_total = net_profit_per_contract * contracts
    net_return = net_profit_total / capital_total if capital_total > 0 else None
    annualized = None
    if net_return is not None and horizon > 0:
        annualized = net_return * SECONDS_PER_YEAR / horizon
    breakeven_accuracy = min(1.0, max(0.0, capital_per_contract + carry_cost_per_contract))
    return CarryEconomics(
        gross_profit_per_contract=gross_profit_per_contract,
        gross_profit_total=gross_profit_total,
        capital_per_contract=capital_per_contract,
        capital_total=capital_total,
        settlement_seconds=horizon,
        settlement_days=horizon / (24.0 * 60.0 * 60.0),
        annual_yield=annual_yield,
        carry_cost_per_contract=carry_cost_per_contract,
        carry_cost_total=carry_cost_total,
        net_profit_per_contract=net_profit_per_contract,
        net_profit_total=net_profit_total,
        net_return_on_capital=net_return,
        annualized_net_return_on_capital=annualized,
        breakeven_verifier_accuracy=breakeven_accuracy,
    )


def execution_plan_from_orderbook(
    payload: dict[str, Any],
    timestamp: int,
    side: str,
    contracts: float,
    min_contracts: float,
    require_full_contracts: bool,
    max_ask: float,
    fee_type: str | None,
    fee_multiplier: float | None,
) -> ExecutionPlan | None:
    requested = max(0.0, float(contracts))
    minimum = max(0.0, float(min_contracts))
    if side not in {"yes", "no"} or requested <= 0 or minimum <= 0:
        return None
    minimum = min(minimum, requested)
    _, levels = quote_from_exact_orderbook(timestamp, payload, source="kalshi_live_orderbook")
    ask_side = "yes_ask" if side == "yes" else "no_ask"
    eligible = sorted(
        [level for level in levels if level.side == ask_side and level.price <= max_ask],
        key=lambda item: item.price,
    )
    if not eligible:
        return None
    available = sum(level.quantity for level in eligible)
    if require_full_contracts and available + 1e-9 < requested:
        return None
    if not require_full_contracts and available + 1e-9 < minimum:
        return None

    filled = requested if require_full_contracts else min(requested, available)
    remaining = filled
    consumed: list[tuple[float, float]] = []
    cost = 0.0
    fee_total = 0.0
    for level in eligible:
        if remaining <= 1e-9:
            break
        fill_qty = min(remaining, level.quantity)
        consumed.append((level.price, fill_qty))
        cost += level.price * fill_qty
        fee_total += kalshi_taker_fee(level.price, fill_qty, fee_multiplier, fee_type)
        remaining -= fill_qty
    if remaining > 1e-9 or not consumed:
        return None

    best_ask = consumed[0][0]
    worst_ask = consumed[-1][0]
    opposing_bid_side = "no_bid" if side == "yes" else "yes_bid"
    return ExecutionPlan(
        side=side,
        requested_contracts=requested,
        min_contracts=minimum,
        filled_contracts=filled,
        sizing_mode="require_full_contracts" if require_full_contracts else "take_available_up_to_contracts",
        fillable_contracts_at_or_below_max=available,
        best_executable_ask=best_ask,
        best_executable_ask_size=eligible[0].quantity,
        worst_executable_ask=worst_ask,
        execution_avg_price=cost / filled,
        execution_cost=cost,
        fee_total=fee_total,
        price_source=(
            "kalshi_orderbook_contra_no_bid_executable_yes_ask"
            if side == "yes"
            else "kalshi_orderbook_contra_yes_bid_executable_no_ask"
        ),
        opposing_bid_side=opposing_bid_side,
        best_opposing_bid_price=round(1.0 - best_ask, 4),
        worst_opposing_bid_price=round(1.0 - worst_ask, 4),
        consumed_levels=";".join(f"{price:.4f}x{quantity:.2f}" for price, quantity in consumed),
    )


class EspnCoreClient:
    def __init__(
        self,
        timeout: float = 20.0,
        max_retries: int = 2,
        retry_sleep: float = 0.5,
    ) -> None:
        self.base_url = "https://sports.core.api.espn.com/v2"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "kalshi-known-outcome-scan/1.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self._json_cache: dict[str, dict[str, Any]] = {}

    def get_json(self, url_or_path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if url_or_path.startswith("http://") or url_or_path.startswith("https://"):
            url = url_or_path.replace("http://", "https://", 1)
        else:
            url = f"{self.base_url}{url_or_path}"
        cache_key = url
        if params:
            cache_key = f"{url}?{tuple(sorted((key, str(value)) for key, value in params.items()))}"
        cached = self._json_cache.get(cache_key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params or {}, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"ESPN returned non-object JSON from {response.url}")
                self._json_cache[cache_key] = data
                return data
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
        raise RuntimeError(f"ESPN request failed for {url}: {last_exc}") from last_exc

    def fetch_team_games(self, league: EspnLeagueConfig, game_date: date) -> list[TeamGameResult]:
        games: list[TeamGameResult] = []
        page = 1
        while True:
            payload = self.get_json(
                f"/sports/{league.sport_path}/leagues/{league.league_path}/events",
                {
                    "dates": game_date.strftime("%Y%m%d"),
                    "limit": 300,
                    "page": page,
                    "lang": "en",
                    "region": "us",
                },
            )
            for item in payload.get("items") or []:
                ref = str((item or {}).get("$ref") or "")
                if not ref:
                    continue
                try:
                    event = self.get_json(ref)
                    parsed = self._parse_event(league, event)
                    if parsed is not None:
                        games.append(parsed)
                except Exception as exc:
                    print(f"Skipping ESPN event {ref}: {exc}", flush=True)
            page_count = int(payload.get("pageCount") or 0)
            if page_count <= 0 or page >= page_count:
                break
            page += 1
        return games

    def _parse_event(self, league: EspnLeagueConfig, event: dict[str, Any]) -> TeamGameResult | None:
        competitions = event.get("competitions") or []
        if not competitions:
            return None
        competition = competitions[0]
        status_payload = _resolve_ref_payload(self, competition.get("status"))
        status_type = (status_payload.get("type") or {}) if status_payload else {}
        completed = bool(status_type.get("completed"))
        status = str(status_type.get("description") or status_type.get("name") or "")
        teams: list[TeamGameTeam] = []
        for competitor in competition.get("competitors") or []:
            if not isinstance(competitor, dict):
                continue
            team_payload = _resolve_ref_payload(self, competitor.get("team"))
            score_payload = _resolve_ref_payload(self, competitor.get("score"))
            teams.append(
                TeamGameTeam(
                    id=str(team_payload.get("id") or competitor.get("id") or ""),
                    abbreviation=str(team_payload.get("abbreviation") or ""),
                    display_name=str(team_payload.get("displayName") or ""),
                    short_display_name=str(team_payload.get("shortDisplayName") or ""),
                    name=str(team_payload.get("name") or ""),
                    location=str(team_payload.get("location") or ""),
                    nickname=str(team_payload.get("nickname") or ""),
                    home_away=str(competitor.get("homeAway") or ""),
                    score=parse_float(score_payload.get("value") if score_payload else None),
                    winner=bool(competitor.get("winner") or (score_payload or {}).get("winner")),
                )
            )
        return TeamGameResult(
            sport_key=league.sport_key,
            league_name=league.display_name,
            external_event_id=str(event.get("id") or competition.get("id") or ""),
            external_event_name=str(event.get("name") or event.get("shortName") or ""),
            game_date_utc=unix_seconds(event.get("date") or competition.get("date")),
            status=status,
            completed=completed,
            teams=tuple(teams),
        )


class WeatherKalshiClient:
    def __init__(self, timeout: float = 20.0, max_retries: int = 2, retry_sleep: float = 0.5) -> None:
        self.base_url = "https://weather.com/kalshi/api"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "kalshi-known-outcome-scan/1.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self._json_cache: dict[str, dict[str, Any]] = {}

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        clean_params = {key: value for key, value in (params or {}).items() if value is not None}
        cache_key = url
        if clean_params:
            cache_key = f"{url}?{tuple(sorted((key, str(value)) for key, value in clean_params.items()))}"
        cached = self._json_cache.get(cache_key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=clean_params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Weather.com returned non-object JSON from {response.url}")
                self._json_cache[cache_key] = data
                return data
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
        raise RuntimeError(f"Weather.com request failed for {url}: {last_exc}") from last_exc

    def fetch_daily_high(self, station_key: str, target_date: date) -> ScalarObservation | None:
        station_key = _code(station_key).removeprefix("CLI")
        if not station_key:
            return None
        for path in ("/climate/primary", "/climate/international"):
            payload = self.get_json(path, {"date": target_date.isoformat()})
            for row in payload.get("results") or []:
                if not isinstance(row, dict):
                    continue
                station = row.get("station") if isinstance(row.get("station"), dict) else {}
                data = row.get("data") if isinstance(row.get("data"), dict) else {}
                candidate_keys = {
                    _code(station.get("cliId")),
                    _code(data.get("stationId")),
                    _code(station.get("icao")),
                    _code(row.get("icao")),
                    _code(row.get("id")),
                }
                if station_key not in candidate_keys:
                    continue
                status = str(row.get("status") or data.get("status") or "")
                actual = parse_float(data.get("maxTemp") if data else row.get("maxTemp"))
                if actual is None:
                    return None
                if status.lower() != "official" and not bool(data.get("isOfficial")):
                    return None
                return ScalarObservation(
                    actual_value=actual,
                    source="weather_com_kalshi_climate_api",
                    status=status or "official",
                    timestamp=int(time.time()),
                    source_key=station_key,
                )
        return None

    def fetch_hourly_temperature(
        self,
        station_key: str,
        station_name: str,
        target_date: date,
        target_hour: int,
    ) -> ScalarObservation | None:
        week_start = target_date - timedelta(days=target_date.weekday())
        station_key = _code(station_key)
        station_name_key = normalize_name(station_name)
        for primary_flag in ("true", "false"):
            payload = self.get_json(
                "/metar",
                {
                    "primary": "true" if primary_flag == "true" else None,
                    "international": "true" if primary_flag == "false" else None,
                    "weekStart": week_start.isoformat(),
                },
            )
            for station in payload.get("stations") or []:
                if not isinstance(station, dict):
                    continue
                candidate_keys = {_code(station.get("icaoId")), _code(station.get("id"))}
                station_display = normalize_name(str(station.get("stationName") or ""))
                station_match = station_key and station_key in candidate_keys
                if not station_match and station_name_key:
                    station_match = station_name_key in station_display or station_display in station_name_key
                if not station_match:
                    continue
                for observation in station.get("observations") or []:
                    if not isinstance(observation, dict):
                        continue
                    if str(observation.get("localDate") or "") != target_date.isoformat():
                        continue
                    if int(parse_float(observation.get("localHour")) or -1) != target_hour:
                        continue
                    actual = parse_float(observation.get("tempF"))
                    status = str(observation.get("status") or "")
                    if actual is None or status.lower() != "settled":
                        return None
                    return ScalarObservation(
                        actual_value=actual,
                        source="weather_com_kalshi_metar_api",
                        status=status,
                        timestamp=unix_seconds(observation.get("reportTimeUTC")) or int(time.time()),
                        source_key=_code(observation.get("icaoId")) or station_key,
                    )
        return None


class FinanceIndexClient:
    def __init__(self, timeout: float = 20.0, max_retries: int = 2, retry_sleep: float = 0.5) -> None:
        self.base_url = "https://query1.finance.yahoo.com/v8/finance/chart"
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 kalshi-known-outcome-scan/1.0",
                "Accept": "application/json,text/plain,*/*",
            }
        )
        self._json_cache: dict[str, dict[str, Any]] = {}

    def get_json(self, symbol: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}/{symbol}"
        cache_key = f"{url}?{tuple(sorted((key, str(value)) for key, value in params.items()))}"
        cached = self._json_cache.get(cache_key)
        if cached is not None:
            return cached
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise RuntimeError(f"Yahoo returned non-object JSON from {response.url}")
                self._json_cache[cache_key] = data
                return data
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_sleep * (2**attempt))
                    continue
        raise RuntimeError(f"Yahoo request failed for {url}: {last_exc}") from last_exc

    def fetch_close(self, symbol: str, target_date: date) -> ScalarObservation | None:
        start = int(datetime.combine(target_date - timedelta(days=7), datetime.min.time(), tzinfo=UTC).timestamp())
        end = int(datetime.combine(target_date + timedelta(days=3), datetime.min.time(), tzinfo=UTC).timestamp())
        payload = self.get_json(
            symbol,
            {
                "period1": start,
                "period2": end,
                "interval": "1d",
                "includePrePost": "false",
                "events": "history",
            },
        )
        chart = (payload.get("chart") or {}) if isinstance(payload.get("chart"), dict) else {}
        results = chart.get("result") or []
        if not results or not isinstance(results[0], dict):
            return None
        result = results[0]
        meta = (result.get("meta") or {}) if isinstance(result.get("meta"), dict) else {}
        tz_name = str(meta.get("exchangeTimezoneName") or "America/New_York")
        try:
            exchange_tz = ZoneInfo(tz_name)
        except Exception:
            exchange_tz = ZoneInfo("America/New_York")
        timestamps = result.get("timestamp") or []
        quotes = (((result.get("indicators") or {}).get("quote") or [{}])[0] or {})
        closes = quotes.get("close") or []
        for raw_ts, raw_close in zip(timestamps, closes):
            close_value = parse_float(raw_close)
            if close_value is None:
                continue
            local_date = datetime.fromtimestamp(int(raw_ts), tz=exchange_tz).date()
            if local_date != target_date:
                continue
            return ScalarObservation(
                actual_value=close_value,
                source="yahoo_finance_chart_api_close",
                status="final_chart_close",
                timestamp=int(raw_ts),
                source_key=symbol,
            )
        return None


def _resolve_ref_payload(client: EspnCoreClient, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    ref = str(value.get("$ref") or "")
    if ref:
        return client.get_json(ref)
    return value


def _series_tickers(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _market_series_ticker_any(market: dict[str, Any]) -> str | None:
    series = market_series_ticker(market)
    if series:
        return series
    raw = str(market.get("series_ticker") or "").strip().upper()
    if raw:
        return raw
    ticker = str(market.get("ticker") or "")
    if "-" not in ticker:
        return ticker.upper() or None
    return ticker.split("-", 1)[0].upper()


def _local_date(config: KnownOutcomeConfig) -> date:
    if config.capture_date:
        return datetime.strptime(config.capture_date, "%Y-%m-%d").date()
    return datetime.now(ZoneInfo(config.timezone)).date()


def _date_range(config: KnownOutcomeConfig) -> tuple[str, str]:
    end_date = _local_date(config)
    days_back = max(1 if config.include_previous_date else 0, int(config.lookback_days or 0))
    start_date = end_date - timedelta(days=max(0, days_back))
    return start_date.isoformat(), end_date.isoformat()


def _feed_status(context: GameContext) -> str:
    status = ((context.feed.get("gameData") or {}).get("status") or {}).get("detailedState")
    return str(status or context.game.status or "")


def _status_is_final(status: str) -> bool:
    normalized = status.lower()
    return any(token in normalized for token in ("final", "game over", "completed early"))


def _status_is_pregame_or_scheduled(status: str) -> bool:
    normalized = status.lower()
    return any(token in normalized for token in ("scheduled", "pre-game", "warmup", "preview", "delayed start"))


def _code(value: str | None) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").upper())


def _selected_team_code(ticker: str) -> str:
    suffix = ticker.rsplit("-", 1)[-1] if "-" in ticker else ""
    return _code(re.sub(r"\d+(?:\.\d+)?$", "", suffix))


def _event_matchup_code(event_ticker: str) -> str:
    match = re.search(r"-\d{2}[A-Z]{3}\d{2}(?:\d{4})?(?P<code>[A-Z0-9]+)$", event_ticker.upper())
    return _code(match.group("code")) if match else ""


def _team_token_matches(segment: str, team: "TeamGameTeam") -> bool:
    segment = _code(segment)
    if not segment:
        return False
    codes = _team_codes(team)
    if segment in codes:
        return True
    keys = {key.replace(" ", "").upper() for key in _team_name_keys(team) if key}
    if segment in keys:
        return True
    if len(segment) >= 3 and any(key.startswith(segment) for key in keys if len(key) >= len(segment)):
        return True
    if segment == "AZ" and "ARI" in codes:
        return True
    if segment == "CHAR" and any(key.startswith("CHARLOTTE") for key in keys):
        return True
    return False


def _game_matches_event_ticker(event_ticker: str, game: "TeamGameResult") -> bool:
    matchup_code = _event_matchup_code(event_ticker)
    teams = list(game.teams)
    if not matchup_code or len(teams) != 2:
        return True
    away = next((team for team in teams if team.home_away.lower() == "away"), teams[0])
    home = next((team for team in teams if team.home_away.lower() == "home"), teams[1])
    for split in range(2, len(matchup_code) - 1):
        first = matchup_code[:split]
        second = matchup_code[split:]
        if _team_token_matches(first, away) and _team_token_matches(second, home):
            return True
        if _team_token_matches(first, home) and _team_token_matches(second, away):
            return True
    return False


def _team_name_keys(team: TeamGameTeam) -> set[str]:
    values = {
        team.display_name,
        team.short_display_name,
        team.name,
        team.location,
        team.nickname,
        f"{team.location} {team.name}".strip(),
        team.abbreviation,
    }
    keys = {normalize_name(value) for value in values if value}
    compact = {key.replace(" ", "") for key in keys if key}
    return keys | compact


def _team_codes(team: TeamGameTeam) -> set[str]:
    return {_code(value) for value in (team.abbreviation, team.id) if value}


def _team_match_strength(team: TeamGameTeam, selected_outcome: str, selected_code: str) -> int:
    if selected_code and selected_code in _team_codes(team):
        return 100
    selected_key = normalize_name(selected_outcome)
    if not selected_key:
        return 0
    keys = _team_name_keys(team)
    if selected_key in keys or selected_key.replace(" ", "") in keys:
        return 80
    if any(selected_key in key or key in selected_key for key in keys if len(key) >= 5):
        return 60
    return 0


def _team_mentioned_in_market_text(team: TeamGameTeam, market_text: str) -> bool:
    text_key = normalize_name(market_text)
    text_compact = text_key.replace(" ", "")
    for key in _team_name_keys(team):
        if len(key) < 3:
            continue
        if key in text_key or key in text_compact:
            return True
    for code in _team_codes(team):
        if code and re.search(rf"\b{re.escape(code)}\b", market_text.upper()):
            return True
    return False


def _winner_team(game: TeamGameResult) -> TeamGameTeam | None:
    winners = [team for team in game.teams if team.winner]
    if len(winners) == 1:
        return winners[0]
    return None


def _score_for_side(game: TeamGameResult, home_away: str) -> float | None:
    team = next((item for item in game.teams if item.home_away.lower() == home_away), None)
    return team.score if team is not None else None


def _team_game_market_from_raw(
    market: dict[str, Any],
    fee_info: dict[str, tuple[str | None, float | None]],
) -> TeamGameMarket | None:
    series = _market_series_ticker_any(market)
    if series not in TEAM_GAME_SERIES:
        return None
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    selected = str(market.get("yes_sub_title") or "").strip()
    if not selected:
        selected = re.sub(r"\s+wins\s*$", "", str(market.get("title") or ""), flags=re.IGNORECASE).strip()
    if not ticker or not event_ticker or not selected:
        return None
    league = TEAM_GAME_SERIES[series]
    fee_type, fee_multiplier = fee_info.get(series, ("quadratic", 0.5))
    return TeamGameMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=str(market.get("title") or ""),
        selected_outcome=selected,
        series_ticker=series,
        sport_key=league.sport_key,
        league_name=league.display_name,
        occurrence_ts=_market_time(market, ("occurrence_datetime", "expected_expiration_time")),
        fee_type=market.get("fee_type") or fee_type,
        fee_multiplier=parse_float(market.get("fee_multiplier")) if market.get("fee_multiplier") is not None else fee_multiplier,
        raw_market=market,
    )


def _is_supported_team_game_market(market: dict[str, Any]) -> bool:
    return _market_series_ticker_any(market) in TEAM_GAME_SERIES


def _is_supported_score_market(market: dict[str, Any]) -> bool:
    return _market_series_ticker_any(market) in SCORE_MARKET_SERIES


def _is_supported_weather_market(market: dict[str, Any]) -> bool:
    series = _market_series_ticker_any(market) or ""
    return series in WEATHER_HOURLY_STATIONS or series in WEATHER_DAILY_HIGH_STATIONS


def _is_supported_finance_index_market(market: dict[str, Any]) -> bool:
    return _market_series_ticker_any(market) in FINANCE_INDEX_SERIES


def _fee_for_market(
    market: dict[str, Any],
    series: str,
    fee_info: dict[str, tuple[str | None, float | None]],
) -> tuple[str | None, float | None]:
    fee_type, fee_multiplier = fee_info.get(series, ("quadratic", 0.5))
    raw_multiplier = parse_float(market.get("fee_multiplier"))
    return market.get("fee_type") or fee_type, raw_multiplier if raw_multiplier is not None else fee_multiplier


def _parse_ticker_date(ticker: str) -> date | None:
    match = re.search(r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?:H?\d{0,4})?(?:-|$)", ticker)
    if not match:
        return None
    month = MONTHS.get(match.group("mon"))
    if month is None:
        return None
    try:
        return date(2000 + int(match.group("yy")), month, int(match.group("dd")))
    except ValueError:
        return None


def _parse_ticker_date_hour(ticker: str) -> tuple[date, int] | None:
    match = re.search(r"-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hh>\d{2})-", ticker)
    if not match:
        return None
    month = MONTHS.get(match.group("mon"))
    if month is None:
        return None
    hour = int(match.group("hh"))
    if hour > 23:
        return None
    try:
        return date(2000 + int(match.group("yy")), month, int(match.group("dd"))), hour
    except ValueError:
        return None


def _clean_number(raw: str | None) -> float | None:
    if raw is None:
        return None
    return parse_float(str(raw).replace(",", "").replace("$", ""))


def _parse_condition(text: str) -> tuple[str, float | None, float | None, float | None] | None:
    compact = text.replace(",", "")
    between = re.search(
        r"\bbetween\s+\$?(?P<lower>\d+(?:\.\d+)?)\s*(?:and|to|-)\s*\$?(?P<upper>\d+(?:\.\d+)?)",
        compact,
        re.IGNORECASE,
    )
    if between:
        lower = _clean_number(between.group("lower"))
        upper = _clean_number(between.group("upper"))
        if lower is not None and upper is not None:
            return "between", None, min(lower, upper), max(lower, upper)
    above = re.search(r"\b(?:above|greater than|more than)\s+\$?(?P<threshold>\d+(?:\.\d+)?)", compact, re.IGNORECASE)
    if above:
        threshold = _clean_number(above.group("threshold"))
        if threshold is not None:
            return "above", threshold, None, None
    below = re.search(r"\b(?:below|less than)\s+\$?(?P<threshold>\d+(?:\.\d+)?)", compact, re.IGNORECASE)
    if below:
        threshold = _clean_number(below.group("threshold"))
        if threshold is not None:
            return "below", threshold, None, None
    return None


def _weather_location_from_text(text: str) -> str:
    recorded = re.search(r"\brecorded at\s+(?P<location>.+?)\s+(?:\([A-Z0-9]+\)\s+)?for\b", text, re.IGNORECASE)
    if recorded:
        return recorded.group("location").strip().rstrip(",")
    title = re.search(r"\btemp(?:erature)? in\s+(?P<location>.+?)\s+be\b", text, re.IGNORECASE)
    if title:
        return title.group("location").strip().rstrip(",")
    return ""


def _weather_source_key(text: str, series: str) -> str:
    coordinate = re.search(r"\bcoordinates\s+(?P<station>[A-Z0-9]+)\b", text, re.IGNORECASE)
    if coordinate:
        return _code(coordinate.group("station"))
    climate_station = re.search(r"\((?P<station>CLI[A-Z0-9]+)\)", text, re.IGNORECASE)
    if climate_station:
        return _code(climate_station.group("station")).removeprefix("CLI")
    if series in WEATHER_HOURLY_STATIONS:
        return WEATHER_HOURLY_STATIONS[series]
    return WEATHER_DAILY_HIGH_STATIONS.get(series, "")


def _weather_market_from_raw(
    market: dict[str, Any],
    fee_info: dict[str, tuple[str | None, float | None]],
) -> ScalarMarket | None:
    series = _market_series_ticker_any(market) or ""
    if not _is_supported_weather_market(market):
        return None
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    text = _market_text(market)
    condition = _parse_condition(text)
    if condition is None or not ticker:
        return None
    comparator, threshold, lower, upper = condition
    fee_type, fee_multiplier = _fee_for_market(market, series, fee_info)
    if series in WEATHER_HOURLY_STATIONS:
        parsed_hour = _parse_ticker_date_hour(ticker)
        if parsed_hour is None:
            return None
        target_date, target_hour = parsed_hour
        family = "hourly_temperature"
        stat_type = "temperature_f"
        source_key = _weather_source_key(text, series)
    else:
        target_date = _parse_ticker_date(ticker)
        if target_date is None:
            return None
        target_hour = None
        family = "daily_high_temperature"
        stat_type = "max_temperature_f"
        source_key = _weather_source_key(text, series)
    if not source_key:
        return None
    selected = str(market.get("yes_sub_title") or "").strip() or _weather_location_from_text(text) or "temperature"
    return ScalarMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=str(market.get("title") or ""),
        series_ticker=series,
        market_family=family,
        sport="weather",
        league="Weather",
        selected_outcome=selected,
        stat_type=stat_type,
        comparator=comparator,
        threshold=threshold,
        lower_bound=lower,
        upper_bound=upper,
        source_key=source_key,
        target_date=target_date,
        target_hour=target_hour,
        occurrence_ts=_market_time(market, ("occurrence_datetime", "expected_expiration_time", "close_time")),
        fee_type=fee_type,
        fee_multiplier=fee_multiplier,
        raw_market=market,
    )


def _finance_index_market_from_raw(
    market: dict[str, Any],
    fee_info: dict[str, tuple[str | None, float | None]],
) -> ScalarMarket | None:
    series = _market_series_ticker_any(market) or ""
    index_config = FINANCE_INDEX_SERIES.get(series)
    if index_config is None:
        return None
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    text = _market_text(market)
    condition = _parse_condition(text)
    if condition is None or not ticker:
        return None
    comparator, threshold, lower, upper = condition
    occurrence_ts = _market_time(market, ("occurrence_datetime", "close_time", "expected_expiration_time"))
    target_date = (
        datetime.fromtimestamp(occurrence_ts, tz=ZoneInfo("America/New_York")).date()
        if occurrence_ts is not None
        else _parse_ticker_date(ticker)
    )
    if target_date is None:
        return None
    fee_type, fee_multiplier = _fee_for_market(market, series, fee_info)
    return ScalarMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=str(market.get("title") or ""),
        series_ticker=series,
        market_family="finance_index_close",
        sport="finance",
        league=index_config.display_name,
        selected_outcome=str(market.get("yes_sub_title") or "").strip() or index_config.display_name,
        stat_type="close_value",
        comparator=comparator,
        threshold=threshold,
        lower_bound=lower,
        upper_bound=upper,
        source_key=index_config.symbol,
        target_date=target_date,
        target_hour=None,
        occurrence_ts=occurrence_ts,
        fee_type=fee_type,
        fee_multiplier=fee_multiplier,
        raw_market=market,
    )


def _parse_score_market_threshold(market_type: str, market: dict[str, Any]) -> tuple[str | None, float] | None:
    texts = [
        str(market.get("yes_sub_title") or ""),
        str(market.get("title") or ""),
        re.sub(r"^\s*if\s+", "", str(market.get("rules_primary") or ""), flags=re.IGNORECASE),
    ]
    if market_type == "game_total":
        for text in texts:
            match = re.search(r"\bover\s+(?P<threshold>\d+(?:\.\d+)?)\s+(?:points|runs|goals)\b", text, re.IGNORECASE)
            if match:
                threshold = _clean_number(match.group("threshold"))
                if threshold is not None:
                    return None, threshold
    if market_type == "team_total":
        for text in texts:
            match = re.search(
                r"^\s*(?P<team>.+?)\s+(?:scores\s+)?over\s+(?P<threshold>\d+(?:\.\d+)?)\s+"
                r"(?:points|runs|goals)\b",
                text,
                re.IGNORECASE,
            )
            if match:
                threshold = _clean_number(match.group("threshold"))
                if threshold is not None:
                    return match.group("team").strip(), threshold
    if market_type == "spread":
        for text in texts:
            match = re.search(
                r"^\s*(?P<team>.+?)\s+wins.*?\bby\s+(?:over|more than)\s+"
                r"(?P<threshold>\d+(?:\.\d+)?)\s+(?:points|runs|goals)\b",
                text,
                re.IGNORECASE,
            )
            if match:
                threshold = _clean_number(match.group("threshold"))
                if threshold is not None:
                    return match.group("team").strip(), threshold
    return None


def _score_market_from_raw(
    market: dict[str, Any],
    fee_info: dict[str, tuple[str | None, float | None]],
) -> ScoreMarket | None:
    series = _market_series_ticker_any(market) or ""
    score_config = SCORE_MARKET_SERIES.get(series)
    if score_config is None:
        return None
    league, market_type = score_config
    parsed = _parse_score_market_threshold(market_type, market)
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    if parsed is None or not ticker or not event_ticker:
        return None
    selected_team_name, threshold = parsed
    fee_type, fee_multiplier = _fee_for_market(market, series, fee_info)
    return ScoreMarket(
        ticker=ticker,
        event_ticker=event_ticker,
        title=str(market.get("title") or ""),
        series_ticker=series,
        sport_key=league.sport_key,
        league_name=league.display_name,
        market_type=market_type,
        selected_outcome=str(market.get("yes_sub_title") or "").strip() or str(market.get("title") or ""),
        selected_team_name=selected_team_name,
        threshold=threshold,
        occurrence_ts=_market_time(market, ("occurrence_datetime", "expected_expiration_time", "close_time")),
        fee_type=fee_type,
        fee_multiplier=fee_multiplier,
        raw_market=market,
    )


def _load_game_contexts(mlb: MlbClient, start_date: str, end_date: str) -> dict[int, GameContext]:
    schedule = mlb.fetch_schedule(start_date, end_date)
    contexts: dict[int, GameContext] = {}
    for game in schedule:
        status = game.status.lower()
        if any(token in status for token in ("postponed", "cancelled", "canceled", "suspended")):
            continue
        try:
            contexts[game.game_pk] = GameContext(game, mlb.fetch_game_feed(game.game_pk))
        except Exception as exc:
            print(f"Skipping MLB feed gamePk={game.game_pk}: {exc}", flush=True)
    return contexts


def _discover_markets(
    kalshi: KalshiClient,
    series_ticker: str,
    max_pages: int | None,
    max_markets: int | None,
) -> list[dict[str, Any]]:
    markets: list[dict[str, Any]] = []
    for series in _series_tickers(series_ticker):
        try:
            markets.extend(kalshi.fetch_markets(series_ticker=series, status="open", max_pages=max_pages))
        except KalshiApiError as exc:
            print(f"Skipping known-outcome series {series}; market fetch failed: {exc}", flush=True)
            if _kalshi_connection_issue(exc):
                print("Skipping remaining known-outcome series after connection-level failure.", flush=True)
                break
    filtered = [
        market
        for market in markets
        if (
            is_supported_player_prop_market(market)
            or _is_supported_team_game_market(market)
            or _is_supported_score_market(market)
            or _is_supported_weather_market(market)
            or _is_supported_finance_index_market(market)
        )
    ]
    if max_markets is not None:
        filtered = filtered[:max_markets]
    return filtered


def _kalshi_connection_issue(exc: Exception) -> bool:
    message = str(exc).lower()
    return "ssl" in message or "handshake" in message or "429" in message or "too many requests" in message


def _fee_info_by_series(kalshi: KalshiClient, series_ticker: str) -> dict[str, tuple[str | None, float | None]]:
    out: dict[str, tuple[str | None, float | None]] = {}
    series_list = _series_tickers(series_ticker)
    for index, series in enumerate(series_list):
        try:
            fee_info = kalshi.get_series_fee_info(series)
            out[series] = (fee_info.fee_type, fee_info.fee_multiplier)
        except Exception as exc:
            print(f"Using default fee settings for {series}; fee metadata fetch failed: {exc}", flush=True)
            out[series] = ("quadratic", 0.5)
            if _kalshi_connection_issue(exc):
                for remaining in series_list[index + 1 :]:
                    out[remaining] = ("quadratic", 0.5)
                print("Skipping remaining fee metadata fetches after connection-level failure.", flush=True)
                break
    return out


def _match_markets(
    markets: list[dict[str, Any]],
    contexts: dict[int, GameContext],
    official_date_min: str,
    official_date_max: str,
    fee_info: dict[str, tuple[str | None, float | None]],
) -> list[MatchedMarket]:
    primary_fee_info = next(iter(fee_info.values()), ("quadratic", 0.5))
    matched: list[MatchedMarket] = []
    for market in markets:
        series = market_series_ticker(market)
        fee_type, fee_multiplier = fee_info.get(series or "", primary_fee_info)
        match = match_open_market_to_game_and_player(
            market,
            contexts,
            official_date_min,
            official_date_max,
            fee_type,
            fee_multiplier,
        )
        if match is not None:
            matched.append(match)
    return matched


def _stat_count(context: GameContext, player_id: int, stat_type: str, timestamp: int) -> int:
    if stat_type == "home_runs":
        return context.player_home_runs_at(player_id, timestamp)
    hits, _ = context.player_stats_at(player_id, timestamp)
    return hits


def _final_stat_count(context: GameContext, player_id: int, stat_type: str) -> int:
    if stat_type == "home_runs":
        return context.final_home_runs.get(player_id, 0)
    return context.final_hits.get(player_id, 0)


def _known_mlb_outcome(
    market: MatchedMarket,
    context: GameContext,
    timestamp: int,
    config: KnownOutcomeConfig,
) -> KnownOutcomeEvidence | None:
    status = _feed_status(context)
    if _status_is_pregame_or_scheduled(status):
        return None
    current_count = _stat_count(context, market.mlb_player_id, market.stat_type, timestamp)
    if _status_is_final(status):
        final_count = _final_stat_count(context, market.mlb_player_id, market.stat_type)
        side = "yes" if final_count >= market.threshold else "no"
        if side == "yes" and not config.include_known_yes_after_final:
            return None
        if side == "no" and not config.include_known_no_after_final:
            return None
        comparator = ">=" if side == "yes" else "<"
        return KnownOutcomeEvidence(
            side=side,
            source="mlb_stats_api_final_boxscore",
            reason=f"official_final_{market.stat_type}={final_count}_{comparator}_{market.threshold}",
            game_status=status,
            current_count=current_count,
            final_count=final_count,
            timestamp=timestamp,
        )
    if config.include_known_yes_in_game and current_count >= market.threshold:
        return KnownOutcomeEvidence(
            side="yes",
            source="mlb_stats_api_live_feed",
            reason=f"live_{market.stat_type}={current_count}_>=_{market.threshold}",
            game_status=status,
            current_count=current_count,
            final_count=None,
            timestamp=timestamp,
        )
    return None


def _market_text(market: dict[str, Any]) -> str:
    return " ".join(
        str(market.get(key) or "")
        for key in (
            "ticker",
            "event_ticker",
            "title",
            "yes_sub_title",
            "rules_primary",
            "rules_secondary",
        )
    )


def _match_team_game_result(market: TeamGameMarket, games: list[TeamGameResult]) -> tuple[TeamGameResult, TeamGameTeam] | None:
    selected_code = _selected_team_code(market.ticker)
    text = _market_text(market.raw_market)
    candidates: list[tuple[float, TeamGameResult, TeamGameTeam]] = []
    for game in games:
        if not game.completed or len(game.teams) != 2 or _winner_team(game) is None:
            continue
        if not _game_matches_event_ticker(market.event_ticker, game):
            continue
        best_team: TeamGameTeam | None = None
        best_strength = 0
        for team in game.teams:
            strength = _team_match_strength(team, market.selected_outcome, selected_code)
            if strength > best_strength:
                best_strength = strength
                best_team = team
        if best_team is None or best_strength <= 0:
            continue
        mentioned = sum(1 for team in game.teams if _team_mentioned_in_market_text(team, text))
        time_delta_hours = 999.0
        if market.occurrence_ts is not None and game.game_date_utc is not None:
            time_delta_hours = abs(market.occurrence_ts - game.game_date_utc) / 3600.0
        conservative_match = mentioned >= 2 or (best_strength >= 100 and time_delta_hours <= 12.0)
        if not conservative_match:
            continue
        score = float(best_strength + mentioned * 10) - min(time_delta_hours, 96.0) / 10.0
        candidates.append((score, game, best_team))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-9:
        return None
    return candidates[0][1], candidates[0][2]


def _team_game_evidence(
    market: TeamGameMarket,
    game: TeamGameResult,
    selected_team: TeamGameTeam,
    timestamp: int,
    config: KnownOutcomeConfig,
) -> KnownOutcomeEvidence | None:
    winner = _winner_team(game)
    if winner is None:
        return None
    side = "yes" if selected_team.id == winner.id else "no"
    if side == "yes" and not config.include_known_yes_after_final:
        return None
    if side == "no" and not config.include_known_no_after_final:
        return None
    return KnownOutcomeEvidence(
        side=side,
        source=f"espn_core_final_scoreboard_{game.sport_key}",
        reason=f"final_winner={winner.display_name or winner.abbreviation};selected={selected_team.display_name or market.selected_outcome}",
        game_status=game.status,
        current_count=None,
        final_count=None,
        timestamp=timestamp,
    )


def _condition_is_yes(market: ScalarMarket, actual: float) -> bool:
    if market.comparator == "above" and market.threshold is not None:
        return actual > market.threshold
    if market.comparator == "below" and market.threshold is not None:
        return actual < market.threshold
    if market.comparator == "between" and market.lower_bound is not None and market.upper_bound is not None:
        return market.lower_bound <= actual <= market.upper_bound
    return False


def _scalar_condition_text(market: ScalarMarket) -> str:
    if market.comparator == "between":
        return f"{market.lower_bound:g}_to_{market.upper_bound:g}"
    return f"{market.comparator}_{market.threshold:g}" if market.threshold is not None else market.comparator


def _scalar_evidence(
    market: ScalarMarket,
    observation: ScalarObservation,
    timestamp: int,
    config: KnownOutcomeConfig,
) -> KnownOutcomeEvidence | None:
    side = "yes" if _condition_is_yes(market, observation.actual_value) else "no"
    if side == "yes" and not config.include_known_yes_after_final:
        return None
    if side == "no" and not config.include_known_no_after_final:
        return None
    return KnownOutcomeEvidence(
        side=side,
        source=observation.source,
        reason=f"actual_{market.stat_type}={observation.actual_value:g};condition={_scalar_condition_text(market)}",
        game_status=observation.status,
        current_count=None,
        final_count=None,
        timestamp=timestamp,
    )


def _match_score_game_result(
    market: ScoreMarket,
    games: list[TeamGameResult],
) -> tuple[TeamGameResult, TeamGameTeam | None] | None:
    selected_code = _selected_team_code(market.ticker)
    has_matchup_code = bool(_event_matchup_code(market.event_ticker))
    text = _market_text(market.raw_market)
    candidates: list[tuple[float, TeamGameResult, TeamGameTeam | None]] = []
    for game in games:
        if not game.completed or len(game.teams) != 2 or any(team.score is None for team in game.teams):
            continue
        if not _game_matches_event_ticker(market.event_ticker, game):
            continue
        mentioned = sum(1 for team in game.teams if _team_mentioned_in_market_text(team, text))
        time_delta_hours = 999.0
        if market.occurrence_ts is not None and game.game_date_utc is not None:
            time_delta_hours = abs(market.occurrence_ts - game.game_date_utc) / 3600.0
        if market.market_type == "game_total":
            if mentioned < 2 and not has_matchup_code:
                continue
            score = float(max(mentioned, 2) * 10) - min(time_delta_hours, 96.0) / 10.0
            candidates.append((score, game, None))
            continue

        best_team: TeamGameTeam | None = None
        best_strength = 0
        selected_name = market.selected_team_name or market.selected_outcome
        for team in game.teams:
            strength = _team_match_strength(team, selected_name, selected_code)
            if strength > best_strength:
                best_strength = strength
                best_team = team
        if best_team is None or best_strength <= 0:
            continue
        conservative_match = mentioned >= 2 or (best_strength >= 80 and time_delta_hours <= 12.0)
        if not conservative_match:
            continue
        score = float(best_strength + mentioned * 10) - min(time_delta_hours, 96.0) / 10.0
        candidates.append((score, game, best_team))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and abs(candidates[0][0] - candidates[1][0]) < 1e-9:
        return None
    return candidates[0][1], candidates[0][2]


def _score_market_evidence(
    market: ScoreMarket,
    game: TeamGameResult,
    selected_team: TeamGameTeam | None,
    timestamp: int,
    config: KnownOutcomeConfig,
) -> KnownOutcomeEvidence | None:
    teams = list(game.teams)
    if len(teams) != 2 or any(team.score is None for team in teams):
        return None
    if market.market_type == "game_total":
        actual = float(teams[0].score or 0.0) + float(teams[1].score or 0.0)
        side = "yes" if actual > market.threshold else "no"
        reason = f"final_total={actual:g}_>_{market.threshold:g}"
    elif market.market_type == "team_total":
        if selected_team is None or selected_team.score is None:
            return None
        actual = float(selected_team.score)
        side = "yes" if actual > market.threshold else "no"
        reason = f"{selected_team.display_name or selected_team.abbreviation}_final_score={actual:g}_>_{market.threshold:g}"
    elif market.market_type == "spread":
        if selected_team is None or selected_team.score is None:
            return None
        opponent = next((team for team in teams if team.id != selected_team.id), None)
        if opponent is None or opponent.score is None:
            return None
        margin = float(selected_team.score) - float(opponent.score)
        side = "yes" if margin > market.threshold else "no"
        reason = f"{selected_team.display_name or selected_team.abbreviation}_final_margin={margin:g}_>_{market.threshold:g}"
    else:
        return None
    if side == "yes" and not config.include_known_yes_after_final:
        return None
    if side == "no" and not config.include_known_no_after_final:
        return None
    return KnownOutcomeEvidence(
        side=side,
        source=f"espn_core_final_scoreboard_{game.sport_key}",
        reason=reason,
        game_status=game.status,
        current_count=None,
        final_count=None,
        timestamp=timestamp,
    )


_ORDERBOOK_CLIENT_LOCAL = threading.local()


def _thread_orderbook_client(config: KnownOutcomeConfig) -> KalshiClient:
    client = getattr(_ORDERBOOK_CLIENT_LOCAL, "client", None)
    if client is None:
        client = KalshiClient(
            api_key_id=config.api_key_id,
            private_key_path=config.private_key_path,
            private_key_pem=config.private_key_pem,
        )
        _ORDERBOOK_CLIENT_LOCAL.client = client
    return client


def _fetch_orderbooks(
    tickers: list[str],
    config: KnownOutcomeConfig,
) -> dict[str, tuple[dict[str, Any] | None, str | None]]:
    if not tickers:
        return {}
    workers = max(1, min(int(config.orderbook_workers or 1), len(tickers)))
    if workers == 1:
        out: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
        client = _thread_orderbook_client(config)
        for ticker in tickers:
            try:
                out[ticker] = (client.get_current_orderbook(ticker, depth=config.orderbook_depth), None)
            except KalshiApiError as exc:
                out[ticker] = (None, str(exc))
        return out

    def fetch(ticker: str) -> tuple[str, dict[str, Any] | None, str | None]:
        client = _thread_orderbook_client(config)
        try:
            return ticker, client.get_current_orderbook(ticker, depth=config.orderbook_depth), None
        except KalshiApiError as exc:
            return ticker, None, str(exc)

    out: dict[str, tuple[dict[str, Any] | None, str | None]] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, ticker) for ticker in tickers]
        for future in as_completed(futures):
            ticker, payload, error = future.result()
            out[ticker] = (payload, error)
    return out


def _market_time(market: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        raw = market.get(key)
        if raw is None or raw == "":
            continue
        if isinstance(raw, (int, float)):
            return unix_seconds(raw)
        parsed = unix_seconds(str(raw))
        if parsed is not None:
            return parsed
    return None


def _settlement_ts(market: MatchedMarket, now_ts: int, fallback_hours: float) -> tuple[int, str]:
    ts = _market_time(
        market.raw_market,
        ("settlement_ts", "close_time", "expiration_time", "expected_expiration_time"),
    )
    if ts is not None and ts > now_ts:
        return ts, "kalshi_market_time"
    fallback = now_ts + int(max(0.0, fallback_hours) * 60.0 * 60.0)
    return fallback, "fallback_settlement_hours"


def _base_row(
    market: MatchedMarket,
    context: GameContext,
    evidence: KnownOutcomeEvidence,
    settlement_ts: int,
    settlement_source: str,
) -> dict[str, Any]:
    return {
        "scan_timestamp_utc": iso_utc(evidence.timestamp),
        "kalshi_market_ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "market_title": market.title,
        "series_ticker": _market_series_ticker_any(market.raw_market),
        "market_family": "player_prop",
        "sport": "mlb",
        "league": "MLB",
        "external_event_id": market.game_pk,
        "external_event_name": f"{context.game.away_team_name} at {context.game.home_team_name}",
        "selected_outcome": market.player_name,
        "winning_outcome": evidence.side,
        "game_id": market.game_pk,
        "official_date": context.game.official_date,
        "away_team": context.game.away_team_name,
        "home_team": context.game.home_team_name,
        "away_score": None,
        "home_score": None,
        "player_id": market.mlb_player_id,
        "player_name": market.player_name,
        "stat_type": market.stat_type,
        "threshold": market.threshold,
        "known_side": evidence.side,
        "verification_source": evidence.source,
        "verification_reason": evidence.reason,
        "game_status": evidence.game_status,
        "current_count": evidence.current_count,
        "final_count": evidence.final_count,
        "settlement_time_utc": iso_utc(settlement_ts),
        "settlement_time_source": settlement_source,
    }


def _base_team_game_row(
    market: TeamGameMarket,
    game: TeamGameResult,
    selected_team: TeamGameTeam,
    evidence: KnownOutcomeEvidence,
    settlement_ts: int,
    settlement_source: str,
) -> dict[str, Any]:
    home_team = next((team for team in game.teams if team.home_away.lower() == "home"), None)
    away_team = next((team for team in game.teams if team.home_away.lower() == "away"), None)
    winner = _winner_team(game)
    official_date = (
        datetime.fromtimestamp(game.game_date_utc, tz=UTC).date().isoformat()
        if game.game_date_utc is not None
        else ""
    )
    return {
        "scan_timestamp_utc": iso_utc(evidence.timestamp),
        "kalshi_market_ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "market_title": market.title,
        "series_ticker": market.series_ticker,
        "market_family": "team_game_winner",
        "sport": market.sport_key,
        "league": market.league_name,
        "external_event_id": game.external_event_id,
        "external_event_name": game.external_event_name,
        "selected_outcome": selected_team.display_name or market.selected_outcome,
        "winning_outcome": winner.display_name if winner is not None else "",
        "game_id": game.external_event_id,
        "official_date": official_date,
        "away_team": away_team.display_name if away_team is not None else "",
        "home_team": home_team.display_name if home_team is not None else "",
        "away_score": away_team.score if away_team is not None else None,
        "home_score": home_team.score if home_team is not None else None,
        "player_id": None,
        "player_name": None,
        "stat_type": "game_winner",
        "threshold": None,
        "known_side": evidence.side,
        "verification_source": evidence.source,
        "verification_reason": evidence.reason,
        "game_status": evidence.game_status,
        "current_count": None,
        "final_count": None,
        "settlement_time_utc": iso_utc(settlement_ts),
        "settlement_time_source": settlement_source,
    }


def _base_score_market_row(
    market: ScoreMarket,
    game: TeamGameResult,
    selected_team: TeamGameTeam | None,
    evidence: KnownOutcomeEvidence,
    settlement_ts: int,
    settlement_source: str,
) -> dict[str, Any]:
    home_team = next((team for team in game.teams if team.home_away.lower() == "home"), None)
    away_team = next((team for team in game.teams if team.home_away.lower() == "away"), None)
    official_date = (
        datetime.fromtimestamp(game.game_date_utc, tz=UTC).date().isoformat()
        if game.game_date_utc is not None
        else ""
    )
    if market.market_type == "game_total":
        actual_value = sum(float(team.score or 0.0) for team in game.teams)
        winning_outcome = "over" if evidence.side == "yes" else "under_or_equal"
    elif market.market_type == "spread" and selected_team is not None:
        opponent = next((team for team in game.teams if team.id != selected_team.id), None)
        actual_value = (selected_team.score or 0.0) - (opponent.score or 0.0) if opponent is not None else None
        winning_outcome = market.selected_outcome if evidence.side == "yes" else "not_" + market.selected_outcome
    else:
        actual_value = selected_team.score if selected_team is not None else None
        winning_outcome = market.selected_outcome if evidence.side == "yes" else "not_" + market.selected_outcome
    return {
        "scan_timestamp_utc": iso_utc(evidence.timestamp),
        "kalshi_market_ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "market_title": market.title,
        "series_ticker": market.series_ticker,
        "market_family": f"sports_{market.market_type}",
        "sport": market.sport_key,
        "league": market.league_name,
        "external_event_id": game.external_event_id,
        "external_event_name": game.external_event_name,
        "selected_outcome": market.selected_outcome,
        "winning_outcome": winning_outcome,
        "game_id": game.external_event_id,
        "official_date": official_date,
        "away_team": away_team.display_name if away_team is not None else "",
        "home_team": home_team.display_name if home_team is not None else "",
        "away_score": away_team.score if away_team is not None else None,
        "home_score": home_team.score if home_team is not None else None,
        "player_id": None,
        "player_name": None,
        "stat_type": market.market_type,
        "threshold": market.threshold,
        "actual_value": actual_value,
        "condition_comparator": "above",
        "condition_lower": None,
        "condition_upper": None,
        "source_key": selected_team.display_name if selected_team is not None else "game_total",
        "known_side": evidence.side,
        "verification_source": evidence.source,
        "verification_reason": evidence.reason,
        "game_status": evidence.game_status,
        "current_count": None,
        "final_count": None,
        "settlement_time_utc": iso_utc(settlement_ts),
        "settlement_time_source": settlement_source,
    }


def _base_scalar_row(
    market: ScalarMarket,
    observation: ScalarObservation,
    evidence: KnownOutcomeEvidence,
    settlement_ts: int,
    settlement_source: str,
) -> dict[str, Any]:
    return {
        "scan_timestamp_utc": iso_utc(evidence.timestamp),
        "kalshi_market_ticker": market.ticker,
        "event_ticker": market.event_ticker,
        "market_title": market.title,
        "series_ticker": market.series_ticker,
        "market_family": market.market_family,
        "sport": market.sport,
        "league": market.league,
        "external_event_id": market.source_key,
        "external_event_name": market.league,
        "selected_outcome": market.selected_outcome,
        "winning_outcome": market.selected_outcome if evidence.side == "yes" else "not_" + market.selected_outcome,
        "game_id": None,
        "official_date": market.target_date.isoformat(),
        "away_team": None,
        "home_team": None,
        "away_score": None,
        "home_score": None,
        "player_id": None,
        "player_name": None,
        "stat_type": market.stat_type,
        "threshold": market.threshold,
        "actual_value": observation.actual_value,
        "condition_comparator": market.comparator,
        "condition_lower": market.lower_bound,
        "condition_upper": market.upper_bound,
        "source_key": observation.source_key,
        "known_side": evidence.side,
        "verification_source": evidence.source,
        "verification_reason": evidence.reason,
        "game_status": evidence.game_status,
        "current_count": None,
        "final_count": None,
        "settlement_time_utc": iso_utc(settlement_ts),
        "settlement_time_source": settlement_source,
    }


KNOWN_OUTCOME_FIELDS = [
    "scan_timestamp_utc",
    "kalshi_market_ticker",
    "event_ticker",
    "market_title",
    "series_ticker",
    "market_family",
    "sport",
    "league",
    "external_event_id",
    "external_event_name",
    "selected_outcome",
    "winning_outcome",
    "game_id",
    "official_date",
    "away_team",
    "home_team",
    "away_score",
    "home_score",
    "player_id",
    "player_name",
    "stat_type",
    "threshold",
    "actual_value",
    "condition_comparator",
    "condition_lower",
    "condition_upper",
    "source_key",
    "known_side",
    "verification_source",
    "verification_reason",
    "game_status",
    "current_count",
    "final_count",
    "settlement_time_utc",
    "settlement_time_source",
    "liquidity_status",
    "orderbook_error",
    "requested_contracts",
    "min_contracts",
    "filled_contracts",
    "sizing_mode",
    "fillable_contracts_at_or_below_max",
    "best_executable_ask",
    "best_executable_ask_size",
    "worst_executable_ask",
    "execution_avg_price",
    "execution_cost",
    "fee_total",
    "fee_per_contract",
    "price_source",
    "opposing_bid_side",
    "best_opposing_bid_price",
    "worst_opposing_bid_price",
    "consumed_levels",
    "capital_per_contract",
    "capital_total",
    "gross_profit_per_contract",
    "gross_profit_total",
    "annual_yield",
    "settlement_seconds",
    "settlement_days",
    "carry_cost_per_contract",
    "carry_cost_total",
    "net_profit_per_contract",
    "net_profit_total",
    "net_return_on_capital",
    "annualized_net_return_on_capital",
    "breakeven_verifier_accuracy",
    "passes_filter",
]

KNOWN_OUTCOME_TRADE_FIELDS = [
    *KNOWN_OUTCOME_FIELDS,
    "entry_key",
    "entry_number",
    "model_name",
    "status",
    "settlement_value",
    "realized_pnl_per_contract",
    "realized_pnl_total",
    "apy_adjusted_realized_pnl_per_contract",
    "apy_adjusted_realized_pnl_total",
]

KNOWN_OUTCOME_PNL_FIELDS = [
    "model_name",
    "bucket",
    "market_family",
    "known_side",
    "trade_count",
    "filled_contracts",
    "capital_total",
    "execution_cost",
    "fee_total",
    "gross_profit_total",
    "carry_cost_total",
    "apy_adjusted_pnl_total",
    "avg_apy_adjusted_pnl_per_contract",
    "avg_breakeven_verifier_accuracy",
    "wins_by_verifier",
    "losses_by_verifier",
    "last_entry_timestamp_utc",
]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KNOWN_OUTCOME_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in KNOWN_OUTCOME_FIELDS})
    tmp_path.replace(path)


def _write_known_outcome_trade_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KNOWN_OUTCOME_TRADE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in KNOWN_OUTCOME_TRADE_FIELDS})
    tmp_path.replace(path)


def _write_known_outcome_pnl_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KNOWN_OUTCOME_PNL_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in KNOWN_OUTCOME_PNL_FIELDS})
    tmp_path.replace(path)


def _known_outcome_entry_key(row: dict[str, Any]) -> str:
    return "|".join(
        str(row.get(key) or "")
        for key in ("kalshi_market_ticker", "known_side")
    )


def _load_known_outcome_trade_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = []
        for row in csv.DictReader(f):
            ticker = str(row.get("kalshi_market_ticker") or "")
            if not ticker.startswith("K"):
                continue
            if not row.get("entry_number") or not row.get("status"):
                continue
            rows.append(dict(row))
        return rows


def _known_outcome_trade_row(row: dict[str, Any], entry_number: int) -> dict[str, Any]:
    gross_per_contract = parse_float(row.get("gross_profit_per_contract"))
    gross_total = parse_float(row.get("gross_profit_total"))
    net_per_contract = parse_float(row.get("net_profit_per_contract"))
    net_total = parse_float(row.get("net_profit_total"))
    return {
        **row,
        "entry_key": _known_outcome_entry_key(row),
        "entry_number": entry_number,
        "model_name": "known_outcome_carry_apy",
        "status": "verified_known_outcome",
        "settlement_value": 1.0,
        "realized_pnl_per_contract": gross_per_contract,
        "realized_pnl_total": gross_total,
        "apy_adjusted_realized_pnl_per_contract": net_per_contract,
        "apy_adjusted_realized_pnl_total": net_total,
    }


def _sum_float(rows: list[dict[str, Any]], field: str) -> float:
    return sum(parse_float(row.get(field)) or 0.0 for row in rows)


def _weighted_average(rows: list[dict[str, Any]], value_field: str, weight_field: str) -> float | None:
    total_weight = _sum_float(rows, weight_field)
    if total_weight <= 0:
        return None
    return sum((parse_float(row.get(value_field)) or 0.0) * (parse_float(row.get(weight_field)) or 0.0) for row in rows) / total_weight


def _known_outcome_trade_won(row: dict[str, Any]) -> bool:
    value = parse_float(row.get("settlement_value"))
    return value is None or value >= 1.0


def _known_outcome_pnl_rows(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in trades:
        key = (str(row.get("market_family") or "unknown"), str(row.get("known_side") or "unknown"))
        groups.setdefault(key, []).append(row)

    def summary(bucket: str, rows: list[dict[str, Any]], market_family: str = "ALL", known_side: str = "ALL") -> dict[str, Any]:
        filled = _sum_float(rows, "filled_contracts")
        wins = sum(1 for row in rows if _known_outcome_trade_won(row))
        return {
            "model_name": "known_outcome_carry_apy",
            "bucket": bucket,
            "market_family": market_family,
            "known_side": known_side,
            "trade_count": len(rows),
            "filled_contracts": filled,
            "capital_total": _sum_float(rows, "capital_total"),
            "execution_cost": _sum_float(rows, "execution_cost"),
            "fee_total": _sum_float(rows, "fee_total"),
            "gross_profit_total": _sum_float(rows, "gross_profit_total"),
            "carry_cost_total": _sum_float(rows, "carry_cost_total"),
            "apy_adjusted_pnl_total": _sum_float(rows, "apy_adjusted_realized_pnl_total"),
            "avg_apy_adjusted_pnl_per_contract": (_sum_float(rows, "apy_adjusted_realized_pnl_total") / filled) if filled > 0 else None,
            "avg_breakeven_verifier_accuracy": _weighted_average(rows, "breakeven_verifier_accuracy", "filled_contracts"),
            "wins_by_verifier": wins,
            "losses_by_verifier": len(rows) - wins,
            "last_entry_timestamp_utc": max((str(row.get("scan_timestamp_utc") or "") for row in rows), default=""),
        }

    out = [summary("ALL", trades)]
    for (market_family, known_side), rows in sorted(groups.items()):
        out.append(summary(f"{market_family}:{known_side}", rows, market_family, known_side))
    return out


def _update_known_outcome_trade_outputs(
    output_dir: Path,
    opportunities: list[dict[str, Any]],
) -> tuple[int, Path, Path]:
    trade_ledger_csv = output_dir / "known_outcome_trades.csv"
    pnl_csv = output_dir / "known_outcome_pnl.csv"
    existing = _load_known_outcome_trade_ledger(trade_ledger_csv)
    seen = {str(row.get("entry_key") or _known_outcome_entry_key(row)) for row in existing}
    trades = list(existing)
    next_entry = len(trades) + 1
    for row in opportunities:
        key = _known_outcome_entry_key(row)
        if key in seen:
            continue
        trades.append(_known_outcome_trade_row(row, next_entry))
        seen.add(key)
        next_entry += 1
    _write_known_outcome_trade_csv(trade_ledger_csv, trades)
    _write_known_outcome_pnl_csv(pnl_csv, _known_outcome_pnl_rows(trades))
    return len(trades), trade_ledger_csv, pnl_csv


def _date_in_range(value: date, start_date: str, end_date: str) -> bool:
    start = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()
    return start <= value <= end


def scan_known_outcomes(config: KnownOutcomeConfig) -> KnownOutcomeResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    start_date, end_date = _date_range(config)
    now_ts = int(time.time())
    kalshi = KalshiClient(
        timeout=8.0,
        max_retries=1,
        retry_sleep=0.25,
        api_key_id=config.api_key_id,
        private_key_path=config.private_key_path,
        private_key_pem=config.private_key_pem,
    )
    if not kalshi.has_auth:
        print(
            "Kalshi auth env vars are not set. If orderbook requests return 401, set "
            "KALSHI_ACCESS_KEY and KALSHI_PRIVATE_KEY_FILE or KALSHI_PRIVATE_KEY.",
            flush=True,
        )
    fee_kalshi = KalshiClient(
        timeout=5.0,
        max_retries=0,
        retry_sleep=0.1,
        api_key_id=config.api_key_id,
        private_key_path=config.private_key_path,
        private_key_pem=config.private_key_pem,
    )
    fee_info = _fee_info_by_series(fee_kalshi, config.series_ticker)
    raw_markets = _discover_markets(
        kalshi,
        config.series_ticker,
        config.max_market_pages,
        config.max_markets,
    )

    player_prop_markets = [market for market in raw_markets if is_supported_player_prop_market(market)]
    team_game_raw_markets = [
        market
        for market in raw_markets
        if config.include_team_game_winners and _is_supported_team_game_market(market)
    ]
    score_raw_markets = [
        market
        for market in raw_markets
        if config.include_score_markets and _is_supported_score_market(market)
    ]
    weather_raw_markets = [
        market
        for market in raw_markets
        if config.include_weather_markets and _is_supported_weather_market(market)
    ]
    finance_raw_markets = [
        market
        for market in raw_markets
        if config.include_finance_index_markets and _is_supported_finance_index_market(market)
    ]

    contexts: dict[int, GameContext] = {}
    matched_player_markets: list[MatchedMarket] = []
    if player_prop_markets:
        mlb = MlbClient()
        contexts = _load_game_contexts(mlb, start_date, end_date)
        matched_player_markets = _match_markets(player_prop_markets, contexts, start_date, end_date, fee_info)

    verified: list[VerifiedKnownMarket] = []
    for market in matched_player_markets:
        context = contexts.get(market.game_pk)
        if context is None:
            continue
        evidence = _known_mlb_outcome(market, context, now_ts, config)
        if evidence is not None:
            settlement_timestamp, settlement_source = _settlement_ts(
                market,
                now_ts,
                config.fallback_settlement_hours,
            )
            verified.append(
                VerifiedKnownMarket(
                    ticker=market.ticker,
                    raw_market=market.raw_market,
                    side=evidence.side,
                    fee_type=market.fee_type,
                    fee_multiplier=market.fee_multiplier,
                    settlement_ts=settlement_timestamp,
                    base_row=_base_row(market, context, evidence, settlement_timestamp, settlement_source),
                )
            )

    weather_markets: list[ScalarMarket] = []
    for raw_market in weather_raw_markets:
        market = _weather_market_from_raw(raw_market, fee_info)
        if market is None or not _date_in_range(market.target_date, start_date, end_date):
            continue
        weather_markets.append(market)

    if weather_markets:
        weather = WeatherKalshiClient(timeout=config.weather_timeout)
        for market in weather_markets:
            try:
                if market.target_hour is None:
                    observation = weather.fetch_daily_high(market.source_key, market.target_date)
                else:
                    observation = weather.fetch_hourly_temperature(
                        market.source_key,
                        _weather_location_from_text(_market_text(market.raw_market)),
                        market.target_date,
                        market.target_hour,
                    )
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping weather verifier for {market.ticker}: {exc}", flush=True)
                continue
            if observation is None:
                continue
            evidence = _scalar_evidence(market, observation, now_ts, config)
            if evidence is None:
                continue
            settlement_timestamp, settlement_source = _settlement_ts(
                market,
                now_ts,
                config.fallback_settlement_hours,
            )
            verified.append(
                VerifiedKnownMarket(
                    ticker=market.ticker,
                    raw_market=market.raw_market,
                    side=evidence.side,
                    fee_type=market.fee_type,
                    fee_multiplier=market.fee_multiplier,
                    settlement_ts=settlement_timestamp,
                    base_row=_base_scalar_row(market, observation, evidence, settlement_timestamp, settlement_source),
                )
            )

    finance_markets: list[ScalarMarket] = []
    for raw_market in finance_raw_markets:
        market = _finance_index_market_from_raw(raw_market, fee_info)
        if market is None or not _date_in_range(market.target_date, start_date, end_date):
            continue
        if market.occurrence_ts is not None and market.occurrence_ts > now_ts:
            continue
        finance_markets.append(market)

    if finance_markets:
        finance = FinanceIndexClient(timeout=config.finance_timeout)
        for market in finance_markets:
            try:
                observation = finance.fetch_close(market.source_key, market.target_date)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping finance verifier for {market.ticker}: {exc}", flush=True)
                continue
            if observation is None:
                continue
            evidence = _scalar_evidence(market, observation, now_ts, config)
            if evidence is None:
                continue
            settlement_timestamp, settlement_source = _settlement_ts(
                market,
                now_ts,
                config.fallback_settlement_hours,
            )
            verified.append(
                VerifiedKnownMarket(
                    ticker=market.ticker,
                    raw_market=market.raw_market,
                    side=evidence.side,
                    fee_type=market.fee_type,
                    fee_multiplier=market.fee_multiplier,
                    settlement_ts=settlement_timestamp,
                    base_row=_base_scalar_row(market, observation, evidence, settlement_timestamp, settlement_source),
                )
            )

    team_game_markets: list[TeamGameMarket] = []
    for raw_market in team_game_raw_markets:
        market = _team_game_market_from_raw(raw_market, fee_info)
        if market is None or market.occurrence_ts is None or market.occurrence_ts > now_ts:
            continue
        occurrence_date = datetime.fromtimestamp(market.occurrence_ts, tz=UTC).date()
        if not _date_in_range(occurrence_date, start_date, end_date):
            continue
        team_game_markets.append(market)

    score_markets: list[ScoreMarket] = []
    for raw_market in score_raw_markets:
        market = _score_market_from_raw(raw_market, fee_info)
        if market is None or market.occurrence_ts is None or market.occurrence_ts > now_ts:
            continue
        occurrence_date = datetime.fromtimestamp(market.occurrence_ts, tz=UTC).date()
        if not _date_in_range(occurrence_date, start_date, end_date):
            continue
        score_markets.append(market)

    if team_game_markets or score_markets:
        espn = EspnCoreClient(timeout=config.espn_timeout, max_retries=config.espn_max_retries)
        espn_cache: dict[tuple[str, date], list[TeamGameResult]] = {}

        def games_for(league: EspnLeagueConfig, occurrence_date: date) -> list[TeamGameResult]:
            cache_key = (f"{league.sport_path}/{league.league_path}", occurrence_date)
            if cache_key not in espn_cache:
                try:
                    espn_cache[cache_key] = espn.fetch_team_games(league, occurrence_date)
                except Exception as exc:
                    print(f"Skipping ESPN {league.display_name} {occurrence_date}: {exc}", flush=True)
                    espn_cache[cache_key] = []
            return espn_cache[cache_key]

        for market in team_game_markets:
            league = TEAM_GAME_SERIES.get(market.series_ticker)
            if league is None or market.occurrence_ts is None:
                continue
            occurrence_date = datetime.fromtimestamp(market.occurrence_ts, tz=UTC).date()
            match = _match_team_game_result(market, games_for(league, occurrence_date))
            if match is None:
                continue
            game, selected_team = match
            evidence = _team_game_evidence(market, game, selected_team, now_ts, config)
            if evidence is None:
                continue
            settlement_timestamp, settlement_source = _settlement_ts(
                market,
                now_ts,
                config.fallback_settlement_hours,
            )
            verified.append(
                VerifiedKnownMarket(
                    ticker=market.ticker,
                    raw_market=market.raw_market,
                    side=evidence.side,
                    fee_type=market.fee_type,
                    fee_multiplier=market.fee_multiplier,
                    settlement_ts=settlement_timestamp,
                    base_row=_base_team_game_row(market, game, selected_team, evidence, settlement_timestamp, settlement_source),
                )
            )
        for market in score_markets:
            score_config = SCORE_MARKET_SERIES.get(market.series_ticker)
            if score_config is None or market.occurrence_ts is None:
                continue
            league, _ = score_config
            occurrence_date = datetime.fromtimestamp(market.occurrence_ts, tz=UTC).date()
            match = _match_score_game_result(market, games_for(league, occurrence_date))
            if match is None:
                continue
            game, selected_team = match
            evidence = _score_market_evidence(market, game, selected_team, now_ts, config)
            if evidence is None:
                continue
            settlement_timestamp, settlement_source = _settlement_ts(
                market,
                now_ts,
                config.fallback_settlement_hours,
            )
            verified.append(
                VerifiedKnownMarket(
                    ticker=market.ticker,
                    raw_market=market.raw_market,
                    side=evidence.side,
                    fee_type=market.fee_type,
                    fee_multiplier=market.fee_multiplier,
                    settlement_ts=settlement_timestamp,
                    base_row=_base_score_market_row(market, game, selected_team, evidence, settlement_timestamp, settlement_source),
                )
            )

    orderbooks = _fetch_orderbooks([item.ticker for item in verified], config)
    rows: list[dict[str, Any]] = []
    opportunities: list[dict[str, Any]] = []
    priced_count = 0
    orderbook_errors = 0
    no_liquidity = 0
    for item in verified:
        row = dict(item.base_row)
        payload, error = orderbooks.get(item.ticker, (None, "missing orderbook result"))
        if payload is None:
            orderbook_errors += 1
            row.update({"liquidity_status": "orderbook_error", "orderbook_error": error, "passes_filter": 0})
            rows.append(row)
            print(f"Orderbook fetch failed for {item.ticker}: {error}", flush=True)
            continue
        plan = execution_plan_from_orderbook(
            payload=payload,
            timestamp=now_ts,
            side=item.side,
            contracts=config.contracts,
            min_contracts=config.min_contracts,
            require_full_contracts=config.require_full_contracts,
            max_ask=config.max_ask,
            fee_type=item.fee_type,
            fee_multiplier=item.fee_multiplier,
        )
        if plan is None:
            no_liquidity += 1
            row.update({"liquidity_status": "no_fillable_ask_at_or_below_max", "orderbook_error": "", "passes_filter": 0})
            rows.append(row)
            continue
        economics = carry_adjusted_economics(
            avg_price=plan.execution_avg_price,
            contracts=plan.filled_contracts,
            fee_total=plan.fee_total,
            annual_yield=config.annual_yield,
            settlement_seconds=max(0, item.settlement_ts - now_ts),
        )
        priced_count += 1
        row.update(
            {
                "liquidity_status": "fillable",
                "orderbook_error": "",
                "requested_contracts": plan.requested_contracts,
                "min_contracts": plan.min_contracts,
                "filled_contracts": plan.filled_contracts,
                "sizing_mode": plan.sizing_mode,
                "fillable_contracts_at_or_below_max": plan.fillable_contracts_at_or_below_max,
                "best_executable_ask": plan.best_executable_ask,
                "best_executable_ask_size": plan.best_executable_ask_size,
                "worst_executable_ask": plan.worst_executable_ask,
                "execution_avg_price": plan.execution_avg_price,
                "execution_cost": plan.execution_cost,
                "fee_total": plan.fee_total,
                "fee_per_contract": plan.fee_total / plan.filled_contracts,
                "price_source": plan.price_source,
                "opposing_bid_side": plan.opposing_bid_side,
                "best_opposing_bid_price": plan.best_opposing_bid_price,
                "worst_opposing_bid_price": plan.worst_opposing_bid_price,
                "consumed_levels": plan.consumed_levels,
                "capital_per_contract": economics.capital_per_contract,
                "capital_total": economics.capital_total,
                "gross_profit_per_contract": economics.gross_profit_per_contract,
                "gross_profit_total": economics.gross_profit_total,
                "annual_yield": economics.annual_yield,
                "settlement_seconds": economics.settlement_seconds,
                "settlement_days": economics.settlement_days,
                "carry_cost_per_contract": economics.carry_cost_per_contract,
                "carry_cost_total": economics.carry_cost_total,
                "net_profit_per_contract": economics.net_profit_per_contract,
                "net_profit_total": economics.net_profit_total,
                "net_return_on_capital": economics.net_return_on_capital,
                "annualized_net_return_on_capital": economics.annualized_net_return_on_capital,
                "breakeven_verifier_accuracy": economics.breakeven_verifier_accuracy,
                "passes_filter": 1 if economics.net_profit_per_contract >= config.min_net_profit_per_contract else 0,
            }
        )
        rows.append(row)
        if int(row["passes_filter"]) == 1:
            opportunities.append(row)

    rows.sort(key=lambda item: (parse_float(item.get("net_profit_total")) or -999.0), reverse=True)
    opportunities.sort(key=lambda item: (parse_float(item.get("net_profit_total")) or -999.0), reverse=True)
    candidates_csv = config.output_dir / "known_outcome_candidates.csv"
    opportunities_csv = config.output_dir / "known_outcome_opportunities.csv"
    _write_csv(candidates_csv, rows)
    _write_csv(opportunities_csv, opportunities)
    trade_count, trade_ledger_csv, pnl_csv = _update_known_outcome_trade_outputs(config.output_dir, opportunities)
    return KnownOutcomeResult(
        raw_markets=len(raw_markets),
        matched_markets=(
            len(matched_player_markets)
            + len(team_game_markets)
            + len(score_markets)
            + len(weather_markets)
            + len(finance_markets)
        ),
        verified_markets=len(verified),
        priced_markets=priced_count,
        opportunities=len(opportunities),
        orderbook_errors=orderbook_errors,
        no_liquidity=no_liquidity,
        candidates_csv=candidates_csv,
        opportunities_csv=opportunities_csv,
        trade_count=trade_count,
        trade_ledger_csv=trade_ledger_csv,
        pnl_csv=pnl_csv,
    )
