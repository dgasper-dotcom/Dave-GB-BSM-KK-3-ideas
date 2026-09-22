from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from .mlb_data import GameContext
from .utils import EASTERN, normalize_name, parse_float, parse_iso_datetime, unix_seconds


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

SUPPORTED_SERIES = {
    "KXMLBHIT": "hits",
    "KXMLBHR": "home_runs",
}

PLAYER_PROP_TITLE_RE = re.compile(
    r"^\s*(?P<player>.+?)\s*:\s*(?P<threshold>\d+)\+\s*"
    r"(?P<label>hits?|home\s*runs?|hrs?)?\s*\?\s*$",
    re.IGNORECASE,
)
TITLE_RE = PLAYER_PROP_TITLE_RE
EVENT_TICKER_RE = re.compile(
    r"^KXMLB(?:HIT|HR)-(?P<yy>\d{2})(?P<mon>[A-Z]{3})(?P<dd>\d{2})(?P<hhmm>\d{4})"
)
MARKET_THRESHOLD_RE = re.compile(r"-(?P<threshold>\d+)$")
RULE_GAME_RE = re.compile(
    r"in the (?P<away>.+?) vs (?P<home>.+?) professional baseball game",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PlayerPropSpec:
    stat_type: str
    threshold: int
    series_ticker: str


@dataclass(frozen=True)
class MatchedMarket:
    ticker: str
    event_ticker: str
    title: str
    player_name: str
    kalshi_player_uuid: str | None
    game_pk: int
    mlb_player_id: int
    stat_type: str
    threshold: int
    fee_type: str | None
    fee_multiplier: float | None
    settlement_time: str | None
    result: str | None
    settlement_value: float | None
    raw_market: dict[str, Any]


def market_series_ticker(market: dict[str, Any]) -> str | None:
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    for series in SUPPORTED_SERIES:
        if ticker.startswith(f"{series}-") or event_ticker.startswith(f"{series}-"):
            return series
    return None


def parse_player_prop_spec(market: dict[str, Any]) -> PlayerPropSpec | None:
    series = market_series_ticker(market)
    if series is None:
        return None
    stat_type = SUPPORTED_SERIES[series]
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    title = str(market.get("title") or "")
    if ticker and not ticker.startswith(f"{series}-"):
        return None
    if event_ticker and not event_ticker.startswith(f"{series}-"):
        return None
    if "hits + runs" in title.lower() or "total bases" in title.lower():
        return None
    title_match = PLAYER_PROP_TITLE_RE.match(title)
    if not title_match:
        return None
    threshold = int(title_match.group("threshold"))
    label = (title_match.group("label") or "").lower().replace(" ", "")
    if label:
        if stat_type == "hits" and label not in {"hit", "hits"}:
            return None
        if stat_type == "home_runs" and label not in {"homerun", "homeruns", "hr", "hrs"}:
            return None
    ticker_match = MARKET_THRESHOLD_RE.search(ticker)
    if ticker_match and int(ticker_match.group("threshold")) != threshold:
        return None
    if threshold < 1:
        return None
    return PlayerPropSpec(stat_type=stat_type, threshold=threshold, series_ticker=series)


def is_supported_player_prop_market(market: dict[str, Any]) -> bool:
    return parse_player_prop_spec(market) is not None


def is_2_plus_hits_market(market: dict[str, Any]) -> bool:
    spec = parse_player_prop_spec(market)
    return spec is not None and spec.stat_type == "hits" and spec.threshold == 2


def parse_player_name(market: dict[str, Any]) -> str | None:
    match = PLAYER_PROP_TITLE_RE.match(str(market.get("title") or ""))
    if not match:
        return None
    return match.group("player").strip()


def parse_event_start_ts(event_ticker: str) -> int | None:
    match = EVENT_TICKER_RE.match(event_ticker)
    if not match:
        return None
    year = 2000 + int(match.group("yy"))
    month = MONTHS.get(match.group("mon"))
    day = int(match.group("dd"))
    hhmm = match.group("hhmm")
    hour = int(hhmm[:2])
    minute = int(hhmm[2:])
    if month is None:
        return None
    try:
        local_dt = datetime(year, month, day, hour, minute, tzinfo=EASTERN)
    except ValueError:
        return None
    return int(local_dt.astimezone(timezone.utc).timestamp())


def parse_rule_teams(market: dict[str, Any]) -> tuple[str | None, str | None]:
    rules = str(market.get("rules_primary") or "")
    match = RULE_GAME_RE.search(rules)
    if not match:
        return None, None
    return match.group("away").strip(), match.group("home").strip()


def market_settlement_value(market: dict[str, Any]) -> float | None:
    return parse_float(market.get("settlement_value_dollars"))


def market_settlement_time(market: dict[str, Any]) -> str | None:
    for key in ("settlement_ts", "close_time", "expiration_time", "expected_expiration_time"):
        dt = parse_iso_datetime(market.get(key))
        if dt is not None:
            return dt.isoformat().replace("+00:00", "Z")
    return None


def is_binary_settled_yes_no(market: dict[str, Any]) -> bool:
    result = str(market.get("result") or "").lower()
    if result not in {"yes", "no"}:
        return False
    value = market_settlement_value(market)
    if value is None:
        return True
    return value in {0.0, 1.0}


def match_game_context(
    market: dict[str, Any],
    contexts: dict[int, GameContext],
    official_date_min: str,
    official_date_max: str,
) -> GameContext | None:
    event_ticker = str(market.get("event_ticker") or "")
    start_ts = parse_event_start_ts(event_ticker)
    away_name, home_name = parse_rule_teams(market)
    candidates = [
        ctx
        for ctx in contexts.values()
        if official_date_min <= ctx.game.official_date <= official_date_max
    ]
    scored = [
        (ctx.game_team_match_score(away_name, home_name, start_ts), ctx)
        for ctx in candidates
    ]
    scored = [(score, ctx) for score, ctx in scored if score >= 4]
    if not scored:
        return None
    scored.sort(key=lambda item: (-item[0], abs(item[1].game.game_date_utc - (start_ts or item[1].game.game_date_utc))))
    return scored[0][1]


def match_market_to_game_and_player(
    market: dict[str, Any],
    contexts: dict[int, GameContext],
    official_date_min: str,
    official_date_max: str,
    fallback_fee_type: str | None,
    fallback_fee_multiplier: float | None,
) -> MatchedMarket | None:
    spec = parse_player_prop_spec(market)
    if spec is None:
        return None
    if not is_binary_settled_yes_no(market):
        return None
    player_name = parse_player_name(market)
    if not player_name:
        return None
    context = match_game_context(market, contexts, official_date_min, official_date_max)
    if context is None:
        return None
    player_id = context.match_player_id(player_name)
    if player_id is None:
        return None
    custom_strike = market.get("custom_strike") or {}
    if not isinstance(custom_strike, dict):
        custom_strike = {}
    return MatchedMarket(
        ticker=str(market.get("ticker") or ""),
        event_ticker=str(market.get("event_ticker") or ""),
        title=str(market.get("title") or ""),
        player_name=context.player_names.get(player_id) or player_name,
        kalshi_player_uuid=custom_strike.get("baseball_player"),
        game_pk=context.game.game_pk,
        mlb_player_id=player_id,
        stat_type=spec.stat_type,
        threshold=spec.threshold,
        fee_type=market.get("fee_type") or fallback_fee_type,
        fee_multiplier=parse_float(market.get("fee_multiplier")) if market.get("fee_multiplier") is not None else fallback_fee_multiplier,
        settlement_time=market_settlement_time(market),
        result=str(market.get("result") or "").lower() or None,
        settlement_value=market_settlement_value(market),
        raw_market=market,
    )


def match_open_market_to_game_and_player(
    market: dict[str, Any],
    contexts: dict[int, GameContext],
    official_date_min: str,
    official_date_max: str,
    fallback_fee_type: str | None,
    fallback_fee_multiplier: float | None,
) -> MatchedMarket | None:
    spec = parse_player_prop_spec(market)
    if spec is None:
        return None
    player_name = parse_player_name(market)
    if not player_name:
        return None
    context = match_game_context(market, contexts, official_date_min, official_date_max)
    if context is None:
        return None
    player_id = context.match_player_id(player_name)
    if player_id is None:
        return None
    custom_strike = market.get("custom_strike") or {}
    if not isinstance(custom_strike, dict):
        custom_strike = {}
    return MatchedMarket(
        ticker=str(market.get("ticker") or ""),
        event_ticker=str(market.get("event_ticker") or ""),
        title=str(market.get("title") or ""),
        player_name=context.player_names.get(player_id) or player_name,
        kalshi_player_uuid=custom_strike.get("baseball_player"),
        game_pk=context.game_pk,
        mlb_player_id=player_id,
        stat_type=spec.stat_type,
        threshold=spec.threshold,
        fee_type=market.get("fee_type") or fallback_fee_type,
        fee_multiplier=parse_float(market.get("fee_multiplier")) if market.get("fee_multiplier") is not None else fallback_fee_multiplier,
        settlement_time=market_settlement_time(market),
        result=str(market.get("result") or "").lower() or None,
        settlement_value=market_settlement_value(market),
        raw_market=market,
    )


def player_name_matches(kalshi_name: str, mlb_name: str) -> bool:
    return normalize_name(kalshi_name) == normalize_name(mlb_name)
