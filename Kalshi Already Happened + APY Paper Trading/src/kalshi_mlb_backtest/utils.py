from __future__ import annotations

import math
import re
import unicodedata
from datetime import datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo


EASTERN = ZoneInfo("America/New_York")
UTC = timezone.utc


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def iso_utc(value: datetime | int | float | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromtimestamp(float(value), tz=UTC)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC).isoformat().replace("+00:00", "Z")


def unix_seconds(value: datetime | str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        raw = int(value)
        return raw // 1000 if raw > 10_000_000_000 else raw
    if isinstance(value, datetime):
        dt = value
    else:
        dt = parse_iso_datetime(str(value))
        if dt is None:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.astimezone(UTC).timestamp())


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def parse_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def cents_from_dollars(value: Any) -> int | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    return int(round(parsed * 100))


def dollars_from_cents(value: int | float | None) -> float | None:
    if value is None:
        return None
    return round(float(value) / 100.0, 4)


def normalize_name(value: str | None) -> str:
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_text = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    ascii_text = ascii_text.lower()
    ascii_text = ascii_text.replace("&", " and ")
    ascii_text = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", ascii_text).strip()


TEAM_ALIASES = {
    "a s": "athletics",
    "as": "athletics",
    "ath": "athletics",
    "oakland": "athletics",
    "sacramento": "athletics",
    "arizona": "diamondbacks",
    "az": "diamondbacks",
    "atlanta": "braves",
    "atl": "braves",
    "baltimore": "orioles",
    "bal": "orioles",
    "boston": "red sox",
    "bos": "red sox",
    "chicago c": "cubs",
    "chc": "cubs",
    "chicago cubs": "cubs",
    "chicago ws": "white sox",
    "chicago w": "white sox",
    "chw": "white sox",
    "cws": "white sox",
    "chicago white sox": "white sox",
    "cincinnati": "reds",
    "cin": "reds",
    "cleveland": "guardians",
    "cle": "guardians",
    "colorado": "rockies",
    "col": "rockies",
    "detroit": "tigers",
    "det": "tigers",
    "houston": "astros",
    "hou": "astros",
    "kansas city": "royals",
    "kc": "royals",
    "kcr": "royals",
    "los angeles a": "angels",
    "laa": "angels",
    "los angeles angels": "angels",
    "los angeles d": "dodgers",
    "lad": "dodgers",
    "los angeles dodgers": "dodgers",
    "miami": "marlins",
    "mia": "marlins",
    "milwaukee": "brewers",
    "mil": "brewers",
    "minnesota": "twins",
    "min": "twins",
    "new york m": "mets",
    "nym": "mets",
    "new york mets": "mets",
    "new york y": "yankees",
    "nyy": "yankees",
    "new york yankees": "yankees",
    "philadelphia": "phillies",
    "phi": "phillies",
    "pittsburgh": "pirates",
    "pit": "pirates",
    "san diego": "padres",
    "sd": "padres",
    "sdp": "padres",
    "san francisco": "giants",
    "sf": "giants",
    "sfg": "giants",
    "seattle": "mariners",
    "sea": "mariners",
    "st louis": "cardinals",
    "stl": "cardinals",
    "tampa bay": "rays",
    "tb": "rays",
    "tbr": "rays",
    "texas": "rangers",
    "tex": "rangers",
    "toronto": "blue jays",
    "tor": "blue jays",
    "washington": "nationals",
    "wsh": "nationals",
    "was": "nationals",
}


def normalize_team(value: str | None) -> str:
    key = normalize_name(value)
    return TEAM_ALIASES.get(key, key)


def previous_calendar_window(
    run_date: str | None,
    days: int,
    tz_name: str,
) -> tuple[datetime, datetime, str, str]:
    tz = ZoneInfo(tz_name)
    if run_date:
        local_date = datetime.strptime(run_date, "%Y-%m-%d").date()
    else:
        local_date = datetime.now(tz).date()
    end_local = datetime.combine(local_date, time.min, tzinfo=tz)
    start_local = end_local - timedelta(days=days)
    return (
        start_local.astimezone(UTC),
        end_local.astimezone(UTC),
        start_local.date().isoformat(),
        (end_local.date() - timedelta(days=1)).isoformat(),
    )


def bucket_no_price(price: float | None) -> str | None:
    if price is None:
        return None
    cents = int(round(price * 100))
    if 85 <= cents <= 89:
        return "85-89c"
    if 90 <= cents <= 91:
        return "90-91c"
    if 92 <= cents <= 93:
        return "92-93c"
    if 94 <= cents <= 95:
        return "94-95c"
    if 96 <= cents <= 97:
        return "96-97c"
    if cents == 98:
        return "98c"
    if cents == 99:
        return "99c"
    if cents == 100:
        return "100c"
    return None


PRICE_BUCKET_ORDER = ["85-89c", "90-91c", "92-93c", "94-95c", "96-97c", "98c", "99c", "100c"]
