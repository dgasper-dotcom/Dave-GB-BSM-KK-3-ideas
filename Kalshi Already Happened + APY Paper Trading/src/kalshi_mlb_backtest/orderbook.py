from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .utils import parse_float, unix_seconds


@dataclass(frozen=True)
class SnapshotQuote:
    timestamp: int
    best_yes_bid: float | None
    best_yes_bid_size: float | None
    best_yes_ask: float | None
    best_yes_ask_size: float | None
    best_no_bid: float | None
    best_no_bid_size: float | None
    best_no_ask: float | None
    best_no_ask_size: float | None
    last_trade_price: float | None
    volume: float | None
    source: str
    exact_orderbook_available: bool
    data_quality_note: str


@dataclass(frozen=True)
class OrderbookLevel:
    side: str
    price: float
    quantity: float


def _nested_price(candle: dict[str, Any], side: str, field: str = "close") -> float | None:
    payload = candle.get(side) or {}
    return parse_float(payload.get(f"{field}_dollars") or payload.get(field))


def _orderbook_price(value: Any) -> float | None:
    parsed = parse_float(value)
    if parsed is None:
        return None
    if parsed > 1.0:
        parsed = parsed / 100.0
    return round(parsed, 4)


def quote_from_candlestick(candle: dict[str, Any]) -> SnapshotQuote | None:
    timestamp = unix_seconds(candle.get("end_period_ts"))
    if timestamp is None:
        return None
    yes_bid = _nested_price(candle, "yes_bid", "close")
    yes_ask = _nested_price(candle, "yes_ask", "close")
    no_bid = round(1.0 - yes_ask, 4) if yes_ask is not None else None
    no_ask = round(1.0 - yes_bid, 4) if yes_bid is not None else None
    price_payload = candle.get("price") or {}
    last_trade = parse_float(
        price_payload.get("previous_dollars")
        or price_payload.get("close_dollars")
        or price_payload.get("previous")
        or price_payload.get("close")
    )
    return SnapshotQuote(
        timestamp=timestamp,
        best_yes_bid=yes_bid,
        best_yes_bid_size=None,
        best_yes_ask=yes_ask,
        best_yes_ask_size=None,
        best_no_bid=no_bid,
        best_no_bid_size=None,
        best_no_ask=no_ask,
        best_no_ask_size=None,
        last_trade_price=last_trade,
        volume=parse_float(candle.get("volume_fp")),
        source="kalshi_candlestick_1m",
        exact_orderbook_available=False,
        data_quality_note=(
            "Kalshi one-minute candlestick. Prices are historical top-of-book/last-price fields; "
            "historical size and depth are not available in this payload."
        ),
    )


def levels_from_orderbook_payload(payload: dict[str, Any]) -> list[OrderbookLevel]:
    book = payload.get("orderbook_fp") or payload.get("orderbook") or {}
    levels: list[OrderbookLevel] = []
    side_keys = (
        ("yes_dollars" if "yes_dollars" in book else "yes", "yes_bid"),
        ("no_dollars" if "no_dollars" in book else "no", "no_bid"),
    )
    for side_key, side in side_keys:
        for raw_level in book.get(side_key) or []:
            if not isinstance(raw_level, (list, tuple)) or len(raw_level) < 2:
                continue
            price = _orderbook_price(raw_level[0])
            quantity = parse_float(raw_level[1])
            if price is None or quantity is None:
                continue
            levels.append(OrderbookLevel(side=side, price=price, quantity=quantity))
    return levels


def buy_no_ask_levels_from_orderbook_payload(payload: dict[str, Any]) -> list[OrderbookLevel]:
    bid_levels = levels_from_orderbook_payload(payload)
    yes_bid_levels = sorted([lvl for lvl in bid_levels if lvl.side == "yes_bid"], key=lambda item: item.price, reverse=True)
    return [
        OrderbookLevel(side="no_ask", price=round(1.0 - lvl.price, 4), quantity=lvl.quantity)
        for lvl in yes_bid_levels
    ]


def buy_no_depth_at_or_below(levels: list[OrderbookLevel], threshold: float) -> float:
    return sum(level.quantity for level in levels if level.side == "no_ask" and level.price <= threshold)


def quote_from_exact_orderbook(
    timestamp: int,
    payload: dict[str, Any],
    source: str = "kalshi_exact_orderbook",
) -> tuple[SnapshotQuote, list[OrderbookLevel]]:
    levels = levels_from_orderbook_payload(payload)
    yes_levels = sorted([lvl for lvl in levels if lvl.side == "yes_bid"], key=lambda item: item.price, reverse=True)
    no_levels = sorted([lvl for lvl in levels if lvl.side == "no_bid"], key=lambda item: item.price, reverse=True)
    best_yes_bid = yes_levels[0].price if yes_levels else None
    best_yes_bid_size = yes_levels[0].quantity if yes_levels else None
    best_no_bid = no_levels[0].price if no_levels else None
    best_no_bid_size = no_levels[0].quantity if no_levels else None
    # Binary-market conversion: a YES bid at X is a NO ask at 1-X, and a NO bid
    # at X is a YES ask at 1-X.
    best_no_ask = round(1.0 - best_yes_bid, 4) if best_yes_bid is not None else None
    best_no_ask_size = best_yes_bid_size
    best_yes_ask = round(1.0 - best_no_bid, 4) if best_no_bid is not None else None
    best_yes_ask_size = best_no_bid_size
    ask_levels = buy_no_ask_levels_from_orderbook_payload(payload) + [
        OrderbookLevel(side="yes_ask", price=round(1.0 - lvl.price, 4), quantity=lvl.quantity)
        for lvl in no_levels
    ]
    quote = SnapshotQuote(
        timestamp=timestamp,
        best_yes_bid=best_yes_bid,
        best_yes_bid_size=best_yes_bid_size,
        best_yes_ask=best_yes_ask,
        best_yes_ask_size=best_yes_ask_size,
        best_no_bid=best_no_bid,
        best_no_bid_size=best_no_bid_size,
        best_no_ask=best_no_ask,
        best_no_ask_size=best_no_ask_size,
        last_trade_price=None,
        volume=None,
        source=source,
        exact_orderbook_available=True,
        data_quality_note="Exact orderbook payload captured at this timestamp.",
    )
    return quote, levels + ask_levels
