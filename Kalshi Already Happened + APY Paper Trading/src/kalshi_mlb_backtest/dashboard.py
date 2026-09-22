from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


@dataclass(frozen=True)
class DashboardConfig:
    db_path: Path
    output_dir: Path
    host: str = "127.0.0.1"
    port: int = 8765
    refresh_seconds: int = 5
    live_limit: int = 300


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sum_float(rows: list[dict[str, Any]], key: str) -> float:
    return sum(value for value in (_as_float(row.get(key)) for row in rows) if value is not None)


def _avg_float(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in (_as_float(row.get(key)) for row in rows) if value is not None]
    return (sum(values) / len(values)) if values else None


def _file_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path), "size_bytes": 0, "mtime_utc": None}
    stat = path.stat()
    return {
        "exists": True,
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _sort_timestamp_desc(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: str(row.get("timestamp_utc") or row.get("scan_timestamp_utc") or row.get("latest_timestamp_utc") or ""),
        reverse=True,
    )


def _contract_label(row: dict[str, Any]) -> str:
    threshold = str(row.get("threshold") or "").strip()
    stat_type = str(row.get("stat_type") or "").strip()
    if stat_type == "home_runs":
        stat_label = "HR"
    elif stat_type == "hits":
        stat_label = "H"
    else:
        stat_label = stat_type or "prop"
    return f"{threshold}+ {stat_label}".strip()


def _trade_bucket(row: dict[str, Any]) -> tuple[str, str]:
    side = str(row.get("side") or "").lower()
    ask = _as_float(row.get("ask_price"))
    if side == "yes":
        if str(row.get("event_already_happened") or "") == "1":
            return "yes_settlement_hold", "Already happened YES"
        return "yes_scalp", "YES scalp"
    if side == "no":
        if ask is None:
            return "no_unknown", "NO unknown"
        if ask >= 0.90:
            return "no_90_plus", "NO >=90c"
        if ask >= 0.80:
            return "no_80_90", "NO 80-90c"
        if ask < 0.50:
            return "no_under_50", "NO <50c"
        return "no_50_80", "NO 50-80c"
    return "other", "Other"


def _enrich_trade(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    enriched["contract_label"] = _contract_label(row)
    bucket_key, bucket_label = _trade_bucket(row)
    enriched["dashboard_bucket"] = bucket_key
    enriched["dashboard_bucket_label"] = bucket_label
    pnl = _as_float(row.get("scalp_pnl_total"))
    if pnl is None:
        pnl = _as_float(row.get("model_exit_pnl_total"))
    if pnl is None:
        pnl = _as_float(row.get("stop_pnl_total"))
    if pnl is None:
        pnl = _as_float(row.get("realized_pnl_total"))
    if pnl is None:
        pnl = _as_float(row.get("mark_pnl_total"))
    enriched["dashboard_pnl_total"] = pnl
    return enriched


def _load_live_snapshots(db_path: Path, limit: int) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2.0)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error:
        return []
    try:
        rows = conn.execute(
            """
            SELECT
                s.snapshot_id,
                s.timestamp_utc,
                s.source,
                m.market_id,
                m.ticker AS kalshi_market_ticker,
                m.title AS market_title,
                m.stat_type,
                m.threshold,
                m.player AS player_name,
                m.player_id,
                m.game_id,
                gs.inning,
                gs.half_inning,
                gs.outs,
                gs.away_score,
                gs.home_score,
                gs.current_hits,
                gs.current_home_runs,
                gs.current_PA,
                gs.estimated_PA_remaining,
                gs.modeled_PA_remaining_mean,
                gs.modeled_PA_remaining_distribution,
                s.best_yes_bid,
                s.best_yes_ask,
                s.best_no_bid,
                s.best_no_ask,
                s.best_yes_bid_size,
                s.best_no_bid_size,
                s.best_yes_ask_size,
                s.best_no_ask_size,
                s.exact_orderbook_available,
                s.data_quality_note
            FROM orderbook_snapshots s
            JOIN markets m ON m.market_id = s.market_id
            LEFT JOIN game_states gs
              ON gs.game_id = m.game_id
             AND gs.player_id = m.player_id
             AND gs.timestamp = s.timestamp
            ORDER BY s.timestamp DESC, s.snapshot_id DESC
            LIMIT ?
            """,
            (int(limit),),
        ).fetchall()
    except sqlite3.Error:
        return []
    finally:
        conn.close()
    out = [dict(row) for row in rows]
    for row in out:
        row["contract_label"] = _contract_label(row)
    return out


def _summary(
    trades: list[dict[str, Any]],
    signals: list[dict[str, Any]],
    live_snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    target_hits = [row for row in trades if row.get("scalp_status") == "target_hit"]
    model_exits = [row for row in trades if row.get("status") == "model_exited"]
    stop_exits = [row for row in trades if row.get("status") == "stop_exited"]
    waiting = [row for row in trades if row.get("status") == "open" and row.get("scalp_status") == "waiting"]
    settled = [row for row in trades if row.get("status") == "settled"]
    open_trades = [row for row in trades if row.get("status") == "open"]
    latest_trade = max((str(row.get("timestamp_utc") or "") for row in trades), default="")
    latest_live = str(live_snapshots[0].get("timestamp_utc") or "") if live_snapshots else ""
    return {
        "trade_count": len(trades),
        "signal_count": len(signals),
        "live_snapshot_count": len(live_snapshots),
        "open_trade_count": len(open_trades),
        "waiting_scalp_count": len(waiting),
        "target_hit_count": len(target_hits),
        "model_exit_count": len(model_exits),
        "stop_exit_count": len(stop_exits),
        "settled_count": len(settled),
        "filled_contracts": _sum_float(trades, "filled_contracts"),
        "expected_profit_total": _sum_float(trades, "expected_profit_total"),
        "mark_pnl_total": _sum_float(trades, "mark_pnl_total"),
        "scalp_pnl_total": _sum_float(trades, "scalp_pnl_total"),
        "model_exit_pnl_total": _sum_float(trades, "model_exit_pnl_total"),
        "stop_pnl_total": _sum_float(trades, "stop_pnl_total"),
        "realized_pnl_total": _sum_float(trades, "realized_pnl_total"),
        "average_edge": _avg_float(trades, "edge_before_fee"),
        "average_ev": _avg_float(trades, "expected_profit_per_contract"),
        "latest_trade_timestamp_utc": latest_trade or None,
        "latest_live_timestamp_utc": latest_live or None,
    }


def _pnl_breakdown(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = ["yes_settlement_hold", "yes_scalp", "no_90_plus", "no_80_90", "no_under_50", "no_50_80", "no_unknown", "other"]
    grouped: dict[str, list[dict[str, Any]]] = {}
    labels: dict[str, str] = {}
    for row in trades:
        bucket_key = str(row.get("dashboard_bucket") or "other")
        bucket_label = str(row.get("dashboard_bucket_label") or "Other")
        grouped.setdefault(bucket_key, []).append(row)
        labels[bucket_key] = bucket_label

    out: list[dict[str, Any]] = []
    for bucket_key in order:
        rows = grouped.get(bucket_key, [])
        if not rows and bucket_key not in {"yes_scalp", "no_90_plus", "no_80_90", "no_under_50"}:
            continue
        out.append(
            {
                "bucket": bucket_key,
                "label": labels.get(
                    bucket_key,
                    {
                        "yes_scalp": "YES scalp",
                        "yes_settlement_hold": "Already happened YES",
                        "no_90_plus": "NO >=90c",
                        "no_80_90": "NO 80-90c",
                        "no_under_50": "NO <50c",
                    }.get(bucket_key, "Other"),
                ),
                "trade_count": len(rows),
                "filled_contracts": _sum_float(rows, "filled_contracts"),
                "expected_profit_total": _sum_float(rows, "expected_profit_total"),
                "mark_pnl_total": _sum_float(rows, "mark_pnl_total"),
                "scalp_pnl_total": _sum_float(rows, "scalp_pnl_total"),
                "model_exit_pnl_total": _sum_float(rows, "model_exit_pnl_total"),
                "stop_pnl_total": _sum_float(rows, "stop_pnl_total"),
                "realized_pnl_total": _sum_float(rows, "realized_pnl_total"),
                "dashboard_pnl_total": _sum_float(rows, "dashboard_pnl_total"),
                "average_edge": _avg_float(rows, "edge_before_fee"),
                "average_ev": _avg_float(rows, "expected_profit_per_contract"),
            }
        )
    return out


def _settled_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if _as_float(row.get("realized_pnl_total")) is not None]


def _settled_summary(label: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    settled = _settled_rows(rows)
    realized_values = [_as_float(row.get("realized_pnl_total")) or 0.0 for row in settled]
    filled_contracts = _sum_float(settled, "filled_contracts")
    entry_cost = _sum_float(settled, "execution_cost") + _sum_float(settled, "fee_total")
    realized_total = sum(realized_values)
    return {
        "label": label,
        "trade_count": len(settled),
        "filled_contracts": filled_contracts,
        "wins": sum(1 for value in realized_values if value > 1e-12),
        "losses": sum(1 for value in realized_values if value < -1e-12),
        "pushes": sum(1 for value in realized_values if abs(value) <= 1e-12),
        "realized_pnl_total": realized_total,
        "realized_pnl_per_contract": (realized_total / filled_contracts) if filled_contracts else None,
        "entry_cost": entry_cost,
        "roi": (realized_total / entry_cost) if entry_cost else None,
    }


def _settled_pnl(trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups = [
        ("All settled", trades),
        ("NO buy & hold", [row for row in trades if str(row.get("side") or "").lower() == "no"]),
        ("NO >=90c", [row for row in trades if row.get("dashboard_bucket") == "no_90_plus"]),
        ("NO 80-90c", [row for row in trades if row.get("dashboard_bucket") == "no_80_90"]),
        ("NO <50c", [row for row in trades if row.get("dashboard_bucket") == "no_under_50"]),
        ("NO 50-80c", [row for row in trades if row.get("dashboard_bucket") == "no_50_80"]),
        ("YES held", [row for row in trades if str(row.get("side") or "").lower() == "yes"]),
    ]
    return [_settled_summary(label, rows) for label, rows in groups]


def _calibration_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple[int, str]:
        active = 0 if str(row.get("active") or "") == "1" else 1
        return active, str(row.get("bucket") or "")

    return sorted(rows, key=sort_key)


def load_dashboard_state(config: DashboardConfig) -> dict[str, Any]:
    trades_csv = config.output_dir / "paper_trades.csv"
    signals_csv = config.output_dir / "model_signals.csv"
    calibration_csv = config.output_dir / "probability_calibration.csv"
    live_csv = config.output_dir / "live_no_ask_liquidity.csv"
    known_outcome_trades_csv = config.output_dir / "known_outcome_trades.csv"
    known_outcome_pnl_csv = config.output_dir / "known_outcome_pnl.csv"
    trades = [_enrich_trade(row) for row in _sort_timestamp_desc(_csv_rows(trades_csv))]
    signals = _sort_timestamp_desc(_csv_rows(signals_csv))
    for row in signals:
        row["contract_label"] = _contract_label(row)
    calibration = _calibration_rows(_csv_rows(calibration_csv))
    known_outcome_pnl = _csv_rows(known_outcome_pnl_csv)
    known_outcome_trades = _sort_timestamp_desc(_csv_rows(known_outcome_trades_csv))
    live_snapshots = _load_live_snapshots(config.db_path, config.live_limit)
    return {
        "generated_at_utc": _utc_now(),
        "refresh_seconds": config.refresh_seconds,
        "summary": _summary(trades, signals, live_snapshots),
        "pnl_breakdown": _pnl_breakdown(trades),
        "settled_pnl": _settled_pnl(trades),
        "calibration": calibration,
        "known_outcome_pnl": known_outcome_pnl,
        "known_outcome_trades": known_outcome_trades,
        "files": {
            "paper_trades": _file_status(trades_csv),
            "model_signals": _file_status(signals_csv),
            "probability_calibration": _file_status(calibration_csv),
            "live_no_ask_liquidity": _file_status(live_csv),
            "known_outcome_trades": _file_status(known_outcome_trades_csv),
            "known_outcome_pnl": _file_status(known_outcome_pnl_csv),
            "database": _file_status(config.db_path),
        },
        "trades": trades,
        "signals": signals,
        "live_snapshots": live_snapshots,
    }


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kalshi MLB Trade Monitor</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d8dee8;
      --soft: #eef2f6;
      --blue: #2364aa;
      --green: #16794c;
      --red: #b42318;
      --amber: #9a6700;
      --teal: #0f766e;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.35 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    button, input, select {
      font: inherit;
    }
    .topbar {
      position: sticky;
      top: 0;
      z-index: 10;
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: center;
      padding: 14px 18px;
      border-bottom: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.96);
      backdrop-filter: blur(8px);
    }
    h1 {
      margin: 0;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .subline {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }
    .statusbar {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 5px 8px;
      background: var(--panel);
      color: var(--muted);
      white-space: nowrap;
    }
    .dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: var(--green);
    }
    .dot.bad { background: var(--red); }
    .iconbtn {
      min-width: 34px;
      min-height: 30px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      color: var(--ink);
      cursor: pointer;
    }
    .iconbtn:hover {
      border-color: #aab4c3;
      background: #f9fafb;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1fr) 360px;
      gap: 14px;
      padding: 14px;
    }
    .main {
      min-width: 0;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .metrics {
      display: grid;
      grid-template-columns: repeat(7, minmax(128px, 1fr));
      gap: 8px;
    }
    .breakdown {
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 8px;
    }
    .settled {
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 8px;
    }
    .metric {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 10px;
      min-height: 74px;
    }
    .bucket {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 10px;
      min-height: 118px;
    }
    .bucketTop {
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 10px;
      margin-bottom: 8px;
    }
    .bucketTitle {
      font-weight: 700;
      letter-spacing: 0;
    }
    .bucketCount {
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    .bucketPnl {
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .bucketRows {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 10px;
      margin-top: 7px;
      color: var(--muted);
      font-size: 12px;
    }
    .bucketRows span:nth-child(even) {
      color: var(--ink);
      text-align: right;
    }
    .metric .label {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .metric .value {
      margin-top: 7px;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    .metric .aux {
      margin-top: 2px;
      color: var(--muted);
      font-size: 12px;
    }
    .toolbar {
      display: grid;
      grid-template-columns: minmax(220px, 1.5fr) repeat(4, minmax(116px, 0.7fr));
      gap: 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      padding: 10px;
    }
    .toolbar input, .toolbar select {
      width: 100%;
      min-height: 34px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--ink);
      padding: 6px 8px;
    }
    .tabs {
      display: flex;
      gap: 2px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      border-radius: 6px 6px 0 0;
      overflow: hidden;
    }
    .tab {
      border: 0;
      border-right: 1px solid var(--line);
      background: var(--panel);
      min-height: 38px;
      padding: 0 14px;
      cursor: pointer;
      color: var(--muted);
    }
    .tab.active {
      color: var(--ink);
      box-shadow: inset 0 -3px 0 var(--blue);
      font-weight: 650;
    }
    .tableSection {
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      min-width: 0;
      overflow: hidden;
    }
    .tableMeta {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
      padding: 8px 10px;
      border-bottom: 1px solid var(--line);
      color: var(--muted);
      font-size: 12px;
    }
    .tableWrap {
      max-height: calc(100vh - 310px);
      min-height: 360px;
      overflow: auto;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      min-width: 1120px;
    }
    th, td {
      border-bottom: 1px solid var(--soft);
      padding: 7px 8px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }
    th {
      position: sticky;
      top: 0;
      z-index: 1;
      background: #f8fafc;
      color: #475467;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      cursor: pointer;
    }
    tr:hover td {
      background: #f8fbff;
    }
    tr.selected td {
      background: #edf5ff;
    }
    .mono {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
    }
    .tag {
      display: inline-flex;
      align-items: center;
      min-height: 22px;
      border-radius: 4px;
      padding: 2px 6px;
      background: var(--soft);
      color: var(--ink);
      font-size: 12px;
      font-weight: 650;
    }
    .tag.yes { color: var(--green); background: #e8f5ee; }
    .tag.no { color: var(--red); background: #fcebea; }
    .tag.waiting { color: var(--amber); background: #fff4d6; }
    .tag.hit { color: var(--teal); background: #e6f4f1; }
    .pos { color: var(--green); font-weight: 650; }
    .neg { color: var(--red); font-weight: 650; }
    .details {
      position: sticky;
      top: 76px;
      align-self: start;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel);
      max-height: calc(100vh - 90px);
      overflow: auto;
    }
    .detailsHeader {
      padding: 12px;
      border-bottom: 1px solid var(--line);
    }
    .detailsHeader h2 {
      margin: 0;
      font-size: 16px;
      letter-spacing: 0;
    }
    .detailGrid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0;
    }
    .detailItem {
      border-bottom: 1px solid var(--soft);
      padding: 9px 10px;
      min-width: 0;
    }
    .detailItem:nth-child(odd) {
      border-right: 1px solid var(--soft);
    }
    .detailLabel {
      color: var(--muted);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .detailValue {
      margin-top: 3px;
      overflow-wrap: anywhere;
      word-break: break-word;
    }
    .wideDetail {
      grid-column: 1 / -1;
      border-right: 0 !important;
    }
    .empty {
      padding: 26px;
      color: var(--muted);
      text-align: center;
    }
    @media (max-width: 1180px) {
      .layout { grid-template-columns: 1fr; }
      .details { position: static; max-height: none; }
      .metrics { grid-template-columns: repeat(3, minmax(128px, 1fr)); }
      .breakdown { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
      .settled { grid-template-columns: repeat(2, minmax(180px, 1fr)); }
    }
    @media (max-width: 760px) {
      .topbar { align-items: flex-start; flex-direction: column; }
      .statusbar { justify-content: flex-start; }
      .layout { padding: 10px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .breakdown { grid-template-columns: 1fr; }
      .settled { grid-template-columns: 1fr; }
      .toolbar { grid-template-columns: 1fr 1fr; }
      .toolbar .search { grid-column: 1 / -1; }
      .tableWrap { max-height: none; }
    }
  </style>
</head>
<body>
  <header class="topbar">
    <div>
      <h1>Kalshi MLB Trade Monitor</h1>
      <div class="subline" id="subtitle">Loading</div>
    </div>
    <div class="statusbar">
      <span class="pill"><span class="dot" id="statusDot"></span><span id="statusText">Connecting</span></span>
      <span class="pill" id="lastUpdated">Updated --</span>
      <label class="pill"><input type="checkbox" id="autoRefresh" checked> Auto</label>
      <button class="iconbtn" id="refreshButton" title="Refresh now" aria-label="Refresh now">R</button>
      <a class="pill" href="/paper_trades.csv" target="_blank">MLB CSV</a>
      <a class="pill" href="/known_outcome_trades.csv" target="_blank">APY CSV</a>
    </div>
  </header>
  <div class="layout">
    <main class="main">
      <section class="metrics" id="metrics"></section>
      <section class="breakdown" id="pnlBreakdown"></section>
      <section class="settled" id="settledPnl"></section>
      <section class="settled" id="knownOutcomePnl"></section>
      <section class="settled" id="calibration"></section>
      <section class="toolbar">
        <input class="search" id="searchInput" placeholder="Search player, ticker, market" autocomplete="off">
        <select id="sideFilter">
          <option value="">All sides</option>
          <option value="yes">YES</option>
          <option value="no">NO</option>
        </select>
        <select id="statFilter">
          <option value="">All props</option>
          <option value="hits">Hits</option>
          <option value="home_runs">Home runs</option>
        </select>
      <select id="statusFilter">
        <option value="">All statuses</option>
        <option value="open">Open</option>
        <option value="scalp_exited">Scalp exited</option>
        <option value="model_exited">Model exited</option>
        <option value="stop_exited">Stop exited</option>
        <option value="settled">Settled</option>
        <option value="settlement_hold">Settlement hold</option>
        <option value="waiting">Waiting scalp</option>
        <option value="target_hit">Target hit</option>
        <option value="stop_exit">Stop hit</option>
      </select>
        <select id="sortSelect">
          <option value="time_desc">Newest</option>
          <option value="edge_desc">Edge</option>
          <option value="ev_desc">EV</option>
          <option value="pnl_desc">PnL</option>
          <option value="fill_desc">Fill size</option>
        </select>
      </section>
      <section class="tableSection">
        <div class="tabs">
          <button class="tab active" data-tab="trades">Trades</button>
          <button class="tab" data-tab="yes">YES</button>
          <button class="tab" data-tab="no">NO</button>
          <button class="tab" data-tab="signals">Signals</button>
          <button class="tab" data-tab="live">Live Snapshots</button>
        </div>
        <div class="tableMeta">
          <span id="tableCount">0 rows</span>
          <span id="fileTimes">Files --</span>
        </div>
        <div class="tableWrap">
          <table>
            <thead id="tableHead"></thead>
            <tbody id="tableBody"></tbody>
          </table>
        </div>
      </section>
    </main>
    <aside class="details" id="details">
      <div class="detailsHeader">
        <h2>Trade Details</h2>
        <div class="subline">Select a row</div>
      </div>
      <div class="empty">No row selected</div>
    </aside>
  </div>
  <script>
    const refreshSeconds = __REFRESH_SECONDS__;
    const app = {
      data: null,
      tab: "trades",
      selectedKey: null,
      timer: null
    };

    const columns = {
      trades: [
        ["timestamp_utc", "Time"],
        ["player_name", "Player"],
        ["contract_label", "Contract"],
        ["player_batting_proximity", "Prox"],
        ["side", "Side"],
        ["ask_price", "Ask"],
        ["fair_side", "Fair"],
        ["calibration_multiplier", "Cal"],
        ["hit_probability_model", "H/PA"],
        ["edge_before_fee", "Edge"],
        ["games_with_2plus_at_bats_rate_30d", "2+AB%"],
        ["expected_profit_per_contract", "EV/ct"],
        ["signal_confirmation_count", "Conf"],
        ["signal_persistence_rule", "Rule"],
        ["filled_contracts", "Fill"],
        ["status", "Status"],
        ["scalp_status", "Scalp"],
        ["stop_exit_reason", "Stop"],
        ["model_exit_edge_to_bid", "Exit edge"],
        ["dashboard_pnl_total", "PnL"],
        ["kalshi_market_ticker", "Ticker"]
      ],
      signals: [
        ["timestamp_utc", "Time"],
        ["player_name", "Player"],
        ["contract_label", "Contract"],
        ["player_batting_proximity", "Prox"],
        ["side", "Side"],
        ["ask_price", "Ask"],
        ["fair_side", "Fair"],
        ["calibration_multiplier", "Cal"],
        ["hit_probability_model", "H/PA"],
        ["edge_before_fee", "Edge"],
        ["games_with_2plus_at_bats_rate_30d", "2+AB%"],
        ["expected_profit_per_contract", "EV/ct"],
        ["signal_confirmation_count", "Conf"],
        ["signal_persistence_rule", "Rule"],
        ["filled_contracts", "Fill"],
        ["event_already_happened", "Done"],
        ["scalp_target_exit_price", "Target"],
        ["kalshi_market_ticker", "Ticker"]
      ],
      live: [
        ["timestamp_utc", "Time"],
        ["player_name", "Player"],
        ["contract_label", "Contract"],
        ["inning", "Inn"],
        ["half_inning", "Half"],
        ["outs", "Outs"],
        ["current_hits", "H"],
        ["current_home_runs", "HR"],
        ["current_PA", "PA"],
        ["best_yes_bid", "Y bid"],
        ["best_no_bid", "N bid"],
        ["best_yes_ask", "Y ask"],
        ["best_no_ask", "N ask"],
        ["kalshi_market_ticker", "Ticker"]
      ]
    };

    function num(value) {
      if (value === null || value === undefined || value === "") return null;
      const parsed = Number(value);
      return Number.isFinite(parsed) ? parsed : null;
    }

    function esc(value) {
      return String(value ?? "").replace(/[&<>"']/g, ch => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#39;"
      }[ch]));
    }

    function fmtTime(value) {
      if (!value) return "";
      return String(value).replace("T", " ").replace("Z", "");
    }

    function fmtPrice(value) {
      const n = num(value);
      if (n === null) return "";
      return `${(n * 100).toFixed(1)}c`;
    }

    function fmtSigned(value) {
      const n = num(value);
      if (n === null) return "";
      const cls = n >= 0 ? "pos" : "neg";
      return `<span class="${cls}">${n >= 0 ? "+" : ""}${(n * 100).toFixed(1)}c</span>`;
    }

    function fmtPct(value) {
      const n = num(value);
      if (n === null) return "";
      return `${(n * 100).toFixed(1)}%`;
    }

    function fmtPnl(value) {
      const n = num(value);
      if (n === null) return "";
      const cls = n >= 0 ? "pos" : "neg";
      return `<span class="${cls}">${n >= 0 ? "+" : ""}${n.toFixed(2)}</span>`;
    }

    function fmtCell(key, row) {
      const value = row[key];
      if (key === "timestamp_utc" || key === "latest_timestamp_utc") return esc(fmtTime(value));
      if (key === "signal_first_seen_timestamp_utc" || key === "signal_confirmed_timestamp_utc") return esc(fmtTime(value));
      if (key === "signal_persistence_seconds") {
        const n = num(value);
        return n === null ? "" : esc(`${n.toFixed(0)}s`);
      }
      if (key === "calibration_multiplier" || key === "calibration_raw_multiplier") {
        const n = num(value);
        return n === null ? "" : esc(`${n.toFixed(3)}x`);
      }
      if (["ask_price", "fair_side", "edge_before_fee", "expected_profit_per_contract", "scalp_target_exit_price", "best_yes_bid", "best_no_bid", "best_yes_ask", "best_no_ask"].includes(key)) {
        return key.includes("edge") || key.includes("expected_profit") ? fmtSigned(value) : esc(fmtPrice(value));
      }
      if (["games_with_2plus_at_bats_rate_30d", "hit_probability_model", "hit_probability_raw", "batting_average_raw", "batting_average_model", "calibration_average_prediction", "calibration_actual_rate"].includes(key)) return esc(fmtPct(value));
      if (key === "dashboard_pnl_total" || key === "model_exit_pnl_total" || key === "stop_pnl_total") return fmtPnl(value);
      if (key === "model_exit_edge_to_bid") return fmtSigned(value);
      if (key === "side") {
        const side = String(value || "").toLowerCase();
        return side ? `<span class="tag ${side}">${esc(side.toUpperCase())}</span>` : "";
      }
      if (key === "status" || key === "scalp_status") {
        const status = String(value || "");
        const cls = status === "target_hit" ? "hit" : ((status === "waiting" || status === "stop_exit" || status === "stop_exited") ? "waiting" : "");
        return status ? `<span class="tag ${cls}">${esc(status.replaceAll("_", " "))}</span>` : "";
      }
      if (key === "event_already_happened") return String(value) === "1" ? "yes" : "";
      if (key === "kalshi_market_ticker") return `<span class="mono">${esc(value)}</span>`;
      return esc(value);
    }

    function rowKey(row) {
      return `${row.snapshot_id || ""}:${row.market_id || ""}:${row.side || ""}:${row.entry_number || ""}`;
    }

    function passesFilters(row) {
      const q = document.getElementById("searchInput").value.trim().toLowerCase();
      const side = document.getElementById("sideFilter").value;
      const stat = document.getElementById("statFilter").value;
      const status = document.getElementById("statusFilter").value;
      if (q) {
        const haystack = `${row.player_name || ""} ${row.kalshi_market_ticker || ""} ${row.market_title || ""}`.toLowerCase();
        if (!haystack.includes(q)) return false;
      }
      if (side && String(row.side || "").toLowerCase() !== side) return false;
      if (stat && String(row.stat_type || "") !== stat) return false;
      if (status) {
        const rowStatus = String(row.status || "");
        const scalpStatus = String(row.scalp_status || "");
        if (rowStatus !== status && scalpStatus !== status) return false;
      }
      return true;
    }

    function sortedRows(rows) {
      const mode = document.getElementById("sortSelect").value;
      const copy = [...rows];
      const keyMap = {
        edge_desc: "edge_before_fee",
        ev_desc: "expected_profit_per_contract",
        pnl_desc: "dashboard_pnl_total",
        fill_desc: "filled_contracts"
      };
      const key = keyMap[mode];
      if (!key) return copy.sort((a, b) => String(b.timestamp_utc || "").localeCompare(String(a.timestamp_utc || "")));
      return copy.sort((a, b) => (num(b[key]) ?? -Infinity) - (num(a[key]) ?? -Infinity));
    }

    function activeRows() {
      if (!app.data) return [];
      let base;
      if (app.tab === "live") {
        base = app.data.live_snapshots;
      } else if (app.tab === "yes") {
        base = app.data.trades.filter(row => String(row.side || "").toLowerCase() === "yes");
      } else if (app.tab === "no") {
        base = app.data.trades.filter(row => String(row.side || "").toLowerCase() === "no");
      } else {
        base = app.data[app.tab];
      }
      return sortedRows(base.filter(passesFilters));
    }

    function renderMetrics() {
      const s = app.data.summary;
      const items = [
        ["Trades", s.trade_count, `${s.filled_contracts.toFixed(0)} contracts`],
        ["Signals", s.signal_count, `${s.live_snapshot_count} live rows`],
        ["Open", s.open_trade_count, `${s.waiting_scalp_count} waiting`],
        ["Targets", s.target_hit_count, `${s.model_exit_count || 0} model, ${s.stop_exit_count || 0} stops`],
        ["Expected", fmtPnl(s.expected_profit_total), "model EV"],
        ["Scalp PnL", fmtPnl(s.scalp_pnl_total), "target exits"],
        ["Model Exit", fmtPnl(s.model_exit_pnl_total), "EV compression"],
        ["Stop Exit", fmtPnl(s.stop_pnl_total), "time/PA/inning"],
        ["Mark PnL", fmtPnl(s.mark_pnl_total), "latest bid"]
      ];
      document.getElementById("metrics").innerHTML = items.map(([label, value, aux]) => `
        <div class="metric">
          <div class="label">${esc(label)}</div>
          <div class="value">${value}</div>
          <div class="aux">${esc(aux)}</div>
        </div>
      `).join("");
    }

    function renderBreakdown() {
      const rows = app.data.pnl_breakdown || [];
      document.getElementById("pnlBreakdown").innerHTML = rows.map(row => `
        <div class="bucket">
          <div class="bucketTop">
            <div class="bucketTitle">${esc(row.label)}</div>
            <div class="bucketCount">${esc(row.trade_count)} trades</div>
          </div>
          <div class="bucketPnl">${fmtPnl(row.dashboard_pnl_total)}</div>
          <div class="bucketRows">
            <span>Expected</span><span>${fmtPnl(row.expected_profit_total)}</span>
            <span>Mark</span><span>${fmtPnl(row.mark_pnl_total)}</span>
            <span>Scalp</span><span>${fmtPnl(row.scalp_pnl_total)}</span>
            <span>Model exit</span><span>${fmtPnl(row.model_exit_pnl_total)}</span>
            <span>Stop exit</span><span>${fmtPnl(row.stop_pnl_total)}</span>
            <span>Contracts</span><span>${esc(Number(row.filled_contracts || 0).toFixed(0))}</span>
          </div>
        </div>
      `).join("");
    }

    function renderSettledPnl() {
      const rows = app.data.settled_pnl || [];
      document.getElementById("settledPnl").innerHTML = rows.map(row => {
        const avg = num(row.realized_pnl_per_contract);
        const roi = num(row.roi);
        return `
          <div class="bucket">
            <div class="bucketTop">
              <div class="bucketTitle">${esc(row.label)}</div>
              <div class="bucketCount">${esc(row.trade_count)} settled</div>
            </div>
            <div class="bucketPnl">${fmtPnl(row.realized_pnl_total)}</div>
            <div class="bucketRows">
              <span>Contracts</span><span>${esc(Number(row.filled_contracts || 0).toFixed(0))}</span>
              <span>W/L/P</span><span>${esc(row.wins || 0)}/${esc(row.losses || 0)}/${esc(row.pushes || 0)}</span>
              <span>Avg/ct</span><span>${avg === null ? "" : fmtSigned(avg)}</span>
              <span>ROI</span><span>${roi === null ? "" : fmtPct(roi)}</span>
            </div>
          </div>
        `;
      }).join("");
    }

    function renderKnownOutcomePnl() {
      const rows = app.data.known_outcome_pnl || [];
      const filtered = rows.filter(row => row.bucket === "ALL" || Number(row.trade_count || 0) > 0);
      document.getElementById("knownOutcomePnl").innerHTML = filtered.map(row => {
        const avg = num(row.avg_apy_adjusted_pnl_per_contract);
        const accuracy = num(row.avg_breakeven_verifier_accuracy);
        return `
          <div class="bucket">
            <div class="bucketTop">
              <div class="bucketTitle">APY ${esc(row.bucket || "")}</div>
              <div class="bucketCount">${esc(row.trade_count || 0)} trades</div>
            </div>
            <div class="bucketPnl">${fmtPnl(row.apy_adjusted_pnl_total)}</div>
            <div class="bucketRows">
              <span>Contracts</span><span>${esc(Number(row.filled_contracts || 0).toFixed(0))}</span>
              <span>Gross</span><span>${fmtPnl(row.gross_profit_total)}</span>
              <span>Carry</span><span>${fmtPnl(-Number(row.carry_cost_total || 0))}</span>
              <span>Avg/ct</span><span>${avg === null ? "" : fmtSigned(avg)}</span>
              <span>Req accuracy</span><span>${accuracy === null ? "" : fmtPct(accuracy)}</span>
            </div>
          </div>
        `;
      }).join("");
    }

    function renderCalibration() {
      const rows = app.data.calibration || [];
      document.getElementById("calibration").innerHTML = rows.map(row => {
        const average = num(row.average_prediction);
        const actual = num(row.actual_rate);
        const multiplier = num(row.applied_multiplier);
        const active = String(row.active || "") === "1";
        return `
          <div class="bucket">
            <div class="bucketTop">
              <div class="bucketTitle">${esc(row.bucket || "")}</div>
              <div class="bucketCount">${esc(row.sample_count || 0)} settled</div>
            </div>
            <div class="bucketPnl">${multiplier === null ? "" : esc(`${multiplier.toFixed(3)}x`)}</div>
            <div class="bucketRows">
              <span>Pred</span><span>${average === null ? "" : fmtPct(average)}</span>
              <span>Actual</span><span>${actual === null ? "" : fmtPct(actual)}</span>
              <span>Raw mult</span><span>${row.raw_multiplier ? esc(`${Number(row.raw_multiplier).toFixed(3)}x`) : ""}</span>
              <span>Active</span><span>${active ? "yes" : ""}</span>
            </div>
          </div>
        `;
      }).join("");
    }

    function renderTable() {
      const rows = activeRows();
      const cols = columns[app.tab] || columns.trades;
      document.getElementById("tableHead").innerHTML = `<tr>${cols.map(([key, label]) => `<th data-key="${esc(key)}">${esc(label)}</th>`).join("")}</tr>`;
      document.getElementById("tableCount").textContent = `${rows.length} ${app.tab}`;
      if (!rows.length) {
        document.getElementById("tableBody").innerHTML = `<tr><td colspan="${cols.length}" class="empty">No rows</td></tr>`;
        return;
      }
      document.getElementById("tableBody").innerHTML = rows.map(row => {
        const key = rowKey(row);
        const selected = key === app.selectedKey ? " selected" : "";
        return `<tr class="${selected}" data-row-key="${esc(key)}">${cols.map(([col]) => `<td>${fmtCell(col, row)}</td>`).join("")}</tr>`;
      }).join("");
      document.querySelectorAll("#tableBody tr[data-row-key]").forEach(tr => {
        tr.addEventListener("click", () => {
          app.selectedKey = tr.dataset.rowKey;
          const row = rows.find(item => rowKey(item) === app.selectedKey);
          renderDetails(row);
          renderTable();
        });
      });
    }

    function renderDetails(row) {
      const target = document.getElementById("details");
      if (!row) {
        target.innerHTML = `<div class="detailsHeader"><h2>Trade Details</h2><div class="subline">Select a row</div></div><div class="empty">No row selected</div>`;
        return;
      }
      const title = `${row.player_name || "Player"} ${row.contract_label || ""} ${String(row.side || "").toUpperCase()}`;
      const fields = [
        ["Time", fmtTime(row.timestamp_utc)],
        ["Status", row.status || row.scalp_status || ""],
        ["Bucket", row.dashboard_bucket_label || ""],
        ["Batting proximity", row.player_batting_proximity],
        ["Ask", fmtPrice(row.ask_price)],
        ["Fair", fmtPrice(row.fair_side)],
        ["Raw fair", fmtPrice(row.fair_side_raw)],
        ["Edge", `${((num(row.edge_before_fee) ?? 0) * 100).toFixed(2)}c`],
        ["EV/ct", `${((num(row.expected_profit_per_contract) ?? 0) * 100).toFixed(2)}c`],
        ["Calibration bucket", row.calibration_bucket],
        ["Calibration samples", row.calibration_sample_size],
        ["Calibration predicted", fmtPct(row.calibration_average_prediction)],
        ["Calibration actual", fmtPct(row.calibration_actual_rate)],
        ["Calibration multiplier", row.calibration_multiplier ? `${Number(row.calibration_multiplier).toFixed(3)}x` : ""],
        ["Calibration active", String(row.calibration_active || "") === "1" ? "yes" : ""],
        ["Signal first seen", fmtTime(row.signal_first_seen_timestamp_utc)],
        ["Signal confirmed", fmtTime(row.signal_confirmed_timestamp_utc)],
        ["Signal confirmations", row.signal_confirmation_count],
        ["Signal persistence", row.signal_persistence_seconds ? `${Number(row.signal_persistence_seconds).toFixed(0)}s` : ""],
        ["Signal rule", row.signal_persistence_rule],
        ["BA", fmtPct(row.batting_average_raw)],
        ["Modeled BA", fmtPct(row.batting_average_model)],
        ["H/PA", fmtPct(row.hit_probability_model)],
        ["H/PA weights", row.hit_probability_blend_weights],
        ["30d 2+AB games", `${row.games_with_2plus_at_bats_30d || ""}/${row.games_played_30d || ""}`],
        ["30d 2+AB rate", fmtPct(row.games_with_2plus_at_bats_rate_30d)],
        ["Requested", row.requested_contracts],
        ["Filled", row.filled_contracts],
        ["Sizing", row.sizing_mode],
        ["Target", fmtPrice(row.scalp_target_exit_price)],
        ["Exit bid", fmtPrice(row.scalp_exit_bid)],
        ["Scalp PnL", row.scalp_pnl_total],
        ["Stop exit bid", fmtPrice(row.stop_exit_bid)],
        ["Stop reason", row.stop_exit_reason],
        ["Stop PnL", row.stop_pnl_total],
        ["Model exit bid", fmtPrice(row.model_exit_bid)],
        ["Model exit fair", fmtPrice(row.model_exit_fair_side)],
        ["Model exit edge", `${((num(row.model_exit_edge_to_bid) ?? 0) * 100).toFixed(2)}c`],
        ["Model exit PnL", row.model_exit_pnl_total],
        ["Current count", row.current_count],
        ["Remaining PA", row.remaining_pa_distribution],
        ["Price source", row.price_source],
        ["Consumed levels", row.consumed_levels],
        ["Market", row.kalshi_market_ticker],
        ["Title", row.market_title]
      ];
      target.innerHTML = `
        <div class="detailsHeader"><h2>${esc(title)}</h2><div class="subline">${esc(row.kalshi_market_ticker || "")}</div></div>
        <div class="detailGrid">
          ${fields.map(([label, value]) => {
            const wide = ["Consumed levels", "Market", "Title", "Price source", "Remaining PA"].includes(label) ? " wideDetail" : "";
            return `<div class="detailItem${wide}"><div class="detailLabel">${esc(label)}</div><div class="detailValue">${esc(value ?? "")}</div></div>`;
          }).join("")}
        </div>
      `;
    }

    function renderStatus(ok, message) {
      document.getElementById("statusDot").classList.toggle("bad", !ok);
      document.getElementById("statusText").textContent = message;
    }

    function renderFileTimes() {
      const f = app.data.files;
      document.getElementById("fileTimes").textContent = `mlb ${fmtTime(f.paper_trades.mtime_utc) || "missing"} | apy ${fmtTime(f.known_outcome_pnl.mtime_utc) || "missing"} | signals ${fmtTime(f.model_signals.mtime_utc) || "missing"} | calibration ${fmtTime(f.probability_calibration.mtime_utc) || "missing"}`;
      document.getElementById("subtitle").textContent = `DB ${f.database.path}`;
    }

    async function refresh() {
      try {
        const response = await fetch("/api/state", { cache: "no-store" });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        app.data = await response.json();
        renderStatus(true, "Live");
        document.getElementById("lastUpdated").textContent = `Updated ${fmtTime(app.data.generated_at_utc)}`;
        renderMetrics();
        renderBreakdown();
        renderSettledPnl();
        renderKnownOutcomePnl();
        renderCalibration();
        renderFileTimes();
        renderTable();
      } catch (error) {
        renderStatus(false, "Error");
        document.getElementById("lastUpdated").textContent = String(error.message || error);
      }
    }

    function setup() {
      document.querySelectorAll(".tab").forEach(button => {
        button.addEventListener("click", () => {
          app.tab = button.dataset.tab;
          app.selectedKey = null;
          document.querySelectorAll(".tab").forEach(item => item.classList.toggle("active", item === button));
          renderDetails(null);
          renderTable();
        });
      });
      ["searchInput", "sideFilter", "statFilter", "statusFilter", "sortSelect"].forEach(id => {
        document.getElementById(id).addEventListener("input", renderTable);
        document.getElementById(id).addEventListener("change", renderTable);
      });
      document.getElementById("refreshButton").addEventListener("click", refresh);
      app.timer = setInterval(() => {
        if (document.getElementById("autoRefresh").checked) refresh();
      }, Math.max(1, refreshSeconds) * 1000);
      refresh();
    }
    setup();
  </script>
</body>
</html>
"""


class DashboardHTTPServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], config: DashboardConfig) -> None:
        super().__init__(server_address, handler)
        self.config = config


class DashboardHandler(BaseHTTPRequestHandler):
    server: DashboardHTTPServer

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict[str, Any]) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=True).encode("utf-8"), "application/json; charset=utf-8")

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json({"error": f"{path.name} not found"})
            return
        self._send_bytes(path.read_bytes(), content_type)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/index.html"}:
            html = INDEX_HTML.replace("__REFRESH_SECONDS__", str(max(1, int(self.server.config.refresh_seconds))))
            self._send_bytes(html.encode("utf-8"), "text/html; charset=utf-8")
            return
        if parsed.path == "/api/state":
            self._send_json(load_dashboard_state(self.server.config))
            return
        if parsed.path == "/healthz":
            self._send_json({"ok": True, "generated_at_utc": _utc_now()})
            return
        if parsed.path == "/paper_trades.csv":
            self._send_file(self.server.config.output_dir / "paper_trades.csv", "text/csv; charset=utf-8")
            return
        if parsed.path == "/model_signals.csv":
            self._send_file(self.server.config.output_dir / "model_signals.csv", "text/csv; charset=utf-8")
            return
        if parsed.path == "/known_outcome_trades.csv":
            self._send_file(self.server.config.output_dir / "known_outcome_trades.csv", "text/csv; charset=utf-8")
            return
        if parsed.path == "/known_outcome_pnl.csv":
            self._send_file(self.server.config.output_dir / "known_outcome_pnl.csv", "text/csv; charset=utf-8")
            return
        self._send_json({"error": "not found"})


def serve_dashboard(config: DashboardConfig) -> None:
    server = DashboardHTTPServer((config.host, config.port), DashboardHandler, config)
    url_host = "localhost" if config.host in {"127.0.0.1", "0.0.0.0"} else config.host
    print(f"Dashboard: http://{url_host}:{config.port}", flush=True)
    print(f"DB: {config.db_path}", flush=True)
    print(f"Output dir: {config.output_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Dashboard stopped.", flush=True)
    finally:
        server.server_close()
