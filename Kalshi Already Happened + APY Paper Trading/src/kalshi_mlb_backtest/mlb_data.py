from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from .utils import normalize_name, normalize_team, parse_int, parse_iso_datetime, unix_seconds


MLB_BASE_URL = "https://statsapi.mlb.com/api"
HIT_EVENT_TYPES = {"single", "double", "triple", "home_run"}
DEFAULT_OUT_PROBABILITY = 0.68
DEFAULT_BOTTOM_NINTH_IF_HOME_LEADS_PROBABILITY = 0.35
DEFAULT_TEAM_RUNS_PER_GAME = 4.4
DEFAULT_TEAM_PA_PER_GAME = 38.0
DEFAULT_RUNS_PER_PA = DEFAULT_TEAM_RUNS_PER_GAME / DEFAULT_TEAM_PA_PER_GAME
MIN_CONTEXT_PA = 9
FULL_CONTEXT_PA = 36.0
MAX_CONTEXT_WEIGHT = 0.70
MIN_CONTEXT_OUT_PROBABILITY = 0.50
MAX_CONTEXT_OUT_PROBABILITY = 0.78
MAX_SCORING_OUT_PROBABILITY_ADJUSTMENT = 0.04


@dataclass(frozen=True)
class MlbGame:
    game_pk: int
    official_date: str
    game_date_utc: int
    status: str
    away_team_id: int
    away_team_name: str
    home_team_id: int
    home_team_name: str


@dataclass(frozen=True)
class Play:
    at_bat_index: int
    start_ts: int
    end_ts: int
    inning: int
    half_inning: str
    team_side: str
    outs_before: int
    outs_after: int
    away_score_after: int
    home_score_after: int
    batter_id: int
    batter_name: str
    event_type: str
    event: str
    is_plate_appearance: bool
    is_hit: bool
    is_home_run: bool


@dataclass(frozen=True)
class GameState:
    game_id: int
    timestamp: int
    inning: int | None
    half_inning: str | None
    outs: int | None
    home_score: int | None
    away_score: int | None
    current_batter_id: int | None
    current_batter_name: str | None


@dataclass(frozen=True)
class PlateAppearanceEstimate:
    mean: float
    distribution: tuple[tuple[int, float], ...]
    source: str


class MlbClient:
    def __init__(self, base_url: str = MLB_BASE_URL, timeout: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "kalshi-mlb-player-prop-backtest/1.0"})

    def get_json(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params or {}, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"MLB returned non-object JSON from {resp.url}")
        return data

    def fetch_schedule(self, start_date: str, end_date: str) -> list[MlbGame]:
        payload = self.get_json(
            "/v1/schedule",
            {"sportId": 1, "startDate": start_date, "endDate": end_date},
        )
        games: list[MlbGame] = []
        for date_payload in payload.get("dates") or []:
            for game in date_payload.get("games") or []:
                teams = game.get("teams") or {}
                away = ((teams.get("away") or {}).get("team") or {})
                home = ((teams.get("home") or {}).get("team") or {})
                game_pk = parse_int(game.get("gamePk"))
                game_ts = unix_seconds(game.get("gameDate"))
                if game_pk is None or game_ts is None:
                    continue
                games.append(
                    MlbGame(
                        game_pk=game_pk,
                        official_date=str(game.get("officialDate") or date_payload.get("date") or ""),
                        game_date_utc=game_ts,
                        status=str((game.get("status") or {}).get("detailedState") or ""),
                        away_team_id=parse_int(away.get("id")) or 0,
                        away_team_name=str(away.get("name") or ""),
                        home_team_id=parse_int(home.get("id")) or 0,
                        home_team_name=str(home.get("name") or ""),
                    )
                )
        return games

    def fetch_game_feed(self, game_pk: int) -> dict[str, Any]:
        return self.get_json(f"/v1.1/game/{game_pk}/feed/live")


class GameContext:
    def __init__(self, game: MlbGame, feed: dict[str, Any]) -> None:
        self.game = game
        self.feed = feed
        self.plays = self._parse_plays()
        self.player_names = self._collect_player_names()
        self.player_team_side = self._collect_player_team_sides()
        self.final_hits = self._collect_final_hits()
        self.final_home_runs = self._collect_final_home_runs()

    @property
    def game_pk(self) -> int:
        return self.game.game_pk

    def _parse_plays(self) -> list[Play]:
        raw_plays = (((self.feed.get("liveData") or {}).get("plays") or {}).get("allPlays") or [])
        plays: list[Play] = []
        last_outs_by_half: dict[tuple[int, str], int] = {}
        for raw in raw_plays:
            about = raw.get("about") or {}
            matchup = raw.get("matchup") or {}
            result = raw.get("result") or {}
            count = raw.get("count") or {}
            batter = matchup.get("batter") or {}
            inning = parse_int(about.get("inning"))
            half = str(about.get("halfInning") or "").lower()
            batter_id = parse_int(batter.get("id"))
            start_ts = unix_seconds(about.get("startTime"))
            end_ts = unix_seconds(about.get("endTime")) or start_ts
            if inning is None or not half or batter_id is None or start_ts is None or end_ts is None:
                continue
            half_key = (inning, half)
            outs_before = last_outs_by_half.get(half_key, 0)
            outs_after = parse_int(count.get("outs"))
            if outs_after is None:
                outs_after = outs_before
            event_type = str(result.get("eventType") or "")
            is_pa = str(result.get("type") or "").lower() == "atbat"
            is_hit = event_type in HIT_EVENT_TYPES
            is_home_run = event_type == "home_run"
            team_side = "away" if half == "top" else "home"
            play = Play(
                at_bat_index=parse_int(about.get("atBatIndex")) or len(plays),
                start_ts=start_ts,
                end_ts=end_ts,
                inning=inning,
                half_inning=half,
                team_side=team_side,
                outs_before=outs_before,
                outs_after=outs_after,
                away_score_after=parse_int(result.get("awayScore")) or 0,
                home_score_after=parse_int(result.get("homeScore")) or 0,
                batter_id=batter_id,
                batter_name=str(batter.get("fullName") or ""),
                event_type=event_type,
                event=str(result.get("event") or ""),
                is_plate_appearance=is_pa,
                is_hit=is_hit,
                is_home_run=is_home_run,
            )
            plays.append(play)
            last_outs_by_half[half_key] = outs_after
        return sorted(plays, key=lambda item: (item.start_ts, item.at_bat_index))

    def _collect_player_names(self) -> dict[int, str]:
        names: dict[int, str] = {}
        boxscore = ((self.feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("away", "home"):
            for raw_player in ((boxscore.get(side) or {}).get("players") or {}).values():
                person = raw_player.get("person") or {}
                player_id = parse_int(person.get("id"))
                if player_id is not None:
                    names[player_id] = str(person.get("fullName") or names.get(player_id) or "")
        for play in self.plays:
            names.setdefault(play.batter_id, play.batter_name)
        return names

    def _collect_player_team_sides(self) -> dict[int, str]:
        sides: dict[int, str] = {}
        for play in self.plays:
            sides.setdefault(play.batter_id, play.team_side)
        boxscore = ((self.feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("away", "home"):
            for raw_player in ((boxscore.get(side) or {}).get("players") or {}).values():
                person = raw_player.get("person") or {}
                player_id = parse_int(person.get("id"))
                if player_id is not None:
                    sides.setdefault(player_id, side)
        return sides

    def _collect_final_hits(self) -> dict[int, int]:
        hits: dict[int, int] = {}
        boxscore = ((self.feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("away", "home"):
            for raw_player in ((boxscore.get(side) or {}).get("players") or {}).values():
                person = raw_player.get("person") or {}
                player_id = parse_int(person.get("id"))
                batting = (raw_player.get("stats") or {}).get("batting") or {}
                if player_id is not None:
                    hits[player_id] = parse_int(batting.get("hits")) or 0
        return hits

    def _collect_final_home_runs(self) -> dict[int, int]:
        home_runs: dict[int, int] = {}
        boxscore = ((self.feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        for side in ("away", "home"):
            for raw_player in ((boxscore.get(side) or {}).get("players") or {}).values():
                person = raw_player.get("person") or {}
                player_id = parse_int(person.get("id"))
                batting = (raw_player.get("stats") or {}).get("batting") or {}
                if player_id is not None:
                    home_runs[player_id] = parse_int(batting.get("homeRuns")) or 0
        return home_runs

    def match_player_id(self, player_name: str) -> int | None:
        target = normalize_name(player_name)
        if not target:
            return None
        exact = [pid for pid, name in self.player_names.items() if normalize_name(name) == target]
        if len(exact) == 1:
            return exact[0]
        compact_target = target.replace(" ", "")
        compact = [
            pid
            for pid, name in self.player_names.items()
            if normalize_name(name).replace(" ", "") == compact_target
        ]
        if len(compact) == 1:
            return compact[0]
        return exact[0] if exact else None

    def state_at(self, timestamp: int) -> GameState:
        completed = [play for play in self.plays if play.end_ts <= timestamp]
        current = next((play for play in self.plays if play.start_ts <= timestamp < play.end_ts), None)
        next_play = next((play for play in self.plays if play.start_ts > timestamp), None)
        score_source = completed[-1] if completed else None
        away_score = score_source.away_score_after if score_source else 0
        home_score = score_source.home_score_after if score_source else 0
        inning_source = current or (next_play if completed else None) or (completed[-1] if completed else None)
        outs = None
        if current:
            outs = current.outs_before
        elif next_play and completed:
            outs = next_play.outs_before
        elif completed:
            outs = completed[-1].outs_after
        return GameState(
            game_id=self.game.game_pk,
            timestamp=timestamp,
            inning=inning_source.inning if inning_source else None,
            half_inning=inning_source.half_inning if inning_source else None,
            outs=outs,
            home_score=home_score,
            away_score=away_score,
            current_batter_id=current.batter_id if current else None,
            current_batter_name=current.batter_name if current else None,
        )

    def player_stats_at(self, player_id: int, timestamp: int) -> tuple[int, int]:
        hits = 0
        pa = 0
        for play in self.plays:
            if play.end_ts > timestamp:
                continue
            if play.batter_id != player_id or not play.is_plate_appearance:
                continue
            pa += 1
            if play.is_hit:
                hits += 1
        return hits, pa

    def player_home_runs_at(self, player_id: int, timestamp: int) -> int:
        home_runs = 0
        for play in self.plays:
            if play.end_ts > timestamp:
                continue
            if play.batter_id != player_id or not play.is_plate_appearance:
                continue
            if play.is_home_run:
                home_runs += 1
        return home_runs

    def player_batting_proximity(
        self,
        player_id: int,
        timestamp: int,
        state: GameState | None = None,
    ) -> str | None:
        state = state or self.state_at(timestamp)
        active_side = "away" if state.half_inning == "top" else "home" if state.half_inning == "bottom" else None
        player_side = self.player_team_side.get(player_id)
        if active_side is None or player_side != active_side:
            return None
        order = self._observed_batting_order(active_side, timestamp)
        if len(order) < 9 or player_id not in order[:9]:
            return None
        lineup = order[:9]
        current = next((play for play in self.plays if play.start_ts <= timestamp < play.end_ts), None)
        started_pas = [
            play
            for play in self.plays
            if play.team_side == active_side and play.is_plate_appearance and play.start_ts <= timestamp
        ]
        next_slot = len(started_pas) % 9
        if current and current.team_side == active_side and current.is_plate_appearance:
            if current.batter_id == player_id:
                return "at_bat"
            return "on_deck" if lineup[next_slot] == player_id else None
        if lineup[next_slot] == player_id:
            return "due_up"
        return "on_deck" if lineup[(next_slot + 1) % 9] == player_id else None

    def estimate_pa_remaining(self, player_id: int, timestamp: int, state: GameState | None = None) -> int | None:
        state = state or self.state_at(timestamp)
        if state.inning not in {8, 9} or not state.half_inning:
            return None
        side = self.player_team_side.get(player_id)
        if side not in {"away", "home"}:
            return None
        order = self._observed_batting_order(side, timestamp)
        if len(order) < 9 or player_id not in order[:9]:
            return None
        lineup = order[:9]
        player_slot = lineup.index(player_id)
        started_pas = [
            play
            for play in self.plays
            if play.team_side == side and play.is_plate_appearance and play.start_ts <= timestamp
        ]
        next_slot = len(started_pas) % 9
        slots_remaining = self._minimum_team_pa_slots_remaining(side, state)
        if slots_remaining is None:
            return None
        estimate = 0
        current = next((play for play in self.plays if play.start_ts <= timestamp < play.end_ts), None)
        if current and current.batter_id == player_id:
            estimate += 1
        for offset in range(slots_remaining):
            if (next_slot + offset) % 9 == player_slot:
                estimate += 1
        return estimate

    def model_pa_remaining(self, player_id: int, timestamp: int, state: GameState | None = None) -> PlateAppearanceEstimate | None:
        state = state or self.state_at(timestamp)
        if state.inning is None or not state.half_inning or state.outs is None:
            return None
        if state.inning > 9:
            return None
        side = self.player_team_side.get(player_id)
        if side not in {"away", "home"}:
            return None
        order = self._observed_batting_order(side, timestamp)
        if len(order) < 9 or player_id not in order[:9]:
            return None
        lineup = order[:9]
        player_slot = lineup.index(player_id)
        started_pas = [
            play
            for play in self.plays
            if play.team_side == side and play.is_plate_appearance and play.start_ts <= timestamp
        ]
        next_slot = len(started_pas) % 9
        out_probability, pace_source = self._contextual_out_probability(side, state)
        team_slots_mean = self._expected_team_pa_slots_remaining(side, state, out_probability)
        if team_slots_mean is None:
            return None
        current = next((play for play in self.plays if play.start_ts <= timestamp < play.end_ts), None)
        current_player_pa = 1 if current and current.batter_id == player_id else 0
        if current and current.team_side == side:
            team_slots_mean = max(0.0, team_slots_mean - 1.0)
        distribution = self._player_pa_distribution(player_slot, next_slot, team_slots_mean, current_player_pa)
        mean = sum(pa * probability for pa, probability in distribution)
        return PlateAppearanceEstimate(
            mean=mean,
            distribution=distribution,
            source=f"lineup_slots_{pace_source}_out_prob={out_probability:.3f}",
        )

    def _observed_batting_order(self, side: str, timestamp: int) -> list[int]:
        order: list[int] = []
        for play in self.plays:
            if play.team_side != side or not play.is_plate_appearance or play.start_ts > timestamp:
                continue
            if play.batter_id not in order:
                order.append(play.batter_id)
            if len(order) >= 9:
                break
        if len(order) < 9:
            for player_id in self._boxscore_batting_order(side):
                if player_id not in order:
                    order.append(player_id)
                if len(order) >= 9:
                    break
        return order

    def _boxscore_batting_order(self, side: str) -> list[int]:
        boxscore = ((self.feed.get("liveData") or {}).get("boxscore") or {}).get("teams") or {}
        players = ((boxscore.get(side) or {}).get("players") or {}).values()
        slots: list[tuple[int, int]] = []
        for raw_player in players:
            person = raw_player.get("person") or {}
            player_id = parse_int(person.get("id"))
            batting_order = parse_int(raw_player.get("battingOrder"))
            if player_id is not None and batting_order is not None and batting_order > 0:
                slots.append((batting_order, player_id))
        return [player_id for _, player_id in sorted(slots)[:9]]

    def _minimum_team_pa_slots_remaining(self, side: str, state: GameState) -> int | None:
        inning = state.inning
        half = state.half_inning
        outs = state.outs
        if inning not in {8, 9} or half not in {"top", "bottom"} or outs is None:
            return None
        current_slots = max(0, 3 - outs)
        if side == "away":
            if inning == 8:
                return (current_slots if half == "top" else 0) + 3
            if inning == 9:
                return current_slots if half == "top" else 0
        if side == "home":
            if inning == 8:
                bottom8 = 3 if half == "top" else current_slots
                bottom9 = 3 if (state.home_score is not None and state.away_score is not None and state.home_score <= state.away_score) else 0
                return bottom8 + bottom9
            if inning == 9:
                if half == "top":
                    return 3 if (state.home_score is not None and state.away_score is not None and state.home_score <= state.away_score) else 0
                return current_slots
        return None

    def _expected_batters_for_outs(
        self,
        outs_remaining: int,
        out_probability: float = DEFAULT_OUT_PROBABILITY,
    ) -> float:
        outs = max(0, min(3, int(outs_remaining)))
        probability = min(MAX_CONTEXT_OUT_PROBABILITY, max(MIN_CONTEXT_OUT_PROBABILITY, float(out_probability)))
        return outs / probability

    def _expected_full_half_inning_batters(self, out_probability: float = DEFAULT_OUT_PROBABILITY) -> float:
        return self._expected_batters_for_outs(3, out_probability)

    def _expected_team_pa_slots_remaining(
        self,
        side: str,
        state: GameState,
        out_probability: float = DEFAULT_OUT_PROBABILITY,
    ) -> float | None:
        inning = state.inning
        half = state.half_inning
        outs = state.outs
        if inning is None or inning > 9 or half not in {"top", "bottom"} or outs is None:
            return None
        current_half = self._expected_batters_for_outs(3 - outs, out_probability)
        full_half = self._expected_full_half_inning_batters(out_probability)
        away_score = state.away_score or 0
        home_score = state.home_score or 0

        if side == "away":
            if half == "top":
                future_full_tops = max(0, 9 - inning)
                return current_half + (future_full_tops * full_half)
            future_full_tops = max(0, 9 - inning)
            return future_full_tops * full_half

        if side == "home":
            total = 0.0
            if half == "bottom":
                total += current_half
                next_bottom = inning + 1
            else:
                next_bottom = inning
            for bottom_inning in range(next_bottom, 10):
                if bottom_inning < 9:
                    total += full_half
                elif bottom_inning == 9:
                    if inning == 9 and half == "bottom":
                        continue
                    if home_score <= away_score:
                        total += full_half
                    else:
                        total += DEFAULT_BOTTOM_NINTH_IF_HOME_LEADS_PROBABILITY * full_half
            return total
        return None

    def _completed_team_pa_outs(self, side: str, timestamp: int) -> tuple[int, int]:
        completed = [
            play
            for play in self.plays
            if play.team_side == side and play.is_plate_appearance and play.end_ts <= timestamp
        ]
        outs = sum(max(0, play.outs_after - play.outs_before) for play in completed)
        return len(completed), outs

    def _contextual_out_probability(self, side: str, state: GameState) -> tuple[float, str]:
        team_pa, team_outs = self._completed_team_pa_outs(side, state.timestamp)
        if team_pa < MIN_CONTEXT_PA or team_outs <= 0:
            return DEFAULT_OUT_PROBABILITY, "mlb_average_pa_pace"

        observed_out_probability = team_outs / team_pa
        team_runs = state.away_score if side == "away" else state.home_score
        runs_per_pa = (float(team_runs or 0) / team_pa) if team_pa else 0.0
        scoring_uplift = max(0.0, (runs_per_pa - DEFAULT_RUNS_PER_PA) / DEFAULT_RUNS_PER_PA)
        scoring_adjustment = min(MAX_SCORING_OUT_PROBABILITY_ADJUSTMENT, scoring_uplift * 0.02)
        context_out_probability = observed_out_probability - scoring_adjustment
        sample_weight = min(
            MAX_CONTEXT_WEIGHT,
            max(0.0, (team_pa - MIN_CONTEXT_PA) / (FULL_CONTEXT_PA - MIN_CONTEXT_PA)) * MAX_CONTEXT_WEIGHT,
        )
        blended = (DEFAULT_OUT_PROBABILITY * (1.0 - sample_weight)) + (context_out_probability * sample_weight)
        clipped = min(MAX_CONTEXT_OUT_PROBABILITY, max(MIN_CONTEXT_OUT_PROBABILITY, blended))
        return clipped, f"contextual_pa_pace_weight={sample_weight:.2f}"

    def _player_pa_distribution(
        self,
        player_slot: int,
        next_slot: int,
        team_slots_mean: float,
        current_player_pa: int,
    ) -> tuple[tuple[int, float], ...]:
        mean = max(0.0, team_slots_mean)
        low = int(mean)
        high = low if abs(mean - low) < 1e-9 else low + 1
        high_weight = mean - low

        def player_pas_for_slots(slots: int) -> int:
            return current_player_pa + sum(
                1 for offset in range(max(0, slots)) if (next_slot + offset) % 9 == player_slot
            )

        if high == low:
            return ((player_pas_for_slots(low), 1.0),)
        buckets: dict[int, float] = {}
        buckets[player_pas_for_slots(low)] = buckets.get(player_pas_for_slots(low), 0.0) + (1.0 - high_weight)
        buckets[player_pas_for_slots(high)] = buckets.get(player_pas_for_slots(high), 0.0) + high_weight
        return tuple(sorted((pa, probability) for pa, probability in buckets.items() if probability > 0))

    def late_window(self) -> tuple[int, int] | None:
        late_plays = [play for play in self.plays if play.inning in {8, 9}]
        if not late_plays:
            return None
        return min(play.start_ts for play in late_plays), max(play.end_ts for play in late_plays)

    def game_window(self) -> tuple[int, int] | None:
        if not self.plays:
            return None
        return min(play.start_ts for play in self.plays), max(play.end_ts for play in self.plays)

    def game_team_match_score(self, away_name: str | None, home_name: str | None, start_ts: int | None) -> int:
        score = 0
        if start_ts is not None:
            diff = abs(self.game.game_date_utc - start_ts)
            if diff <= 15 * 60:
                score += 4
            elif diff <= 90 * 60:
                score += 2
        if away_name and normalize_team(away_name) == normalize_team(self.game.away_team_name):
            score += 3
        if home_name and normalize_team(home_name) == normalize_team(self.game.home_team_name):
            score += 3
        return score
