"""NBA Stats ingestion adapter.

Wraps ``nba_api`` endpoints with bounded retries and rate limiting per
SRD Section 3.2 (0.5s-2s spacing). ``nba_api`` is imported lazily so the
module — and the entire test suite — works with zero live network calls:
tests inject a fake ``endpoint_caller`` that returns fixture payloads in
the same shape as ``nba_api``'s ``get_normalized_dict()``.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

from src.ingestion.db import ProphitDB

EndpointCaller = Callable[[str, dict], dict]


def _default_endpoint_caller(endpoint_name: str, params: dict) -> dict:
    """Call a real nba_api endpoint by name and return normalized dict output.

    Imported lazily so tests never need the network (or even the package).
    """
    import importlib

    module = importlib.import_module("nba_api.stats.endpoints")
    endpoint_cls = getattr(module, endpoint_name)
    return endpoint_cls(**params).get_normalized_dict()


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    return str(value)


def parse_minutes(value: Any) -> Optional[float]:
    """nba_api returns MIN either as 'MM:SS' strings or plain numbers."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = re.fullmatch(r"(\d+):(\d+(?:\.\d+)?)", text)
    if match:
        return int(match.group(1)) + float(match.group(2)) / 60.0
    try:
        return float(text)
    except ValueError:
        return None


def parse_matchup(matchup: Optional[str]) -> tuple[Optional[str], Optional[bool]]:
    """'LAL vs. BOS' -> ('BOS', True); 'LAL @ BOS' -> ('BOS', False)."""
    if not matchup:
        return None, None
    parts = re.split(r"\s+(?:vs\.?|@)\s+", matchup.strip())
    if len(parts) != 2:
        return None, None
    return parts[1].strip(), "vs" in matchup.lower()


def parse_game_date(value: Optional[str]) -> Optional[str]:
    """Normalize nba_api date strings ('OCT 22, 2025' or ISO) to YYYY-MM-DD."""
    if not value:
        return None
    text = str(value).strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    for fmt in ("%b %d, %Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"):
        try:
            return time.strftime("%Y-%m-%d", time.strptime(text, fmt))
        except ValueError:
            continue
    return text


class NBAStatsClient:
    """Rate-limited, retrying wrapper around nba_api endpoints."""

    def __init__(
        self,
        sleep_seconds: float = 0.6,
        max_retries: int = 3,
        timeout: int = 30,
        endpoint_caller: Optional[EndpointCaller] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.sleep_seconds = sleep_seconds
        self.max_retries = max_retries
        self.timeout = timeout
        self._caller = endpoint_caller or _default_endpoint_caller
        self._sleep = sleep_fn

    def _call(self, endpoint_name: str, **params: Any) -> dict:
        params.setdefault("timeout", self.timeout)
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                result = self._caller(endpoint_name, params)
                self._sleep(self.sleep_seconds)
                return result
            except Exception as exc:  # nba_api raises requests exceptions
                last_error = exc
                self._sleep(self.sleep_seconds * (2**attempt))
        raise RuntimeError(
            f"nba_api endpoint {endpoint_name} failed after "
            f"{self.max_retries} attempts"
        ) from last_error

    def get_player_game_log(
        self, player_id: int, season: str, season_type: str = "Regular Season"
    ) -> list[dict]:
        data = self._call(
            "PlayerGameLog",
            player_id=player_id,
            season=season,
            season_type_all_star=season_type,
        )
        return data.get("PlayerGameLog", [])

    def get_league_game_log(
        self, season: str, season_type: str = "Regular Season"
    ) -> list[dict]:
        data = self._call(
            "LeagueGameLog",
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation="P",
        )
        return data.get("LeagueGameLog", [])

    def get_league_dash_team_defense(
        self, season: str, defense_category: str = "Overall"
    ) -> list[dict]:
        data = self._call(
            "LeagueDashPtTeamDefend",
            season=season,
            defense_category=defense_category,
            per_mode_simple="PerGame",
        )
        return data.get("LeagueDashPtTeamDefend", [])


class PlayerGameLogIngestion:
    """Parses normalized game-log rows and persists them to SQLite."""

    def __init__(self, db: ProphitDB, client: Optional[NBAStatsClient] = None):
        self.db = db
        self.client = client

    def ingest_player_season(
        self, player_id: int, player_name: str, season: str
    ) -> int:
        """Fetch (via client) and store one player's season game log."""
        if self.client is None:
            raise ValueError("ingest_player_season requires an NBAStatsClient")
        rows = self.client.get_player_game_log(player_id, season)
        return self.ingest_rows(rows, default_player_name=player_name)

    def ingest_rows(
        self, rows: list[dict], default_player_name: Optional[str] = None
    ) -> int:
        """Persist normalized PlayerGameLog rows. Pure & offline-safe."""
        written = 0
        for raw in rows:
            game = self._parse_row(raw, default_player_name)
            if game is None:
                continue
            self.db.upsert_player_game(game)
            written += 1
        self.db.log_ingestion("nba_stats.player_game_log", written)
        return written

    def _parse_row(
        self, raw: dict, default_player_name: Optional[str]
    ) -> Optional[dict]:
        game_id = _to_str(raw.get("Game_ID") or raw.get("GAME_ID"))
        game_date = parse_game_date(_to_str(raw.get("GAME_DATE")))
        if not game_id or not game_date:
            return None

        player_id = _to_int(raw.get("Player_ID") or raw.get("PLAYER_ID"))
        player_name = _to_str(raw.get("PLAYER_NAME")) or default_player_name
        team_abbr = _to_str(raw.get("TEAM_ABBREVIATION"))
        team_id = _to_int(raw.get("TEAM_ID"))

        if team_abbr:
            resolved_team_id = team_id or self.db.team_id_for_abbr(team_abbr)
            self.db.upsert_team(resolved_team_id, team_abbr)
            team_id = resolved_team_id

        if player_id is not None and player_name:
            self.db.upsert_player(player_id, player_name, team_id=team_id)
        elif player_id is None and player_name:
            player_id = self.db.get_or_create_player_id(player_name)
        if player_id is None:
            return None

        opp_abbr, is_home = parse_matchup(_to_str(raw.get("MATCHUP")))
        opponent_team_id = self.db.team_id_for_abbr(opp_abbr) if opp_abbr else None

        return {
            "game_id": game_id,
            "player_id": player_id,
            "team_id": team_id,
            "opponent_team_id": opponent_team_id,
            "season_id": _to_str(raw.get("SEASON_ID")),
            "game_date": game_date,
            "matchup": _to_str(raw.get("MATCHUP")),
            "is_home": None if is_home is None else int(is_home),
            "wl": _to_str(raw.get("WL")),
            "minutes": parse_minutes(raw.get("MIN")),
            "pts": _to_int(raw.get("PTS")),
            "fgm": _to_int(raw.get("FGM")),
            "fga": _to_int(raw.get("FGA")),
            "fg_pct": _to_float(raw.get("FG_PCT")),
            "fg3m": _to_int(raw.get("FG3M")),
            "fg3a": _to_int(raw.get("FG3A")),
            "fg3_pct": _to_float(raw.get("FG3_PCT")),
            "ftm": _to_int(raw.get("FTM")),
            "fta": _to_int(raw.get("FTA")),
            "ft_pct": _to_float(raw.get("FT_PCT")),
            "oreb": _to_int(raw.get("OREB")),
            "dreb": _to_int(raw.get("DREB")),
            "reb": _to_int(raw.get("REB")),
            "ast": _to_int(raw.get("AST")),
            "stl": _to_int(raw.get("STL")),
            "blk": _to_int(raw.get("BLK")),
            "tov": _to_int(raw.get("TOV")),
            "pf": _to_int(raw.get("PF")),
            "plus_minus": _to_float(raw.get("PLUS_MINUS")),
        }


class TeamDefenseIngestion:
    """Persists LeagueDashPtTeamDefend snapshots into fact_team_defense_daily."""

    def __init__(self, db: ProphitDB, client: Optional[NBAStatsClient] = None):
        self.db = db
        self.client = client

    def ingest_rows(self, rows: list[dict], snapshot_date: str) -> int:
        written = 0
        for raw in rows:
            team_id = _to_int(raw.get("TEAM_ID"))
            team_abbr = _to_str(raw.get("TEAM_ABBREVIATION"))
            if team_abbr:
                team_id = team_id or self.db.team_id_for_abbr(team_abbr)
                self.db.upsert_team(team_id, team_abbr)
            if team_id is None:
                continue
            self.db.upsert_team_defense(
                {
                    "team_id": team_id,
                    "snapshot_date": snapshot_date,
                    "def_rating": _to_float(raw.get("DEF_RATING")),
                    "def_rating_rank": _to_int(raw.get("DEF_RATING_RANK")),
                    "opp_pts": _to_float(raw.get("OPP_PTS")),
                    "opp_reb": _to_float(raw.get("OPP_REB")),
                    "opp_ast": _to_float(raw.get("OPP_AST")),
                    "opp_fg3m": _to_float(raw.get("OPP_FG3M")),
                    "position_scope": _to_str(raw.get("DEFENSE_CATEGORY")) or "ALL",
                }
            )
            written += 1
        self.db.log_ingestion("nba_stats.team_defense", written)
        return written
