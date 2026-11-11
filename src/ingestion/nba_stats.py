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

from src.ingestion.consensus_odds import _RetryingClient
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


def nba_date_param(iso_date: Optional[str]) -> Optional[str]:
    """'2025-10-22' -> '10/22/2025' for nba_api's ``date_from_nullable``."""
    if not iso_date:
        return None
    year, month, day = str(iso_date)[:10].split("-")
    return f"{month}/{day}/{year}"


def _season_label(iso_date: str) -> str:
    """'2025-10-23' -> '2025-26' (NBA season runs Oct-Jun)."""
    year, month = int(iso_date[:4]), int(iso_date[5:7])
    start = year if month >= 7 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _split_made_attempted(value: Any) -> tuple[Optional[int], Optional[int]]:
    """ESPN 'M-A' strings: '11-18' -> (11, 18); '--' -> (None, None)."""
    if value is None:
        return None, None
    match = re.fullmatch(r"(\d+)\s*-\s*(\d+)", str(value).strip())
    if match:
        return int(match.group(1)), int(match.group(2))
    return _to_int(value), None


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
        self,
        season: str,
        season_type: str = "Regular Season",
        date_from: Optional[str] = None,
    ) -> list[dict]:
        params = {
            "season": season,
            "season_type_all_star": season_type,
            "player_or_team_abbreviation": "P",
        }
        if date_from:
            params["date_from_nullable"] = date_from
        data = self._call("LeagueGameLog", **params)
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


ESPN_NBA_BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"


class ESPNBoxScoreClient(_RetryingClient):
    """Zero-auth fallback for completed-game box scores via ESPN's public
    JSON endpoints — no key, no headers, safe when stats.nba.com throttles.

    Rows are normalized into the exact dict shape accepted by
    ``PlayerGameLogIngestion.ingest_rows()``. ESPN's own numeric team /
    athlete ids are intentionally dropped: ``TEAM_ABBREVIATION`` +
    ``PLAYER_NAME`` resolve through the same ``team_id_for_abbr`` /
    ``get_or_create_player_id`` path used by the odds feeds, so ESPN ids
    never pollute the dim tables.
    """

    def __init__(self, **kwargs: Any):
        super().__init__(ESPN_NBA_BASE_URL, **kwargs)

    def get_scoreboard(self, date_yyyymmdd: str) -> dict:
        return self._get_json("/scoreboard", params={"dates": date_yyyymmdd})

    def get_summary(self, event_id: str) -> dict:
        return self._get_json("/summary", params={"event": event_id})

    def fetch_missing_dates(self, dates: list) -> list[dict]:
        """Normalized game-log rows for each ISO 'YYYY-MM-DD' date given."""
        rows: list[dict] = []
        for iso_date in dates:
            for event in self._completed_events(str(iso_date)[:10]):
                rows.extend(self._event_rows(event, str(iso_date)[:10]))
        return rows

    def _completed_events(self, iso_date: str) -> list:
        payload = self.get_scoreboard(iso_date.replace("-", ""))
        events = []
        for event in payload.get("events") or []:
            status = (event.get("status") or {}).get("type") or {}
            if status.get("completed") or status.get("state") == "post":
                events.append(event)
        return events

    def _event_rows(self, event: dict, game_date: str) -> list:
        event_id = str(event.get("id") or "")
        competitions = event.get("competitions") or []
        competitors = (competitions[0] if competitions else {}).get("competitors") or []
        sides = {c.get("homeAway"): c for c in competitors}
        home, away = sides.get("home"), sides.get("away")
        if not event_id or not home or not away:
            return []
        home_abbr = (home.get("team") or {}).get("abbreviation")
        away_abbr = (away.get("team") or {}).get("abbreviation")
        if not home_abbr or not away_abbr:
            return []
        winners = self._winning_abbrs(competitors)

        rows = []
        summary = self.get_summary(event_id)
        team_blocks = (summary.get("boxscore") or {}).get("players") or []
        for block in team_blocks:
            team_abbr = (block.get("team") or {}).get("abbreviation")
            # parse_matchup reads "AWY @ HOME" -> away player, or
            # "HOME vs. AWY" -> home player. Format per team block so
            # home players get is_home=1 and the right opponent_team_id.
            team_matchup = (
                f"{home_abbr} vs. {away_abbr}"
                if team_abbr == home_abbr
                else f"{away_abbr} @ {home_abbr}"
            )
            for group in block.get("statistics") or []:
                keys = group.get("keys") or []
                for entry in group.get("athletes") or []:
                    row = self._athlete_row(
                        entry,
                        keys,
                        team_abbr,
                        event_id,
                        game_date,
                        team_matchup,
                        winners,
                    )
                    if row:
                        rows.append(row)
        return rows

    @staticmethod
    def _winning_abbrs(competitors: list) -> set:
        winners = {
            (c.get("team") or {}).get("abbreviation")
            for c in competitors
            if c.get("winner")
        }
        winners.discard(None)
        if not winners:
            scores = {}
            for c in competitors:
                abbr = (c.get("team") or {}).get("abbreviation")
                try:
                    scores[abbr] = float(c.get("score") or 0)
                except (TypeError, ValueError):
                    continue
            if scores:
                top = max(scores.values())
                winners = {a for a, s in scores.items() if s == top}
        return winners

    @staticmethod
    def _athlete_row(
        entry: dict,
        keys: list,
        team_abbr: Optional[str],
        event_id: str,
        game_date: str,
        matchup: str,
        winners: set,
    ) -> Optional[dict]:
        if entry.get("didNotPlay") or not team_abbr:
            return None
        name = (entry.get("athlete") or {}).get("displayName")
        stats = {str(k).lower(): v for k, v in zip(keys, entry.get("stats") or [])}
        minutes = stats.get("min")
        if not name or minutes in (None, "", "--"):
            return None
        fgm, fga = _split_made_attempted(stats.get("fg"))
        fg3m, fg3a = _split_made_attempted(stats.get("3pt"))
        ftm, fta = _split_made_attempted(stats.get("ft"))
        return {
            "Game_ID": f"espn-{event_id}",
            "GAME_DATE": game_date,
            "PLAYER_NAME": name,
            "TEAM_ABBREVIATION": team_abbr,
            "MATCHUP": matchup,
            "WL": "W" if team_abbr in winners else ("L" if winners else None),
            "MIN": minutes,
            "PTS": _to_int(stats.get("pts")),
            "FGM": fgm,
            "FGA": fga,
            "FG_PCT": (fgm / fga) if fgm is not None and fga else None,
            "FG3M": fg3m,
            "FG3A": fg3a,
            "FG3_PCT": (fg3m / fg3a) if fg3m is not None and fg3a else None,
            "FTM": ftm,
            "FTA": fta,
            "FT_PCT": (ftm / fta) if ftm is not None and fta else None,
            "OREB": _to_int(stats.get("oreb")),
            "DREB": _to_int(stats.get("dreb")),
            "REB": _to_int(stats.get("reb")),
            "AST": _to_int(stats.get("ast")),
            "STL": _to_int(stats.get("stl")),
            "BLK": _to_int(stats.get("blk")),
            "TOV": _to_int(stats.get("to") or stats.get("tov")),
            "PF": _to_int(stats.get("pf")),
            "PLUS_MINUS": _to_float(stats.get("+/-")),
        }


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

    def latest_game_date(self, season_prefix: Optional[str] = None) -> Optional[str]:
        """Incremental watermark: most recent stored game_date (ISO).

        ``season_prefix`` accepts either an ISO lower bound ('2025-10-01')
        or a season label ('2025-26' -> games on/after Aug 1 of the start
        year). Pass the result through ``nba_date_param`` to produce the
        ``MM/DD/YYYY`` string nba_api's ``date_from_nullable`` expects.
        """
        sql = "SELECT MAX(game_date) AS d FROM fact_player_games"
        params: tuple = ()
        if season_prefix:
            bound = (
                f"{season_prefix[:4]}-08-01"
                if re.fullmatch(r"\d{4}-\d{2}$", season_prefix)
                else str(season_prefix)[:10]
            )
            sql += " WHERE game_date >= ?"
            params = (bound,)
        row = self.db.query_one(sql, params)
        return row["d"] if row and row["d"] else None

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

    # --- local DvP derivation (zero API calls) ------------------------------

    _ALLOWED_SUMS = (
        "SUM(COALESCE(g.pts,0)) pts, SUM(COALESCE(g.reb,0)) reb, "
        "SUM(COALESCE(g.ast,0)) ast, SUM(COALESCE(g.fg3m,0)) fg3m, "
        "SUM(COALESCE(g.fgm,0)) fgm, SUM(COALESCE(g.fga,0)) fga, "
        "SUM(COALESCE(g.fg3a,0)) fg3a, SUM(COALESCE(g.fta,0)) fta, "
        "SUM(COALESCE(g.oreb,0)) oreb, SUM(COALESCE(g.tov,0)) tov"
    )

    def _allowed_rows(
        self,
        snapshot_date: str,
        season_start: Optional[str],
        group_by_position: bool = False,
    ) -> list:
        """Per-(defense, game_id) totals allowed, strictly on/before the
        snapshot date. ``group_by_position`` splits by dim_players.position
        for position-scoped DvP."""
        params: list = [str(snapshot_date)[:10]]
        bound_sql = ""
        if season_start:
            bound_sql = " AND g.game_date >= ?"
            params.append(str(season_start)[:10])
        if group_by_position:
            return self.db.query(
                f"SELECT g.opponent_team_id AS team_id, p.position AS position,"
                f" g.game_id, {self._ALLOWED_SUMS}"
                " FROM fact_player_games g"
                " JOIN dim_players p ON p.player_id = g.player_id"
                " WHERE g.opponent_team_id IS NOT NULL"
                " AND p.position IS NOT NULL AND p.position != ''"
                " AND g.game_date <= ?"
                f"{bound_sql}"
                " GROUP BY g.opponent_team_id, p.position, g.game_id",
                tuple(params),
            )
        return self.db.query(
            f"SELECT opponent_team_id AS team_id, game_id, {self._ALLOWED_SUMS}"
            " FROM fact_player_games g"
            " WHERE opponent_team_id IS NOT NULL AND game_date <= ?"
            f"{bound_sql}"
            " GROUP BY opponent_team_id, game_id",
            tuple(params),
        )

    @staticmethod
    def _scopes_for_position(position: Optional[str]) -> set:
        """'PG' -> {'PG','G'}; 'G-F' -> {'G','F'}; 'Center' -> {'C'}."""
        text = str(position or "").upper().strip()
        exact = {"PG", "SG", "SF", "PF", "C"}
        scopes = {text} & exact
        if "G" in text or "GUARD" in text:
            scopes.add("G")
        if "F" in text or "FORWARD" in text:
            scopes.add("F")
        if "C" in text or "CENTER" in text:
            scopes.add("C")
        return scopes

    @staticmethod
    def _defense_metrics(totals: dict) -> dict:
        games = totals["games"]
        poss = totals["fga"] + 0.44 * totals["fta"] - totals["oreb"] + totals["tov"]
        twos_m = totals["fgm"] - totals["fg3m"]
        twos_a = totals["fga"] - totals["fg3a"]
        return {
            "opp_pts": totals["pts"] / games,
            "opp_reb": totals["reb"] / games,
            "opp_ast": totals["ast"] / games,
            "opp_fg3m": totals["fg3m"] / games,
            "paint_pts_allowed": 2.0 * twos_m / games,
            "perimeter_fg3_pct": (
                totals["fg3m"] / totals["fg3a"] if totals["fg3a"] else None
            ),
            "rim_fg_pct_allowed": (twos_m / twos_a) if twos_a else None,
            "def_rating": (100.0 * totals["pts"] / poss) if poss > 0 else None,
        }

    @staticmethod
    def _rank_defense(metrics_by_team: dict) -> None:
        """Rank 1 = best (lowest allowed) within each metric, in place."""
        for key, rank_key in (
            ("def_rating", "def_rating_rank"),
            ("paint_pts_allowed", "paint_pts_rank"),
            ("perimeter_fg3_pct", "perimeter_fg3_rank"),
            ("rim_fg_pct_allowed", "rim_fg_pct_rank"),
        ):
            for team_id, metrics in metrics_by_team.items():
                value = metrics.get(key)
                if value is None:
                    metrics[rank_key] = None
                    continue
                metrics[rank_key] = 1 + sum(
                    1
                    for other_id, other in metrics_by_team.items()
                    if other_id != team_id
                    and other.get(key) is not None
                    and other[key] < value
                )

    def derive_from_games(
        self, snapshot_date: str, season_start: Optional[str] = None
    ) -> int:
        """Derive overall + position-scoped DvP rows from fact_player_games.

        Zero API calls — fully replaces LeagueDashPtTeamDefend during daily
        refreshes. Uses games strictly on/before ``snapshot_date`` (and
        ``>= season_start`` when given, falling back to all available games
        when the current season has fewer than 5 team-games). Returns the
        number of fact_team_defense_daily rows written.
        """
        game_date = str(snapshot_date)[:10]
        effective_start = season_start
        rows = self._allowed_rows(game_date, effective_start)
        if season_start and len({r["game_id"] for r in rows}) < 5:
            effective_start = None
            rows = self._allowed_rows(game_date, None)

        totals_by_team: dict = {}
        for r in rows:
            t = totals_by_team.setdefault(
                r["team_id"], {"games": 0, **{k: 0.0 for k in _DEFENSE_STATS}}
            )
            t["games"] += 1
            for k in _DEFENSE_STATS:
                t[k] += r[k] or 0.0

        written = 0
        season_id = _season_label(game_date)
        metrics_all = {
            tid: self._defense_metrics(t) for tid, t in totals_by_team.items()
        }
        self._rank_defense(metrics_all)
        for team_id, metrics in metrics_all.items():
            self.db.upsert_team_defense(
                {
                    "team_id": team_id,
                    "snapshot_date": game_date,
                    "season_id": season_id,
                    "position_scope": "ALL",
                    **metrics,
                }
            )
            written += 1

        # Position-scoped DvP: split allowed totals by player position.
        scope_totals: dict = {}
        for r in self._allowed_rows(game_date, effective_start, group_by_position=True):
            for scope in self._scopes_for_position(r["position"]):
                key = (r["team_id"], scope)
                t = scope_totals.setdefault(
                    key, {"games": 0, **{k: 0.0 for k in _DEFENSE_STATS}}
                )
                t["games"] += 1
                for k in _DEFENSE_STATS:
                    t[k] += r[k] or 0.0

        by_scope: dict = {}
        for (team_id, scope), totals in scope_totals.items():
            by_scope.setdefault(scope, {})[team_id] = self._defense_metrics(totals)
        for scope, metrics_by_team in by_scope.items():
            self._rank_defense(metrics_by_team)
            for team_id, metrics in metrics_by_team.items():
                self.db.upsert_team_defense(
                    {
                        "team_id": team_id,
                        "snapshot_date": game_date,
                        "season_id": season_id,
                        "position_scope": scope,
                        **metrics,
                    }
                )
                written += 1

        self.db.log_ingestion("nba_stats.team_defense.derived", written)
        return written


_DEFENSE_STATS = (
    "pts",
    "reb",
    "ast",
    "fg3m",
    "fgm",
    "fga",
    "fg3a",
    "fta",
    "oreb",
    "tov",
)
