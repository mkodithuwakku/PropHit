"""Consensus prop-line ingestion — 100% free, zero-auth public sources.

Primary source: PrizePicks Public REST API
  ``GET https://api.prizepicks.com/projections?league_id=7`` (NBA)
  returns the entire evening slate in one JSON:API payload — no key, no
  headers, no rate limits on a single daily pull.
Fallback: Underdog Fantasy public over/under lines.

Both adaptors normalize into the same ``fact_consensus_lines`` schema:
``player_name``, ``stat_type``/market, ``line``, ``over_price``,
``under_price`` (flat -110 unless the source gives payouts), ``game_date``.
Each source row is written once per bookmaker plus once as the
``CONSENSUS`` row so the rest of the pipeline is source-agnostic.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.db import ProphitDB

CONSENSUS_BOOKMAKER = "CONSENSUS"
STATPICK_BOOKMAKER = "statpick"
PRIZEPICKS_BOOKMAKER = "prizepicks"
UNDERDOG_BOOKMAKER = "underdog"
FLAT_PRICE = -110  # PrizePicks-style flat payout -> -110 equivalent

STATPICK_STAT_TO_MARKET = {
    "points": "player_points",
    "rebounds": "player_rebounds",
    "assists": "player_assists",
    "threes": "player_threes",
    "steals": "player_steals",
    "blocks": "player_blocks",
    "turnovers": "player_turnovers",
    "points-rebounds": "player_points_rebounds",
    "points-assists": "player_points_assists",
    "points-rebounds-assists": "player_points_rebounds_assists",
    "rebounds-assists": "player_rebounds_assists",
}


def _statpick_market(slug: Optional[str]) -> Optional[str]:
    if not slug:
        return None
    if slug in STATPICK_STAT_TO_MARKET:
        return STATPICK_STAT_TO_MARKET[slug]
    clean = re.sub(r"[^a-z0-9]+", "_", slug.strip().lower()).strip("_")
    return f"player_{clean}" if clean else None


def _nba_id_from_headshot(url: Optional[str]) -> Optional[int]:
    """statpick headshots embed the official NBA player id."""
    if not url:
        return None
    m = re.search(r"/(\d+)\.[a-zA-Z]+$", str(url))
    return int(m.group(1)) if m else None


PP_STAT_TYPE_TO_MARKET = {
    "Points": "player_points",
    "Rebounds": "player_rebounds",
    "Assists": "player_assists",
    "3-PT Made": "player_threes",
    "Pts+Rebs+Asts": "player_points_rebounds_assists",
    "Pts+Rebs": "player_points_rebounds",
    "Pts+Asts": "player_points_assists",
    "Rebs+Asts": "player_rebounds_assists",
    "Blocked Shots": "player_blocks",
    "Steals": "player_steals",
    "Turnovers": "player_turnovers",
}


def _market_key(stat_type: Optional[str]) -> Optional[str]:
    if not stat_type:
        return None
    if stat_type in PP_STAT_TYPE_TO_MARKET:
        return PP_STAT_TYPE_TO_MARKET[stat_type]
    slug = re.sub(r"[^a-z0-9]+", "_", stat_type.strip().lower()).strip("_")
    return f"player_{slug}" if slug else None


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _multiplier_to_american(mult: Optional[float]) -> Optional[int]:
    if not mult or mult <= 1.0:
        return None
    if mult >= 2.0:
        return int(round((mult - 1.0) * 100.0))
    return int(round(-100.0 / (mult - 1.0)))


class _RetryingClient:
    """Shared zero-auth HTTP client with bounded retry + backoff."""

    def __init__(
        self,
        base_url: str,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._sleep = sleep_fn

    def _get_json(self, path: str, params: Optional[dict] = None) -> Any:
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params or {}, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(min(0.5 * (2**attempt), 4.0))
        raise RuntimeError(
            f"GET {url} failed after {self.max_retries} attempts"
        ) from last_error


class PrizePicksClient(_RetryingClient):
    """Zero-auth client for the PrizePicks public projections feed."""

    def __init__(self, league_id: int = 7, **kwargs):
        super().__init__("https://api.prizepicks.com", **kwargs)
        self.league_id = league_id

    def get_projections(self) -> dict:
        return self._get_json(
            "/projections",
            {"league_id": self.league_id, "single_stat": "true"},
        )


class StatPickClient(_RetryingClient):
    """Zero-auth client for the statpick.ai public prop-lines API.

    statpick publishes documented, unauthenticated JSON endpoints that
    aggregate lines across ~5 sportsbooks — designed for programmatic
    access, unlike bot-protected sportsbook endpoints.
    """

    def __init__(self, sport: str = "nba", **kwargs):
        super().__init__("https://api.statpick.ai", **kwargs)
        self.sport = sport

    def get_slate(self) -> dict:
        return self._get_json(f"/api/{self.sport}/prop-pages/slate")

    def get_player_prop_page(self, player_slug: str) -> dict:
        """All stat-market lines for one player in a single call."""
        return self._get_json(f"/api/{self.sport}/prop-pages/player/{player_slug}")


class UnderdogClient(_RetryingClient):
    """Zero-auth fallback client for Underdog public over/under lines."""

    def __init__(self, **kwargs):
        super().__init__("https://api.underdogfantasy.com", **kwargs)

    def get_over_under_lines(self) -> dict:
        return self._get_json("/beta/v3/over_under_lines", {"sport": "nba"})


class ConsensusOddsIngestion:
    """Parses PrizePicks/Underdog payloads into fact_consensus_lines."""

    def __init__(
        self,
        db: ProphitDB,
        statpick: Optional["StatPickClient"] = None,
        primary: Optional[PrizePicksClient] = None,
        fallback: Optional[UnderdogClient] = None,
        snapshot_time: Optional[str] = None,
    ):
        self.db = db
        self.statpick = statpick
        self.primary = primary
        self.fallback = fallback
        self._snapshot_time = snapshot_time

    # --- live pull ---------------------------------------------------------

    def ingest_morning_pull(self) -> dict:
        """Daily pull across the zero-auth source chain.

        Order: statpick.ai (designed for programmatic access) ->
        PrizePicks public projections -> Underdog public lines. The first
        source returning a non-empty board wins.
        """
        chain = []
        if self.statpick is not None:
            chain.append(
                (
                    STATPICK_BOOKMAKER,
                    self.statpick.get_slate,
                    self.ingest_statpick,
                )
            )
        if self.primary is not None:
            chain.append(
                (
                    PRIZEPICKS_BOOKMAKER,
                    self.primary.get_projections,
                    self.ingest_prizepicks,
                )
            )
        if self.fallback is not None:
            chain.append(
                (
                    UNDERDOG_BOOKMAKER,
                    self.fallback.get_over_under_lines,
                    self.ingest_underdog,
                )
            )
        if not chain:
            raise ValueError("ingest_morning_pull requires a client")

        errors: dict[str, str] = {}
        for name, fetch, ingest in chain:
            try:
                payload = fetch()
                written = ingest(payload)
                if name == STATPICK_BOOKMAKER and written:
                    # Slate only carries each player's headline prop; pull
                    # per-player prop pages for full-market coverage.
                    written += self._enrich_statpick(payload)
                if written:
                    result = {"source": name, "written": written}
                    if errors:
                        result["skipped_sources"] = errors
                    return result
                errors[name] = "empty board"
            except Exception as exc:
                errors[name] = str(exc)
        raise RuntimeError(f"all consensus sources failed: {errors}")

    # --- statpick.ai ---------------------------------------------------------

    def ingest_statpick(
        self, payload: dict, snapshot_time: Optional[str] = None
    ) -> int:
        """Persist a statpick.ai prop-pages slate payload."""
        from src.ingestion.db import utcnow_iso

        snapshot = snapshot_time or self._snapshot_time or utcnow_iso()
        data = payload.get("data", payload)
        # statpick's slate date is the NBA evening slate; games tipping
        # after midnight UTC still belong to it.
        slate_date = str(data.get("date") or "")[:10] or None

        written = 0
        for game in data.get("games", []):
            home_abbr = (game.get("homeTeam") or {}).get("abbreviation")
            away_abbr = (game.get("awayTeam") or {}).get("abbreviation")
            event_id = f"sp-{game.get('gameId')}"
            commence = game.get("dateAndTime")

            for player in game.get("players", []):
                player_name = player.get("playerName")
                team_abbr = player.get("teamAbbr")
                nba_id = _nba_id_from_headshot(player.get("headshotUrl"))
                stats = player.get("stats") or [player.get("headline")]
                for stat in stats:
                    if not stat:
                        continue
                    market = _statpick_market(stat.get("statSlug") or stat.get("slug"))
                    # Slate headlines carry flat fields; prop-page stats nest
                    # them under a 'line' object.
                    line_obj = stat.get("line")
                    if isinstance(line_obj, dict):
                        line = _to_float(line_obj.get("line"))
                        over = line_obj.get("overOdds")
                        under = line_obj.get("underOdds")
                    else:
                        line = _to_float(line_obj)
                        over = stat.get("overOdds")
                        under = stat.get("underOdds")
                    if not player_name or market is None or line is None:
                        continue
                    written += self._write_line(
                        bookmaker=STATPICK_BOOKMAKER,
                        event_id=event_id,
                        player_name=player_name,
                        market=market,
                        line=line,
                        over_price=over,
                        under_price=under,
                        snapshot=snapshot,
                        commence_time=commence,
                        game_date=slate_date,
                        home_abbr=home_abbr,
                        away_abbr=away_abbr,
                        team_abbr=team_abbr,
                        position=player.get("position"),
                        player_id=nba_id,
                    )
        self.db.log_ingestion("consensus_odds.statpick", written)
        return written

    def ingest_statpick_prop_page(
        self, payload: dict, snapshot_time: Optional[str] = None
    ) -> int:
        """Persist a single player's prop-page (all stat markets).

        The page's `nextGame` supplies event context — gameUrl carries the
        event id, opponentAbbr + isHome reconstruct the matchup.
        """
        from src.ingestion.db import utcnow_iso

        snapshot = snapshot_time or self._snapshot_time or utcnow_iso()
        data = payload.get("data", payload)
        player = data.get("player") or {}
        nxt = data.get("nextGame") or {}

        player_name = player.get("playerName")
        nba_id = _nba_id_from_headshot(player.get("headshotUrl"))
        team_abbr = player.get("teamAbbr")
        opp_abbr = nxt.get("opponentAbbr")
        is_home = bool(nxt.get("isHome"))
        home_abbr = team_abbr if is_home else opp_abbr
        away_abbr = opp_abbr if is_home else team_abbr
        m = re.search(r"/games/nba/(\d+)", str(nxt.get("gameUrl") or ""))
        event_id = f"sp-{m.group(1)}" if m else f"sp-{player.get('playerSlug')}"
        game_date = str(nxt.get("date") or "")[:10] or None

        written = 0
        for stat in data.get("stats", []):
            market = _statpick_market(stat.get("statSlug"))
            line_obj = stat.get("line") or {}
            line = _to_float(line_obj.get("line"))
            if not player_name or market is None or line is None:
                continue
            written += self._write_line(
                bookmaker=STATPICK_BOOKMAKER,
                event_id=event_id,
                player_name=player_name,
                market=market,
                line=line,
                over_price=line_obj.get("overOdds"),
                under_price=line_obj.get("underOdds"),
                snapshot=snapshot,
                commence_time=nxt.get("dateAndTime"),
                game_date=game_date,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                team_abbr=team_abbr,
                position=player.get("position"),
                player_id=nba_id,
            )
        return written

    def _enrich_statpick(self, slate_payload: dict) -> int:
        """Fetch every slate player's prop page for full-market coverage."""
        if self.statpick is None or not hasattr(self.statpick, "get_player_prop_page"):
            return 0
        data = slate_payload.get("data", slate_payload)
        slugs = {
            p.get("playerSlug")
            for g in data.get("games", [])
            for p in g.get("players", [])
            if p.get("playerSlug")
        }
        written = 0
        for slug in sorted(slugs):
            try:
                written += self.ingest_statpick_prop_page(
                    self.statpick.get_player_prop_page(slug)
                )
            except Exception:
                continue  # one bad page must not sink the board
        return written

    # --- PrizePicks ---------------------------------------------------------

    def ingest_prizepicks(
        self, payload: dict, snapshot_time: Optional[str] = None
    ) -> int:
        """Persist a PrizePicks JSON:API payload. Pure & offline-safe."""
        from src.ingestion.db import utcnow_iso

        snapshot = snapshot_time or self._snapshot_time or utcnow_iso()
        players = {
            inc["id"]: inc.get("attributes", {})
            for inc in payload.get("included", [])
            if inc.get("type") in ("new_player", "player")
        }

        written = 0
        for proj in payload.get("data", []):
            attrs = proj.get("attributes", {})
            rel = proj.get("relationships", {}).get("new_player", {})
            player_ref = (rel.get("data") or {}).get("id")
            player = players.get(player_ref, {})

            player_name = player.get("name") or attrs.get("player_name")
            market = _market_key(attrs.get("stat_type"))
            line = _to_float(attrs.get("line_score"))
            if not player_name or market is None or line is None:
                continue

            team_abbr = player.get("team")
            home_abbr, away_abbr = self._pp_matchup(attrs.get("description"), team_abbr)
            commence = attrs.get("start_time")
            event_id = str(
                attrs.get("game_id") or f"pp-{(commence or '')[:10]}-{player_name}"
            )
            position = player.get("position")

            written += self._write_line(
                bookmaker=PRIZEPICKS_BOOKMAKER,
                event_id=event_id,
                player_name=player_name,
                market=market,
                line=line,
                over_price=FLAT_PRICE,
                under_price=FLAT_PRICE,
                snapshot=snapshot,
                commence_time=commence,
                home_abbr=home_abbr,
                away_abbr=away_abbr,
                team_abbr=team_abbr,
                position=position,
            )
        self.db.log_ingestion("consensus_odds.prizepicks", written)
        return written

    @staticmethod
    def _pp_matchup(
        description: Optional[str], team_abbr: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Parse PrizePicks description into (home_abbr, away_abbr).

        Formats seen: 'UTA' (opponent only), 'vs UTA', 'DEN @ UTA',
        '@ UTA'. Opponent-only descriptions store the opponent as away and
        leave home None — home/away is not part of the public payload.
        """
        if not description:
            return None, None
        desc = str(description).strip()
        if "@" in desc:
            away, _, home = desc.partition("@")
            return home.strip() or None, away.strip() or None
        if desc.lower().startswith("vs"):
            opp = desc[2:].strip()
            return team_abbr, opp or None
        return None, desc or None

    # --- Underdog fallback ---------------------------------------------------

    def ingest_underdog(
        self, payload: dict, snapshot_time: Optional[str] = None
    ) -> int:
        """Persist an Underdog over_under_lines payload (best-effort)."""
        from src.ingestion.db import utcnow_iso

        snapshot = snapshot_time or self._snapshot_time or utcnow_iso()
        players = {p["id"]: p for p in payload.get("players", [])}
        games = {g["id"]: g for g in payload.get("games", [])}

        written = 0
        for entry in payload.get("over_under_lines", []):
            ou = entry.get("over_under", entry)
            stat_meta = ou.get("appearance_stat") or ou.get("stat") or {}
            stat_type = (
                stat_meta.get("display_stat") or stat_meta.get("abbr") or ou.get("stat")
            )
            market = _market_key(stat_type)
            line = _to_float(ou.get("stat_value"))
            if line is None:
                line = self._line_from_title(ou.get("title"))

            player = players.get(ou.get("player_id")) or {}
            player_name = (
                f"{player.get('first_name', '')} " f"{player.get('last_name', '')}"
            ).strip() or ou.get("player_name")
            if not player_name or market is None or line is None:
                continue

            game = games.get(ou.get("game_id")) or {}
            prices = self._ud_prices(entry.get("options", []))
            written += self._write_line(
                bookmaker=UNDERDOG_BOOKMAKER,
                event_id=str(ou.get("game_id") or entry.get("id") or "ud"),
                player_name=player_name,
                market=market,
                line=line,
                over_price=prices[0],
                under_price=prices[1],
                snapshot=snapshot,
                commence_time=game.get("scheduled_at"),
                home_abbr=game.get("home_team_id"),
                away_abbr=game.get("away_team_id"),
                team_abbr=player.get("team_id"),
                position=player.get("position"),
            )
        self.db.log_ingestion("consensus_odds.underdog", written)
        return written

    @staticmethod
    def _line_from_title(title: Optional[str]) -> Optional[float]:
        if not title:
            return None
        m = re.search(r"(\d+(?:\.\d+)?)", str(title))
        return float(m.group(1)) if m else None

    @staticmethod
    def _ud_prices(options: list) -> tuple[Optional[int], Optional[int]]:
        over = under = None
        for opt in options:
            choice = (opt.get("choice") or "").lower()
            american = _multiplier_to_american(_to_float(opt.get("payout_multiplier")))
            if choice in ("higher", "over"):
                over = american
            elif choice in ("lower", "under"):
                under = american
        return (
            over if over is not None else FLAT_PRICE,
            under if under is not None else FLAT_PRICE,
        )

    # --- shared writer --------------------------------------------------------

    def _write_line(
        self,
        bookmaker: str,
        event_id: str,
        player_name: str,
        market: str,
        line: float,
        over_price: Optional[int],
        under_price: Optional[int],
        snapshot: str,
        commence_time: Optional[str],
        game_date: Optional[str] = None,
        home_abbr: Optional[str] = None,
        away_abbr: Optional[str] = None,
        team_abbr: Optional[str] = None,
        position: Optional[str] = None,
        player_id: Optional[int] = None,
    ) -> int:
        """Write the bookmaker row + the CONSENSUS mirror row."""
        if player_id is not None:
            # Official NBA id supplied (e.g. statpick headshotUrl) —
            # upsert auto-promotes any pre-existing name stub.
            self.db.upsert_player(player_id, player_name, position=position)
        else:
            player_id = self.db.player_id_for_name(player_name)
            if player_id is None:
                player_id = self.db.get_or_create_player_id(player_name)
        if team_abbr:
            team_id = self.db.team_id_for_abbr(team_abbr)
            self.db.execute(
                "UPDATE dim_players SET team_id = ?, position = "
                "COALESCE(?, position) WHERE player_id = ?",
                (team_id, position, player_id),
            )
        if home_abbr:
            self.db.team_id_for_abbr(home_abbr)
        if away_abbr:
            self.db.team_id_for_abbr(away_abbr)

        base = {
            "event_id": event_id,
            "game_date": game_date or (commence_time or "")[:10] or None,
            "home_team_abbr": home_abbr,
            "away_team_abbr": away_abbr,
            "player_id": player_id,
            "player_name": player_name,
            "market": market,
            "line": line,
            "over_price": over_price,
            "under_price": under_price,
            "snapshot_time": snapshot,
            "commence_time": commence_time,
        }
        self.db.insert_consensus_line(
            {**base, "bookmaker": bookmaker, "is_consensus": 0}
        )
        self.db.insert_consensus_line(
            {
                **base,
                "bookmaker": CONSENSUS_BOOKMAKER,
                "is_consensus": 1,
            }
        )
        return 2
