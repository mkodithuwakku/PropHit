"""Consensus odds ingestion adapter.

Wraps The Odds API v4 (or a compatible aggregator) with bounded retries and
caching into SQLite per SRD Section 3.1.3. The HTTP session is injectable so
tests replay fixture payloads with zero network calls.

Storage model (fact_consensus_lines):
  - One row per (event, player, market, bookmaker, snapshot) with the book's
    line and over/under American prices.
  - One additional row per (event, player, market) with
    ``bookmaker='CONSENSUS'`` and ``is_consensus=1`` holding the median line
    and median prices across books.
"""

from __future__ import annotations

import statistics
import time
from typing import Any, Callable, Optional

import requests

from src.ingestion.db import ProphitDB

CONSENSUS_BOOKMAKER = "CONSENSUS"

DEFAULT_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_points_rebounds_assists",
    "player_points_rebounds",
    "player_points_assists",
]


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


class OddsAPIClient:
    """Minimal client for The Odds API v4 with retry + backoff."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.the-odds-api.com/v4",
        sport: str = "basketball_nba",
        regions: str = "us",
        odds_format: str = "american",
        markets: Optional[list[str]] = None,
        max_retries: int = 3,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.sport = sport
        self.regions = regions
        self.odds_format = odds_format
        self.markets = markets or DEFAULT_MARKETS
        self.max_retries = max_retries
        self.session = session or requests.Session()
        self._sleep = sleep_fn

    def _get(self, path: str, params: dict) -> Any:
        params = {**params, "apiKey": self.api_key}
        url = f"{self.base_url}{path}"
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                self._sleep(min(0.5 * (2**attempt), 4.0))
        raise RuntimeError(
            f"Odds API request {path} failed after {self.max_retries} attempts"
        ) from last_error

    def get_events(self) -> list[dict]:
        return self._get(f"/sports/{self.sport}/events", {})

    def get_event_odds(
        self, event_id: str, markets: Optional[list[str]] = None
    ) -> dict:
        return self._get(
            f"/sports/{self.sport}/events/{event_id}/odds",
            {
                "regions": self.regions,
                "oddsFormat": self.odds_format,
                "markets": ",".join(markets or self.markets),
            },
        )


class ConsensusOddsIngestion:
    """Parses event-odds payloads and persists bookmaker + consensus lines."""

    def __init__(
        self,
        db: ProphitDB,
        client: Optional[OddsAPIClient] = None,
        snapshot_time: Optional[str] = None,
    ):
        self.db = db
        self.client = client
        self._snapshot_time = snapshot_time

    def ingest_morning_pull(self, markets: Optional[list[str]] = None) -> int:
        """Full daily pull: list events, fetch odds per event, store all."""
        if self.client is None:
            raise ValueError("ingest_morning_pull requires an OddsAPIClient")
        written = 0
        for event in self.client.get_events():
            payload = self.client.get_event_odds(event["id"], markets)
            payload.setdefault("home_team", event.get("home_team"))
            payload.setdefault("away_team", event.get("away_team"))
            payload.setdefault("commence_time", event.get("commence_time"))
            written += self.ingest_event_odds(payload)
        return written

    def ingest_event_odds(
        self, payload: dict, snapshot_time: Optional[str] = None
    ) -> int:
        """Persist one event's odds payload. Pure & offline-safe."""
        from src.ingestion.db import utcnow_iso

        snapshot = snapshot_time or self._snapshot_time or utcnow_iso()
        event_id = str(payload.get("id", ""))
        commence_time = payload.get("commence_time")
        game_date = (commence_time or "")[:10] or None

        home_team_id, away_team_id = None, None
        home_abbr = self._team_abbr(payload.get("home_team"))
        away_abbr = self._team_abbr(payload.get("away_team"))
        if home_abbr:
            home_team_id = self.db.team_id_for_abbr(home_abbr)
        if away_abbr:
            away_team_id = self.db.team_id_for_abbr(away_abbr)

        # player/market -> list of per-book line records
        grouped: dict[tuple[str, str], list[dict]] = {}
        for bookmaker in payload.get("bookmakers", []):
            book_key = bookmaker.get("key") or bookmaker.get("title") or "?"
            for market in bookmaker.get("markets", []):
                market_key = market.get("key")
                for player_name, record in self._parse_market(market, book_key).items():
                    grouped.setdefault((player_name, market_key), []).append(record)

        written = 0
        for (player_name, market_key), records in grouped.items():
            player_id = self.db.player_id_for_name(player_name)
            if player_id is None:
                # Create a stub dim_player so FK + future joins stay intact.
                player_id = self.db.get_or_create_player_id(player_name)

            base = {
                "event_id": event_id,
                "game_date": game_date,
                "home_team_abbr": home_abbr,
                "away_team_abbr": away_abbr,
                "player_id": player_id,
                "player_name": player_name,
                "market": market_key,
                "snapshot_time": snapshot,
                "commence_time": commence_time,
            }
            for record in records:
                self.db.insert_consensus_line({**base, **record, "is_consensus": 0})
                written += 1

            consensus = self._consensus(records)
            if consensus:
                self.db.insert_consensus_line(
                    {
                        **base,
                        **consensus,
                        "bookmaker": CONSENSUS_BOOKMAKER,
                        "is_consensus": 1,
                    }
                )
                written += 1

        self.db.log_ingestion("consensus_odds.event_odds", written)
        return written

    @staticmethod
    def _team_abbr(full_team_name: Optional[str]) -> Optional[str]:
        """The Odds API uses full names ('Los Angeles Lakers'); map to abbr."""
        if not full_team_name:
            return None
        return TEAM_NAME_TO_ABBR.get(full_team_name)

    @staticmethod
    def _parse_market(market: dict, book_key: str) -> dict[str, dict]:
        """Extract per-player {player_name: line_record} from one market."""
        over: dict[str, dict] = {}
        under: dict[str, dict] = {}
        for outcome in market.get("outcomes", []):
            name = outcome.get("description") or outcome.get("participant")
            if not name:
                continue
            side = (outcome.get("name") or "").lower()
            entry = {
                "price": _to_int(outcome.get("price")),
                "point": _to_float(outcome.get("point")),
            }
            if side == "over":
                over[name] = entry
            elif side == "under":
                under[name] = entry

        players: dict[str, dict] = {}
        for player_name, o in over.items():
            u = under.get(player_name, {})
            players[player_name] = {
                "bookmaker": book_key,
                "line": (
                    o.get("point") if o.get("point") is not None else u.get("point")
                ),
                "over_price": o.get("price"),
                "under_price": u.get("price"),
            }
        return players

    @staticmethod
    def _consensus(records: list[dict]) -> Optional[dict]:
        """Median line and median prices across books."""
        lines = [r["line"] for r in records if r.get("line") is not None]
        overs = [r["over_price"] for r in records if r.get("over_price") is not None]
        unders = [r["under_price"] for r in records if r.get("under_price") is not None]
        if not lines and not overs and not unders:
            return None
        return {
            "line": statistics.median(lines) if lines else None,
            "over_price": round(statistics.median(overs)) if overs else None,
            "under_price": (round(statistics.median(unders)) if unders else None),
        }


TEAM_NAME_TO_ABBR = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}
