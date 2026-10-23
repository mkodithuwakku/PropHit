"""SQLite persistence layer for Prophit.

Implements the schema defined in SRD Section 4.1:
  - dim_players / dim_teams
  - fact_player_games
  - fact_team_defense_daily
  - fact_consensus_lines
  - fact_model_evaluations

All ingestion adapters write through ``ProphitDB`` so tests can point at an
in-memory or temporary database with zero external state.
"""

from __future__ import annotations

import sqlite3
import zlib
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

DEFAULT_DB_PATH = Path("data/prophit.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS dim_teams (
    team_id          INTEGER PRIMARY KEY,
    abbreviation     TEXT NOT NULL UNIQUE,
    team_name        TEXT,
    team_city        TEXT,
    conference       TEXT,
    division         TEXT,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dim_players (
    player_id        INTEGER PRIMARY KEY,
    full_name        TEXT NOT NULL,
    normalized_name  TEXT NOT NULL UNIQUE,
    first_name       TEXT,
    last_name        TEXT,
    position         TEXT,
    team_id          INTEGER REFERENCES dim_teams(team_id),
    height_inches    REAL,
    weight_lbs       REAL,
    is_active        INTEGER NOT NULL DEFAULT 1,
    updated_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS fact_player_games (
    game_id               TEXT NOT NULL,
    player_id             INTEGER NOT NULL REFERENCES dim_players(player_id),
    team_id               INTEGER REFERENCES dim_teams(team_id),
    opponent_team_id      INTEGER REFERENCES dim_teams(team_id),
    season_id             TEXT,
    game_date             TEXT NOT NULL,
    matchup               TEXT,
    is_home               INTEGER,
    wl                    TEXT,
    minutes               REAL,
    pts                   INTEGER,
    fgm                   INTEGER,
    fga                   INTEGER,
    fg_pct                REAL,
    fg3m                  INTEGER,
    fg3a                  INTEGER,
    fg3_pct               REAL,
    ftm                   INTEGER,
    fta                   INTEGER,
    ft_pct                REAL,
    oreb                  INTEGER,
    dreb                  INTEGER,
    reb                   INTEGER,
    ast                   INTEGER,
    stl                   INTEGER,
    blk                   INTEGER,
    tov                   INTEGER,
    pf                    INTEGER,
    plus_minus            REAL,
    -- Player tracking / hustle metrics (nullable; populated by tracking feeds)
    touches               REAL,
    front_court_touches   REAL,
    time_of_poss          REAL,
    potential_ast         REAL,
    rebound_chances       REAL,
    deflections           REAL,
    screen_assists        REAL,
    contested_shots       REAL,
    ingested_at           TEXT NOT NULL,
    PRIMARY KEY (game_id, player_id)
);

CREATE INDEX IF NOT EXISTS idx_player_games_player
    ON fact_player_games (player_id, game_date);
CREATE INDEX IF NOT EXISTS idx_player_games_date
    ON fact_player_games (game_date);

CREATE TABLE IF NOT EXISTS fact_team_defense_daily (
    team_id               INTEGER NOT NULL REFERENCES dim_teams(team_id),
    snapshot_date         TEXT NOT NULL,
    season_id             TEXT,
    def_rating            REAL,
    def_rating_rank       INTEGER,
    paint_pts_allowed     REAL,
    paint_pts_rank        INTEGER,
    perimeter_fg3_pct     REAL,
    perimeter_fg3_rank    INTEGER,
    rim_fg_pct_allowed    REAL,
    rim_fg_pct_rank       INTEGER,
    opp_pts               REAL,
    opp_reb               REAL,
    opp_ast               REAL,
    opp_fg3m              REAL,
    position_scope        TEXT NOT NULL DEFAULT 'ALL',
    ingested_at           TEXT NOT NULL,
    PRIMARY KEY (team_id, snapshot_date, position_scope)
);

CREATE TABLE IF NOT EXISTS fact_consensus_lines (
    line_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id           TEXT NOT NULL,
    game_date          TEXT,
    home_team_abbr     TEXT,
    away_team_abbr     TEXT,
    player_team_abbr   TEXT,
    player_id          INTEGER REFERENCES dim_players(player_id),
    player_name        TEXT NOT NULL,
    market             TEXT NOT NULL,
    bookmaker          TEXT NOT NULL,
    line               REAL,
    over_price         INTEGER,
    under_price        INTEGER,
    is_consensus       INTEGER NOT NULL DEFAULT 0,
    snapshot_time      TEXT NOT NULL,
    commence_time      TEXT,
    ingested_at        TEXT NOT NULL,
    UNIQUE (event_id, player_name, market, bookmaker, snapshot_time)
);

CREATE INDEX IF NOT EXISTS idx_consensus_lines_event
    ON fact_consensus_lines (event_id, market);
CREATE INDEX IF NOT EXISTS idx_consensus_lines_player
    ON fact_consensus_lines (player_id, snapshot_time);

CREATE TABLE IF NOT EXISTS fact_model_evaluations (
    eval_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    eval_time          TEXT NOT NULL,
    game_id            TEXT NOT NULL,
    player_id          INTEGER NOT NULL REFERENCES dim_players(player_id),
    stat_type          TEXT NOT NULL,
    model_version      TEXT,
    projected_mean     REAL,
    projected_minutes  REAL,
    pmf_json           TEXT,
    actual_value       REAL,
    UNIQUE (game_id, player_id, stat_type, model_version)
);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    source           TEXT NOT NULL,
    started_at       TEXT NOT NULL,
    finished_at      TEXT,
    records_written  INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'running',
    notes            TEXT
);

CREATE TABLE IF NOT EXISTS model_settings (
    key              TEXT PRIMARY KEY,
    value            TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    reason           TEXT
);
"""


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize_player_name(name: str) -> str:
    """Lowercase, strip punctuation/diacritics-lite for odds<->stats joins."""
    cleaned = []
    for ch in name.strip().lower():
        cleaned.append(ch if ch.isalnum() else " ")
    return " ".join("".join(cleaned).split())


def synthetic_id(namespace: str) -> int:
    """Deterministic ID for entities lacking an official identifier.

    Uses CRC32 offset into a reserved high range so synthetic IDs never
    collide with real NBA player/team IDs.
    """
    return 2_000_000_000 + zlib.crc32(namespace.encode("utf-8")) % 100_000_000


class ProphitDB:
    """Thin wrapper around sqlite3 for all Prophit persistence."""

    def __init__(self, path: "str | Path" = DEFAULT_DB_PATH):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self.initialize_schema()

    def initialize_schema(self) -> None:
        self._conn.executescript(SCHEMA_SQL)
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(fact_consensus_lines)")
        }
        if "player_team_abbr" not in columns:
            self._conn.execute(
                "ALTER TABLE fact_consensus_lines ADD COLUMN player_team_abbr TEXT"
            )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        return self._conn.execute(sql, params)

    def query(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return list(self._conn.execute(sql, params))

    def query_one(self, sql: str, params: tuple = ()) -> Optional[sqlite3.Row]:
        return self._conn.execute(sql, params).fetchone()

    # --- runtime calibration settings --------------------------------------

    def get_setting(self, key: str) -> Optional[str]:
        row = self.query_one("SELECT value FROM model_settings WHERE key = ?", (key,))
        return row["value"] if row else None

    def set_setting(self, key: str, value: str, reason: str = "") -> None:
        self.execute(
            """
            INSERT INTO model_settings (key, value, updated_at, reason)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at,
                reason = excluded.reason
            """,
            (key, str(value), utcnow_iso(), reason),
        )
        self._conn.commit()

    # --- dimension upserts -------------------------------------------------

    def upsert_team(
        self,
        team_id: int,
        abbreviation: str,
        team_name: Optional[str] = None,
        team_city: Optional[str] = None,
    ) -> int:
        # Merge a synthetic team row (created from an abbreviation seen in
        # odds feeds) into the official team id before upserting.
        existing = self.query_one(
            "SELECT team_id FROM dim_teams WHERE abbreviation = ?",
            (abbreviation,),
        )
        if existing and int(existing["team_id"]) != team_id:
            self._promote_team_id(int(existing["team_id"]), team_id)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dim_teams (team_id, abbreviation, team_name, team_city, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(team_id) DO UPDATE SET
                    abbreviation = excluded.abbreviation,
                    team_name    = COALESCE(excluded.team_name, dim_teams.team_name),
                    team_city    = COALESCE(excluded.team_city, dim_teams.team_city),
                    updated_at   = excluded.updated_at
                """,
                (team_id, abbreviation, team_name, team_city, utcnow_iso()),
            )
        return team_id

    def team_id_for_abbr(self, abbreviation: str) -> int:
        row = self.query_one(
            "SELECT team_id FROM dim_teams WHERE abbreviation = ?", (abbreviation,)
        )
        if row:
            return int(row["team_id"])
        return self.upsert_team(synthetic_id(f"team:{abbreviation}"), abbreviation)

    def upsert_player(
        self,
        player_id: int,
        full_name: str,
        first_name: Optional[str] = None,
        last_name: Optional[str] = None,
        position: Optional[str] = None,
        team_id: Optional[int] = None,
        is_active: bool = True,
    ) -> int:
        # If a stub row (synthetic ID from odds feeds) exists for this name,
        # merge it into the real ID before upserting.
        existing = self.player_id_for_name(full_name)
        if existing is not None and existing != player_id:
            self._promote_player_id(existing, player_id)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dim_players (
                    player_id, full_name, normalized_name, first_name, last_name,
                    position, team_id, is_active, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(player_id) DO UPDATE SET
                    full_name  = excluded.full_name,
                    first_name = COALESCE(excluded.first_name, dim_players.first_name),
                    last_name  = COALESCE(excluded.last_name, dim_players.last_name),
                    position   = COALESCE(excluded.position, dim_players.position),
                    team_id    = COALESCE(excluded.team_id, dim_players.team_id),
                    is_active  = excluded.is_active,
                    updated_at = excluded.updated_at
                """,
                (
                    player_id,
                    full_name,
                    normalize_player_name(full_name),
                    first_name,
                    last_name,
                    position,
                    team_id,
                    int(is_active),
                    utcnow_iso(),
                ),
            )
        return player_id

    def _promote_team_id(self, old_id: int, new_id: int) -> None:
        """Merge a synthetic team row into the official team id.

        Repoints every fact/dim reference to ``new_id``, then renames the
        stub (or deletes it if the real row already exists). FK checks are
        deferred to commit.
        """
        if old_id == new_id:
            return
        conn = self._conn
        conn.execute("PRAGMA defer_foreign_keys = ON")
        for table, col in (
            ("dim_players", "team_id"),
            ("fact_player_games", "team_id"),
            ("fact_player_games", "opponent_team_id"),
            ("fact_team_defense_daily", "team_id"),
        ):
            conn.execute(
                f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                (new_id, old_id),
            )
        if self.query_one("SELECT 1 FROM dim_teams WHERE team_id = ?", (new_id,)):
            conn.execute("DELETE FROM dim_teams WHERE team_id = ?", (old_id,))
        else:
            conn.execute(
                "UPDATE dim_teams SET team_id = ? WHERE team_id = ?",
                (new_id, old_id),
            )
        conn.commit()

    def _promote_player_id(self, old_id: int, new_id: int) -> None:
        """Merge a stub player row into the real-NBA-id counterpart.

        Repoints every fact table to ``new_id``, then either renames the
        stub row (if ``new_id`` is unused) or deletes it (if the real row
        already exists). FK checks are deferred to commit.
        """
        if old_id == new_id:
            return
        conn = self._conn
        conn.execute("PRAGMA defer_foreign_keys = ON")
        for table, col in (
            ("fact_player_games", "player_id"),
            ("fact_consensus_lines", "player_id"),
            ("fact_model_evaluations", "player_id"),
        ):
            conn.execute(
                f"UPDATE {table} SET {col} = ? WHERE {col} = ?",
                (new_id, old_id),
            )
        if self.query_one("SELECT 1 FROM dim_players WHERE player_id = ?", (new_id,)):
            conn.execute("DELETE FROM dim_players WHERE player_id = ?", (old_id,))
        else:
            conn.execute(
                "UPDATE dim_players SET player_id = ? WHERE player_id = ?",
                (new_id, old_id),
            )
        conn.commit()

    def player_id_for_name(self, full_name: str) -> Optional[int]:
        """Resolve a player by normalized name; None if unknown."""
        row = self.query_one(
            "SELECT player_id FROM dim_players WHERE normalized_name = ?",
            (normalize_player_name(full_name),),
        )
        return int(row["player_id"]) if row else None

    def get_or_create_player_id(self, full_name: str) -> int:
        existing = self.player_id_for_name(full_name)
        if existing is not None:
            return existing
        return self.upsert_player(
            synthetic_id(f"player:{normalize_player_name(full_name)}"), full_name
        )

    # --- fact inserts ------------------------------------------------------

    def upsert_player_game(self, game: dict) -> None:
        columns = [
            "game_id",
            "player_id",
            "team_id",
            "opponent_team_id",
            "season_id",
            "game_date",
            "matchup",
            "is_home",
            "wl",
            "minutes",
            "pts",
            "fgm",
            "fga",
            "fg_pct",
            "fg3m",
            "fg3a",
            "fg3_pct",
            "ftm",
            "fta",
            "ft_pct",
            "oreb",
            "dreb",
            "reb",
            "ast",
            "stl",
            "blk",
            "tov",
            "pf",
            "plus_minus",
            "touches",
            "front_court_touches",
            "time_of_poss",
            "potential_ast",
            "rebound_chances",
            "deflections",
            "screen_assists",
            "contested_shots",
            "ingested_at",
        ]
        row = {col: game.get(col) for col in columns}
        row["ingested_at"] = utcnow_iso()
        placeholders = ", ".join("?" for _ in columns)
        with self.transaction() as conn:
            conn.execute(
                f"""
                INSERT INTO fact_player_games ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(game_id, player_id) DO UPDATE SET
                    team_id          = excluded.team_id,
                    opponent_team_id = excluded.opponent_team_id,
                    minutes          = excluded.minutes,
                    pts              = excluded.pts,
                    reb              = excluded.reb,
                    ast              = excluded.ast,
                    ingested_at      = excluded.ingested_at
                """,
                tuple(row[col] for col in columns),
            )

    def insert_consensus_line(self, line: dict) -> None:
        columns = [
            "event_id",
            "game_date",
            "home_team_abbr",
            "away_team_abbr",
            "player_team_abbr",
            "player_id",
            "player_name",
            "market",
            "bookmaker",
            "line",
            "over_price",
            "under_price",
            "is_consensus",
            "snapshot_time",
            "commence_time",
            "ingested_at",
        ]
        row = {col: line.get(col) for col in columns}
        row["ingested_at"] = utcnow_iso()
        placeholders = ", ".join("?" for _ in columns)
        with self.transaction() as conn:
            conn.execute(
                f"""
                INSERT INTO fact_consensus_lines ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(event_id, player_name, market, bookmaker, snapshot_time)
                DO UPDATE SET
                    line        = excluded.line,
                    over_price  = excluded.over_price,
                    under_price = excluded.under_price,
                    player_team_abbr = excluded.player_team_abbr,
                    ingested_at = excluded.ingested_at
                """,
                tuple(row[col] for col in columns),
            )

    def upsert_team_defense(self, row_data: dict) -> None:
        columns = [
            "team_id",
            "snapshot_date",
            "season_id",
            "def_rating",
            "def_rating_rank",
            "paint_pts_allowed",
            "paint_pts_rank",
            "perimeter_fg3_pct",
            "perimeter_fg3_rank",
            "rim_fg_pct_allowed",
            "rim_fg_pct_rank",
            "opp_pts",
            "opp_reb",
            "opp_ast",
            "opp_fg3m",
            "position_scope",
            "ingested_at",
        ]
        row = {col: row_data.get(col) for col in columns}
        row["position_scope"] = row["position_scope"] or "ALL"
        row["ingested_at"] = utcnow_iso()
        placeholders = ", ".join("?" for _ in columns)
        with self.transaction() as conn:
            conn.execute(
                f"""
                INSERT INTO fact_team_defense_daily ({", ".join(columns)})
                VALUES ({placeholders})
                ON CONFLICT(team_id, snapshot_date, position_scope) DO UPDATE SET
                    def_rating         = excluded.def_rating,
                    def_rating_rank    = excluded.def_rating_rank,
                    paint_pts_allowed  = excluded.paint_pts_allowed,
                    perimeter_fg3_pct  = excluded.perimeter_fg3_pct,
                    rim_fg_pct_allowed = excluded.rim_fg_pct_allowed,
                    ingested_at        = excluded.ingested_at
                """,
                tuple(row[col] for col in columns),
            )

    # --- ingestion audit ---------------------------------------------------

    def log_ingestion(
        self, source: str, records_written: int, status: str = "ok", notes: str = ""
    ) -> None:
        now = utcnow_iso()
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_runs
                    (source, started_at, finished_at, records_written, status, notes)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (source, now, now, records_written, status, notes),
            )
