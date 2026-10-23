"""Feature transformations & EWMA windows (Milestone 2).

Every feature for a target game date is built strictly from rows with
``game_date < target_date`` — same-day and future games are never visible,
enforcing the zero-leakage rule from AGENTS.md.

Feature groups:
  - Rolling means over 3/7/20-game windows per box-score stat
  - EWMA (recent games weighted heaviest) per box-score stat
  - Per-minute rates and usage-volume proxies
  - Rest schedule: days rest, back-to-back, 3-in-4 flags
  - Context: blowout index, altitude flag
  - Matchup vulnerability: DvP percentile from fact_team_defense_daily
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterator, Optional

from src.ingestion.db import ProphitDB

ROLLING_WINDOWS = (3, 7, 20)
EWMA_HALFLIFE_GAMES = 3.0
ALTITUDE_HOME_TEAMS = {"DEN", "UTA"}
LEAGUE_TEAM_COUNT = 30
BLOWOUT_TOTAL_BASELINE = 220.0

BOX_SCORE_STATS = (
    "minutes",
    "pts",
    "reb",
    "ast",
    "fg3m",
    "stl",
    "blk",
    "tov",
    "oreb",
    "dreb",
    "fga",
    "fta",
    "pf",
    "plus_minus",
)
RATE_STATS = ("pts", "reb", "ast", "fg3m")
USAGE_FTA_COEFF = 0.44

# stat_type -> generosity column in fact_team_defense_daily
STAT_TO_DEFENSE_COLUMN = {
    "pts": "opp_pts",
    "reb": "opp_reb",
    "ast": "opp_ast",
    "fg3m": "opp_fg3m",
}


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _parse_date(value) -> "datetime.date":
    return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()


def ewma(values: list[float], halflife: float = EWMA_HALFLIFE_GAMES) -> Optional[float]:
    """Exponentially weighted mean; ``values[0]`` is the most recent game."""
    series = [v for v in values if v is not None]
    if not series:
        return None
    weights = [0.5 ** (i / halflife) for i in range(len(series))]
    return sum(v * w for v, w in zip(series, weights)) / sum(weights)


def usage_volume(game_row: dict) -> float:
    """Possessions used in one game: FGA + 0.44*FTA + TOV."""
    return (
        (game_row.get("fga") or 0.0)
        + USAGE_FTA_COEFF * (game_row.get("fta") or 0.0)
        + (game_row.get("tov") or 0.0)
    )


def walk_forward_splits(
    games: list, min_train_games: int = 10
) -> Iterator[tuple[list, object]]:
    """Yield (train, test) pairs expanding over time.

    ``games`` are dicts/rows carrying ``game_date``. Every yielded train set
    is guaranteed to contain only games dated strictly before the test game.
    """
    ordered = sorted(games, key=lambda g: (g["game_date"], str(g["game_id"])))
    for i in range(min_train_games, len(ordered)):
        train, test = ordered[:i], ordered[i]
        test_date = _parse_date(test["game_date"])
        if any(_parse_date(g["game_date"]) >= test_date for g in train):
            raise AssertionError("leakage: train game on/after test date")
        yield train, test


class FeatureStore:
    """Computes leak-free player features from fact_player_games."""

    def __init__(self, db: ProphitDB, halflife: float = EWMA_HALFLIFE_GAMES):
        self.db = db
        self.halflife = halflife

    def games_before(
        self, player_id: int, game_date: str, limit: Optional[int] = None
    ) -> list:
        sql = (
            "SELECT * FROM fact_player_games "
            "WHERE player_id = ? AND game_date < ? "
            "ORDER BY game_date DESC"
        )
        params: tuple = (player_id, str(game_date)[:10])
        if limit:
            sql += " LIMIT ?"
            params += (int(limit),)
        return self.db.query(sql, params)

    def rolling_features(self, player_id: int, game_date: str) -> dict:
        """Rolling window means, EWMA, rates, and rest flags."""
        target = _parse_date(game_date)
        games = self.games_before(player_id, game_date, limit=max(ROLLING_WINDOWS))

        feats: dict = {
            "player_id": player_id,
            "as_of_date": str(game_date)[:10],
            "games_available": len(games),
        }
        for stat in BOX_SCORE_STATS:
            series = [g[stat] for g in games if g[stat] is not None]
            for window in ROLLING_WINDOWS:
                feats[f"{stat}_l{window}"] = _mean(series[:window])
            feats[f"{stat}_ewma"] = ewma(series, self.halflife)

        window_20 = games[:20]
        total_minutes = sum(g["minutes"] or 0.0 for g in window_20)
        for stat in RATE_STATS:
            total = sum(g[stat] or 0.0 for g in window_20)
            feats[f"{stat}_per_min_l20"] = (
                total / total_minutes if total_minutes > 0 else None
            )

        usage_series = [usage_volume(dict(g)) for g in games]
        feats["usage_volume_l7"] = _mean(usage_series[:7])

        if games:
            last_game = _parse_date(games[0]["game_date"])
            feats["days_rest"] = max((target - last_game).days - 1, 0)
            feats["is_b2b"] = int(feats["days_rest"] == 0)
            feats["games_last_4d"] = sum(
                1 for g in games if (target - _parse_date(g["game_date"])).days <= 4
            )
            feats["is_3_in_4"] = int(feats["games_last_4d"] >= 3)
        else:
            feats.update(days_rest=None, is_b2b=0, games_last_4d=0, is_3_in_4=0)
        return feats

    # --- DvP / matchup vulnerability ---------------------------------------

    def _latest_defense_snapshot_date(self, game_date: str) -> Optional[str]:
        row = self.db.query_one(
            "SELECT MAX(snapshot_date) AS d FROM fact_team_defense_daily "
            "WHERE snapshot_date <= ?",
            (str(game_date)[:10],),
        )
        return row["d"] if row else None

    def defense_snapshot(self, team_id: int, game_date: str, scope: str = "ALL"):
        """Most recent defensive snapshot for a team on/before game_date."""
        return self.db.query_one(
            "SELECT * FROM fact_team_defense_daily "
            "WHERE team_id = ? AND position_scope = ? AND snapshot_date <= ? "
            "ORDER BY snapshot_date DESC LIMIT 1",
            (team_id, scope, str(game_date)[:10]),
        )

    def matchup_vulnerability(
        self,
        opponent_team_id: int,
        stat_type: str,
        game_date: str,
        scope: str = "ALL",
    ) -> Optional[float]:
        """DvP percentile in [0, 1]: 1.0 = softest matchup in the league.

        Computed as the share of teams in the latest snapshot allowing less
        of ``stat_type`` than the opponent — i.e. the opponent's generosity
        percentile. Uses only snapshots dated on/before ``game_date``.
        """
        column = STAT_TO_DEFENSE_COLUMN.get(stat_type)
        if column is None:
            return None
        snapshot_date = self._latest_defense_snapshot_date(game_date)
        if snapshot_date is None:
            return None

        rows = self.db.query(
            f"SELECT team_id, {column} AS allowed "
            "FROM fact_team_defense_daily "
            "WHERE snapshot_date = ? AND position_scope = ? "
            f"AND {column} IS NOT NULL",
            (snapshot_date, scope),
        )
        if not rows:
            return None
        opp = next((r for r in rows if r["team_id"] == opponent_team_id), None)
        if opp is None or opp["allowed"] is None:
            return None
        return sum(1 for r in rows if r["allowed"] <= opp["allowed"]) / len(rows)

    # --- game context -------------------------------------------------------

    def game_context(
        self,
        game_date: str,
        opponent_team_id: Optional[int] = None,
        is_home: Optional[bool] = None,
        spread: Optional[float] = None,
        total: Optional[float] = None,
    ) -> dict:
        """Situational features for the upcoming game (all pre-tip-off)."""
        feats: dict = {}
        if spread is not None and total is not None:
            feats["blowout_index"] = abs(spread) * (total / BLOWOUT_TOTAL_BASELINE)
        else:
            feats["blowout_index"] = None

        feats["altitude_game"] = 0
        if opponent_team_id is not None and is_home is False:
            opp = self.db.query_one(
                "SELECT abbreviation FROM dim_teams WHERE team_id = ?",
                (opponent_team_id,),
            )
            if opp and opp["abbreviation"] in ALTITUDE_HOME_TEAMS:
                feats["altitude_game"] = 1

        if spread is not None:
            feats["vegas_spread"] = spread
        if total is not None:
            feats["vegas_total"] = total
        return feats
