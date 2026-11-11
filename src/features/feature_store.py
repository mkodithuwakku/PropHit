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

import re
import statistics
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
PER_MINUTE_STATS = RATE_STATS + (
    "stl",
    "blk",
    "tov",
    "oreb",
    "dreb",
    "fga",
    "fta",
    "fg3a",
)
USAGE_FTA_COEFF = 0.44
NEUTRAL_TS_PCT = 0.55

# Canonical NBA team-game totals used when the DB only tracks a single
# player per team (synthetic fixtures): keeps on-floor shares well-scaled.
TEAM_GAME_BASELINE = {
    "minutes": 240.0,
    "fga": 88.0,
    "fta": 22.0,
    "tov": 14.0,
    "reb": 44.0,
    "oreb": 10.0,
    "dreb": 34.0,
    "ast": 26.0,
    "fgm": 42.0,
}

# stat_type -> generosity column in fact_team_defense_daily
STAT_TO_DEFENSE_COLUMN = {
    "pts": "opp_pts",
    "reb": "opp_reb",
    "ast": "opp_ast",
    "fg3m": "opp_fg3m",
}

# fact_player_games columns that may be used for locally-derived DvP when
# fact_team_defense_daily has no matching column (e.g. stl / blk / tov).
_DERIVABLE_STATS = frozenset(BOX_SCORE_STATS) | {"fgm", "ftm", "fg3a"}

_STAT_ALIASES = {
    "pts": "pts",
    "points": "pts",
    "reb": "reb",
    "rebounds": "reb",
    "ast": "ast",
    "assists": "ast",
    "fg3m": "fg3m",
    "threes": "fg3m",
    "3pm": "fg3m",
    "3pt": "fg3m",
    "stl": "stl",
    "steals": "stl",
    "blk": "blk",
    "blocks": "blk",
    "tov": "tov",
    "turnovers": "tov",
}
_COMBO_ALIASES = {
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
}


def _stat_components(stat_type: str) -> list:
    """'pts_reb_ast' / 'pra' / 'player_points' -> canonical stat tokens."""
    text = re.sub(r"[^a-z0-9]+", "_", str(stat_type or "").lower()).strip("_")
    if text.startswith("player_"):
        text = text[len("player_") :]
    if text in _COMBO_ALIASES:
        return list(_COMBO_ALIASES[text])
    components = [
        _STAT_ALIASES[token] for token in text.split("_") if token in _STAT_ALIASES
    ]
    return components or [stat_type]


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
        self._poss_cache: dict = {}
        self._features_cache: dict = {}

    def _row_count(self) -> int:
        """Cache-busting key: any inserted row invalidates both caches."""
        row = self.db.query_one("SELECT COUNT(*) AS n FROM fact_player_games")
        return int(row["n"]) if row else 0

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
        key = (player_id, str(game_date)[:10], self._row_count())
        cached = self._features_cache.get(key)
        if cached is not None:
            return dict(cached)
        feats = self._compute_rolling_features(player_id, game_date)
        self._features_cache[key] = feats
        return dict(feats)

    def _compute_rolling_features(self, player_id: int, game_date: str) -> dict:
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
        for stat in PER_MINUTE_STATS:
            total = sum(g[stat] or 0.0 for g in window_20)
            feats[f"{stat}_per_min_l20"] = (
                total / total_minutes if total_minutes > 0 else None
            )
            per_min_series = [
                (g[stat] or 0.0) / g["minutes"] for g in games if g["minutes"]
            ]
            feats[f"{stat}_per_min_ewma"] = ewma(per_min_series, self.halflife)

        usage_series = [usage_volume(dict(g)) for g in games]
        feats["usage_volume_l3"] = _mean(usage_series[:3])
        feats["usage_volume_l7"] = _mean(usage_series[:7])
        feats["usage_volume_l20"] = _mean(usage_series[:20])

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

        self._add_efficiency_features(feats, games)
        self._add_on_floor_features(feats, games)
        self._add_impact_features(feats, games)
        return feats

    # --- advanced stat helpers ---------------------------------------------

    def _add_efficiency_features(self, feats: dict, games: list) -> None:
        """Shooting/scoring efficiency over L3/L7/L20 + EWMA windows."""

        def total(gs, stat):
            return sum(g[stat] or 0.0 for g in gs)

        def ts_pct(gs):
            denom = 2.0 * (total(gs, "fga") + USAGE_FTA_COEFF * total(gs, "fta"))
            return total(gs, "pts") / denom if denom > 0 else NEUTRAL_TS_PCT

        feats["ts_pct_l3"] = ts_pct(games[:3])
        feats["ts_pct_l7"] = ts_pct(games[:7])
        feats["ts_pct_l20"] = ts_pct(games[:20])
        feats["ts_pct_ewma"] = ewma([ts_pct([g]) for g in games], self.halflife)
        feats["ts_regression_delta"] = feats["ts_pct_l3"] - feats["ts_pct_l20"]

        for label, gs in (("l7", games[:7]), ("l20", games[:20])):
            fga = total(gs, "fga")
            feats[f"efg_pct_{label}"] = (
                (total(gs, "fgm") + 0.5 * total(gs, "fg3m")) / fga if fga > 0 else None
            )

        w20 = games[:20]
        fga20 = total(w20, "fga")
        feats["ftr_l20"] = total(w20, "fta") / fga20 if fga20 > 0 else None
        feats["fg3a_rate_l20"] = total(w20, "fg3a") / fga20 if fga20 > 0 else None

    def _team_game_totals(self, game_ids: list) -> dict:
        """{(game_id, team_id): summed team box score} for the given games."""
        if not game_ids:
            return {}
        placeholders = ", ".join("?" for _ in game_ids)
        rows = self.db.query(
            "SELECT game_id, team_id, COUNT(DISTINCT player_id) AS n_players,"
            " SUM(minutes) AS minutes, SUM(fga) AS fga, SUM(fta) AS fta,"
            " SUM(tov) AS tov, SUM(reb) AS reb, SUM(oreb) AS oreb,"
            " SUM(dreb) AS dreb, SUM(ast) AS ast, SUM(fgm) AS fgm"
            f" FROM fact_player_games WHERE game_id IN ({placeholders})"
            " GROUP BY game_id, team_id",
            tuple(game_ids),
        )
        return {(r["game_id"], r["team_id"]): r for r in rows}

    @staticmethod
    def _team_totals_or_baseline(row) -> dict:
        if row is None or (row["n_players"] or 0) <= 1:
            return dict(TEAM_GAME_BASELINE)
        return {k: row[k] or 0.0 for k in TEAM_GAME_BASELINE}

    def _add_on_floor_features(self, feats: dict, games: list) -> None:
        """True team-relative on-floor shares (usage, rebounding, assists)."""
        window = games[:20]
        keys = (
            "usg_pct_l7",
            "usg_pct_l20",
            "usg_pct_ewma",
            "reb_pct_l20",
            "oreb_pct_l20",
            "dreb_pct_l20",
            "ast_pct_l20",
            "tov_pct_l20",
            "ast_to_tov_l20",
            "role_expansion_delta",
            "minutes_trend_delta",
        )
        if not window:
            feats.update({k: None for k in keys})
            return
        totals = self._team_game_totals([g["game_id"] for g in window])

        def on_floor(p_stat, p_min, t_stat, t_min):
            if not p_min or not t_stat or not t_min:
                return None
            return 100.0 * p_stat * (t_min / 5.0) / (p_min * t_stat)

        series = {"usg": [], "reb": [], "oreb": [], "dreb": [], "ast": []}
        sum_tov = sum_usage = sum_ast = 0.0
        for g in window:
            tm = self._team_totals_or_baseline(totals.get((g["game_id"], g["team_id"])))
            p_min = g["minutes"] or 0.0
            p_usage = usage_volume(dict(g))
            t_usage = tm["fga"] + USAGE_FTA_COEFF * tm["fta"] + tm["tov"]
            series["usg"].append(on_floor(p_usage, p_min, t_usage, tm["minutes"]))
            series["reb"].append(
                on_floor(g["reb"] or 0.0, p_min, tm["reb"], tm["minutes"])
            )
            series["oreb"].append(
                on_floor(g["oreb"] or 0.0, p_min, tm["oreb"], tm["minutes"])
            )
            series["dreb"].append(
                on_floor(g["dreb"] or 0.0, p_min, tm["dreb"], tm["minutes"])
            )
            teammate_fgm = tm["fgm"] - (g["fgm"] or 0.0)
            series["ast"].append(
                on_floor(g["ast"] or 0.0, p_min, teammate_fgm, tm["minutes"])
                if teammate_fgm > 0
                else None
            )
            sum_tov += g["tov"] or 0.0
            sum_usage += p_usage
            sum_ast += g["ast"] or 0.0

        def present(values):
            return [v for v in values if v is not None]

        feats["usg_pct_l7"] = _mean(present(series["usg"][:7]))
        feats["usg_pct_l20"] = _mean(present(series["usg"]))
        feats["usg_pct_ewma"] = ewma(present(series["usg"]), self.halflife)
        feats["reb_pct_l20"] = _mean(present(series["reb"]))
        feats["oreb_pct_l20"] = _mean(present(series["oreb"]))
        feats["dreb_pct_l20"] = _mean(present(series["dreb"]))
        feats["ast_pct_l20"] = _mean(present(series["ast"]))
        feats["tov_pct_l20"] = 100.0 * sum_tov / sum_usage if sum_usage > 0 else None
        feats["ast_to_tov_l20"] = sum_ast / max(sum_tov, 1.0)

        uv3, uv20 = feats["usage_volume_l3"], feats["usage_volume_l20"]
        min3, min20 = feats["minutes_l3"], feats["minutes_l20"]
        feats["role_expansion_delta"] = (
            uv3 / max(min3, 1.0) - uv20 / max(min20, 1.0)
            if uv3 is not None and uv20 is not None and min3 and min20
            else None
        )
        feats["minutes_trend_delta"] = (
            min3 - min20 if min3 is not None and min20 is not None else None
        )

    @staticmethod
    def _game_score(g) -> float:
        """Hollinger Game Score for a single game."""
        return (
            (g["pts"] or 0.0)
            + 0.4 * (g["fgm"] or 0.0)
            - 0.7 * (g["fga"] or 0.0)
            - 0.4 * ((g["fta"] or 0.0) - (g["ftm"] or 0.0))
            + 0.7 * (g["oreb"] or 0.0)
            + 0.3 * (g["dreb"] or 0.0)
            + (g["stl"] or 0.0)
            + 0.7 * (g["ast"] or 0.0)
            + 0.7 * (g["blk"] or 0.0)
            - 0.4 * (g["pf"] or 0.0)
            - (g["tov"] or 0.0)
        )

    @staticmethod
    def _pie_numerator(g) -> float:
        """Player Impact Estimate numerator proxy for a single game."""
        return (
            (g["pts"] or 0.0)
            + (g["fgm"] or 0.0)
            + (g["ftm"] or 0.0)
            - (g["fga"] or 0.0)
            - (g["fta"] or 0.0)
            + (g["dreb"] or 0.0)
            + 0.5 * (g["oreb"] or 0.0)
            + (g["ast"] or 0.0)
            + (g["stl"] or 0.0)
            + 0.5 * (g["blk"] or 0.0)
            - (g["pf"] or 0.0)
            - (g["tov"] or 0.0)
        )

    def _add_impact_features(self, feats: dict, games: list) -> None:
        """Composite impact, volatility, foul & plus-minus rate features."""
        window = games[:20]
        scores = [self._game_score(g) for g in games]
        feats["game_score_l7"] = _mean(scores[:7])
        feats["game_score_l20"] = _mean(scores[:20])
        feats["game_score_ewma"] = ewma(scores, self.halflife)

        minutes = [g["minutes"] for g in window if g["minutes"] is not None]
        feats["minutes_std_l20"] = (
            statistics.stdev(minutes) if len(minutes) >= 2 else None
        )

        for stat in ("pts", "reb", "ast", "fg3m"):
            series = [g[stat] for g in window if g[stat] is not None]
            mean = _mean(series)
            feats[f"{stat}_cv_l20"] = (
                statistics.stdev(series) / mean if len(series) >= 2 and mean else None
            )

        total_min = sum(g["minutes"] or 0.0 for g in window)
        feats["foul_rate_per36_l20"] = (
            36.0 * sum(g["pf"] or 0.0 for g in window) / total_min
            if total_min > 0
            else None
        )
        feats["plus_minus_per_min_l20"] = (
            sum(g["plus_minus"] or 0.0 for g in window) / total_min
            if total_min > 0
            else None
        )
        feats["pie_l20"] = (
            sum(self._pie_numerator(g) for g in window) / total_min
            if total_min > 0
            else None
        )

    # --- team pace ----------------------------------------------------------

    def _team_game_possessions(self, game_date: str, min_players: int = 1) -> dict:
        """{team_id: [possessions per game]} strictly before ``game_date``."""
        key = (str(game_date)[:10], min_players, self._row_count())
        if key in self._poss_cache:
            return self._poss_cache[key]
        rows = self.db.query(
            "SELECT team_id, game_id,"
            " SUM(COALESCE(fga,0)) AS fga, SUM(COALESCE(fta,0)) AS fta,"
            " SUM(COALESCE(oreb,0)) AS oreb, SUM(COALESCE(tov,0)) AS tov,"
            " COUNT(DISTINCT player_id) AS n_players"
            " FROM fact_player_games"
            " WHERE team_id IS NOT NULL AND game_date < ?"
            " GROUP BY team_id, game_id"
            " HAVING n_players >= ?",
            (str(game_date)[:10], min_players),
        )
        poss: dict = {}
        for r in rows:
            value = r["fga"] + USAGE_FTA_COEFF * r["fta"] - r["oreb"] + r["tov"]
            poss.setdefault(r["team_id"], []).append(value)
        self._poss_cache[key] = poss
        return poss

    def team_pace(self, team_id: int, game_date: str) -> Optional[float]:
        """Estimated possessions per game strictly before ``game_date``."""
        games = self._team_game_possessions(game_date).get(team_id)
        return sum(games) / len(games) if games else None

    def expected_pace_factor(
        self,
        team_id: Optional[int],
        opp_id: Optional[int],
        game_date: str,
    ) -> float:
        """Expected game pace vs league average, clamped to [0.92, 1.08].

        Returns 1.0 when fewer than 3 multi-player team games exist so
        single-player synthetic fixtures stay neutral.
        """
        poss = self._team_game_possessions(game_date, min_players=2)
        if sum(len(v) for v in poss.values()) < 3:
            return 1.0
        league = [p for games in poss.values() for p in games]
        league_avg = _mean(league)
        if not league_avg:
            return 1.0
        expected = []
        for tid in (team_id, opp_id):
            if tid is not None and poss.get(tid):
                expected.append(sum(poss[tid]) / len(poss[tid]))
        if not expected:
            return 1.0
        factor = (sum(expected) / len(expected)) / league_avg
        return min(max(factor, 0.92), 1.08)

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
        position: Optional[str] = None,
    ) -> Optional[float]:
        """DvP percentile in [0, 1]: 1.0 = softest matchup in the league.

        Supports every box-score stat plus combos ('pts_reb_ast', 'pra'),
        which average their component vulnerabilities. When ``position``
        is given ('PG', 'C', ...) the matching position_scope snapshot is
        used if present, falling back to ``scope`` ('ALL'). Stats without
        a defense-table column (stl/blk/tov) are derived directly from
        fact_player_games — all strictly before ``game_date``.
        """
        if position:
            scoped = self._scope_available(str(position).upper(), game_date)
            if scoped:
                scope = scoped

        components = _stat_components(stat_type)
        if len(components) != 1:
            values = [
                self._single_vulnerability(
                    opponent_team_id, component, game_date, scope
                )
                for component in components
            ]
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else None
        return self._single_vulnerability(
            opponent_team_id, components[0], game_date, scope
        )

    def _scope_available(self, scope: str, game_date: str) -> Optional[str]:
        snapshot_date = self._latest_defense_snapshot_date(game_date)
        if snapshot_date is None:
            return None
        row = self.db.query_one(
            "SELECT 1 FROM fact_team_defense_daily "
            "WHERE snapshot_date = ? AND position_scope = ? LIMIT 1",
            (snapshot_date, scope),
        )
        return scope if row else None

    def _single_vulnerability(
        self,
        opponent_team_id: int,
        stat_type: str,
        game_date: str,
        scope: str,
    ) -> Optional[float]:
        column = STAT_TO_DEFENSE_COLUMN.get(stat_type)
        if column is None:
            return self._derived_vulnerability(opponent_team_id, stat_type, game_date)
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
        opp = next((r for r in rows if r["team_id"] == opponent_team_id), None)
        if not rows or opp is None or opp["allowed"] is None:
            # position-scoped snapshot missing this defense -> ALL scope
            if scope != "ALL":
                return self._single_vulnerability(
                    opponent_team_id, stat_type, game_date, "ALL"
                )
            return None
        return sum(1 for r in rows if r["allowed"] <= opp["allowed"]) / len(rows)

    def _derived_vulnerability(
        self, opponent_team_id: int, stat_type: str, game_date: str
    ) -> Optional[float]:
        """Generosity percentile for stats the defense table doesn't track.

        Computed from per-game totals allowed in fact_player_games strictly
        before ``game_date`` — the opponent's per-game allowed average vs
        every other defense in the league.
        """
        if stat_type not in _DERIVABLE_STATS:
            return None
        rows = self.db.query(
            "SELECT team_id, AVG(game_total) AS allowed FROM ("
            " SELECT opponent_team_id AS team_id, game_id,"
            f" SUM(COALESCE({stat_type},0)) AS game_total"
            " FROM fact_player_games"
            " WHERE opponent_team_id IS NOT NULL AND game_date < ?"
            " GROUP BY opponent_team_id, game_id"
            ") GROUP BY team_id",
            (str(game_date)[:10],),
        )
        if len(rows) < 2:
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
