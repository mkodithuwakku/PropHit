"""Vacated usage & minutes redistribution (Milestone 2).

When teammates are ruled OUT, their per-game minutes and usage volume are
redistributed to the remaining rotation players. Minutes follow each active
player's established minutes share; usage follows each active player's
established usage share — a practical on/off-split approximation computed
strictly from games played before the target date (zero leakage).
"""

from __future__ import annotations

from typing import Iterable, Optional

from src.features.feature_store import usage_volume
from src.ingestion.db import ProphitDB, normalize_player_name

DEFAULT_BASELINE_WINDOW = 10


class VacatedUsageEngine:
    """Computes per-player baseline shares and absence redistribution."""

    def __init__(self, db: ProphitDB, window: int = DEFAULT_BASELINE_WINDOW):
        self.db = db
        self.window = window

    def team_player_baselines(self, team_id: int, game_date: str) -> dict:
        """Recent per-game minutes & usage volume for each team player.

        Returns ``{player_id: {"minutes_pg", "usage_pg", "games"}}`` using
        each player's last ``window`` games strictly before ``game_date``.
        """
        rows = self.db.query(
            "SELECT * FROM fact_player_games "
            "WHERE team_id = ? AND game_date < ? "
            "ORDER BY game_date DESC",
            (team_id, str(game_date)[:10]),
        )
        per_player: dict[int, list] = {}
        for row in rows:
            per_player.setdefault(row["player_id"], [])
            if len(per_player[row["player_id"]]) < self.window:
                per_player[row["player_id"]].append(row)

        baselines = {}
        for player_id, games in per_player.items():
            minutes = [g["minutes"] or 0.0 for g in games]
            usage = [usage_volume(dict(g)) for g in games]
            baselines[player_id] = {
                "minutes_pg": sum(minutes) / len(minutes),
                "usage_pg": sum(usage) / len(usage),
                "games": len(games),
            }
        return baselines

    def resolve_out_players(
        self,
        entries: Iterable,
        team_id: int,
        statuses: tuple[str, ...] = ("out",),
    ) -> set[int]:
        """Map injury report entries (names) to player_ids on ``team_id``.

        Accepts ``InjuryEntry`` objects or dicts with ``player_name`` /
        ``status`` keys. Only the given statuses count as absent.
        """
        team = self.db.query_one(
            "SELECT abbreviation FROM dim_teams WHERE team_id = ?", (team_id,)
        )
        team_abbr = team["abbreviation"] if team else None

        out_ids: set[int] = set()
        for entry in entries:
            status = (
                getattr(entry, "status", None)
                if not isinstance(entry, dict)
                else entry.get("status")
            )
            name = (
                getattr(entry, "player_name", None)
                if not isinstance(entry, dict)
                else entry.get("player_name") or entry.get("player")
            )
            entry_team = (
                getattr(entry, "team_abbr", None)
                if not isinstance(entry, dict)
                else entry.get("team_abbr") or entry.get("team")
            )
            if not name or str(status).lower() not in statuses:
                continue
            if entry_team and team_abbr and entry_team != team_abbr:
                continue
            row = self.db.query_one(
                "SELECT player_id FROM dim_players "
                "WHERE normalized_name = ? AND team_id = ?",
                (normalize_player_name(str(name)), team_id),
            )
            if row:
                out_ids.add(int(row["player_id"]))
        return out_ids

    def redistribute(
        self, team_id: int, out_player_ids: Iterable[int], game_date: str
    ) -> dict:
        """Redistribute OUT teammates' minutes & usage to active players.

        Returns::

            {
              "team_id": ..., "game_date": ...,
              "vacated_minutes": float, "vacated_usage": float,
              "out_player_ids": [...],
              "adjustments": {
                  player_id: {
                      "base_minutes_pg", "base_usage_pg",
                      "extra_minutes", "extra_usage",
                      "adj_minutes_pg", "adj_usage_pg",
                      "usage_boost_pct",
                  }
              },
            }
        """
        baselines = self.team_player_baselines(team_id, game_date)
        out_ids = {int(pid) for pid in out_player_ids} & set(baselines)
        active = {pid: b for pid, b in baselines.items() if pid not in out_ids}

        vacated_minutes = sum(baselines[p]["minutes_pg"] for p in out_ids)
        vacated_usage = sum(baselines[p]["usage_pg"] for p in out_ids)

        total_min = sum(b["minutes_pg"] for b in active.values())
        total_usg = sum(b["usage_pg"] for b in active.values())

        adjustments = {}
        for pid, base in active.items():
            min_share = base["minutes_pg"] / total_min if total_min else 0.0
            usg_share = base["usage_pg"] / total_usg if total_usg else 0.0
            extra_minutes = vacated_minutes * min_share
            extra_usage = vacated_usage * usg_share
            adjustments[pid] = {
                "base_minutes_pg": base["minutes_pg"],
                "base_usage_pg": base["usage_pg"],
                "extra_minutes": extra_minutes,
                "extra_usage": extra_usage,
                "adj_minutes_pg": base["minutes_pg"] + extra_minutes,
                "adj_usage_pg": base["usage_pg"] + extra_usage,
                "usage_boost_pct": (
                    extra_usage / base["usage_pg"] if base["usage_pg"] else 0.0
                ),
            }

        return {
            "team_id": team_id,
            "game_date": str(game_date)[:10],
            "vacated_minutes": vacated_minutes,
            "vacated_usage": vacated_usage,
            "out_player_ids": sorted(out_ids),
            "adjustments": adjustments,
        }
