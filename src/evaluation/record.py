"""Model record: settle recommendations against actual results.

For every past ``game_date`` that has both consensus lines and actual
box scores, the slate generator is re-run for that date — features are
strictly pre-date, so the reconstructed recommendation is the same one
the pipeline would have produced that morning. Each recommended play is
then graded against the player's actual stat.

The aggregate is the honest track record surfaced on the dashboard:
record (W-L-P), units won at the offered price, ROI, and favored-side
accuracy across every evaluated line.
"""

from __future__ import annotations

import json
from typing import Any, Optional

from src.ingestion.db import ProphitDB, utcnow_iso
from src.publication.static_builder import (
    COMBO_STATS,
    SlateGenerator,
)

MODEL_VERSION = "nb-mc-v1"
FLAT_PRICE = -110


def _payout_units(american: Optional[int]) -> float:
    """Profit per 1 unit staked at the given American price."""
    price = american if american else FLAT_PRICE
    if price > 0:
        return price / 100.0
    return 100.0 / abs(price)


def _actual_stat(
    db: ProphitDB, player_id: int, game_date: str, stat: str
) -> Optional[float]:
    parts = COMBO_STATS.get(stat, (stat,))
    expr = "+".join(f"COALESCE({s}, 0)" for s in parts)
    row = db.query_one(
        f"SELECT {expr} AS v FROM fact_player_games "
        "WHERE player_id = ? AND game_date = ?",
        (player_id, game_date),
    )
    return float(row["v"]) if row else None


def settled_plays(
    db: ProphitDB, min_ev_pct: float = 5.0, through_date: Optional[str] = None
) -> list[dict[str, Any]]:
    """Reconstruct + grade every past play that has an actual result."""
    rows = db.query("""
        SELECT DISTINCT l.game_date AS d
        FROM fact_consensus_lines l
        WHERE l.is_consensus = 1 AND l.game_date IS NOT NULL
          AND EXISTS (
              SELECT 1 FROM fact_player_games g
              WHERE g.game_date = l.game_date AND g.minutes IS NOT NULL
          )
        ORDER BY d
        """)
    dates = [r["d"] for r in rows if not through_date or r["d"] < through_date]

    generator = SlateGenerator(db, min_ev_pct=min_ev_pct)
    records: list[dict[str, Any]] = []
    for game_date in dates:
        for play in generator.generate_plays(game_date):
            actual = _actual_stat(db, play["player_id"], game_date, play["stat"])
            if actual is None:
                continue
            line = play["line"]
            side = play.get("recommended_side")
            price = play["over_price"] if side == "over" else play["under_price"]

            if actual == line:
                result, units = "push", 0.0
            elif side is None:
                result, units = "no_bet", 0.0
            else:
                won = (side == "over") == (actual > line)
                result = "win" if won else "loss"
                units = _payout_units(price) if won else -1.0

            favored = "over" if (play.get("p_over") or 0) >= 0.5 else "under"
            favored_hit = (actual > line) if favored == "over" else (actual < line)

            records.append(
                {
                    "game_date": game_date,
                    "player_id": play["player_id"],
                    "player_name": play["player_name"],
                    "stat": play["stat"],
                    "market": play["market"],
                    "line": line,
                    "model_mean": play["model_mean"],
                    "p_over": play["p_over"],
                    "side": side,
                    "price": price,
                    "actual": actual,
                    "result": result,
                    "units": round(units, 4),
                    "favored_hit": bool(favored_hit) if actual != line else None,
                }
            )
            _log_evaluation(db, play, game_date, actual)
    return records


def _log_evaluation(db: ProphitDB, play: dict, game_date: str, actual: float) -> None:
    """Audit row in fact_model_evaluations (spec 4.1)."""
    db.execute(
        """
        INSERT OR IGNORE INTO fact_model_evaluations (
            eval_time, game_id, player_id, stat_type, model_version,
            projected_mean, projected_minutes, pmf_json, actual_value
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utcnow_iso(),
            f"{game_date}:{play['player_id']}:{play['market']}",
            play["player_id"],
            play["stat"],
            MODEL_VERSION,
            play["model_mean"],
            (play.get("factors") or {}).get("projected_minutes"),
            json.dumps(play.get("pmf") or []),
            actual,
        ),
    )


def summarize_model_stats(
    db: ProphitDB, min_ev_pct: float = 5.0, through_date: Optional[str] = None
) -> dict[str, Any]:
    """Aggregate the betting record + accuracy metrics for the dashboard."""
    records = settled_plays(db, min_ev_pct, through_date)

    bets = [r for r in records if r["result"] in ("win", "loss", "push")]
    wins = sum(1 for r in bets if r["result"] == "win")
    losses = sum(1 for r in bets if r["result"] == "loss")
    pushes = sum(1 for r in bets if r["result"] == "push")
    units = round(sum(r["units"] for r in bets), 2)
    staked = len(bets)
    roi_pct = round(100.0 * units / staked, 1) if staked else None

    decided = [r for r in records if r["favored_hit"] is not None]
    favored_hits = sum(1 for r in decided if r["favored_hit"])
    favored_hit_rate = (
        round(100.0 * favored_hits / len(decided), 1) if decided else None
    )

    graded_dates = sorted({r["game_date"] for r in records})
    return {
        "record": {
            "bets": staked,
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "units": units,
            "roi_pct": roi_pct,
        },
        "accuracy": {
            "sample": len(decided),
            "favored_hit_rate": favored_hit_rate,
        },
        "settled_through": graded_dates[-1] if graded_dates else None,
        "note": "Settled automatically once box scores land in the database.",
    }
