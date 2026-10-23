"""Historical proxy backtest: model vs trailing-mean lines.

No free source provides historical prop lines (books only post lines the
morning of each game), so this backtest synthesizes a book-realistic
price for every past player-game: the player's pre-game trailing mean,
rounded to the nearest 0.5, at flat -110 both sides. That is precisely
how books set opening prop lines — near recent production.

The model must therefore beat a price built from the same information
any casual bettor has. Its edge can only come from the things the naive
line ignores: EWMA recency weighting, projected minutes, rest/B2B
factors, opponent context, shrinkage, and the distributional PMF.
"""

from __future__ import annotations

from typing import Any, Optional

from src.decision.ev_calculator import EVCalculator
from src.features.feature_store import FeatureStore
from src.ingestion.db import ProphitDB
from src.models.distribution import DistributionEngine
from src.publication.static_builder import COMBO_STATS

STATS = ("pts", "reb", "ast", "fg3m")
MIN_HISTORY = 10
MINUTES_FLOOR = 10.0  # ignore garbage-time cameos


def _round_half(x: float) -> float:
    return round(x * 2.0) / 2.0


def _trailing_values(
    db: ProphitDB, player_id: int, stat: str, game_date: str, n: int = 20
) -> tuple[list[float], list[float]]:
    """(stat values, minutes) over the n games strictly before game_date."""
    parts = COMBO_STATS.get(stat, (stat,))
    expr = "+".join(f"COALESCE({s}, 0)" for s in parts)
    rows = db.query(
        f"SELECT {expr} AS v, minutes FROM fact_player_games "
        "WHERE player_id = ? AND game_date < ? "
        "ORDER BY game_date DESC LIMIT ?",
        (player_id, str(game_date)[:10], n),
    )
    return (
        [float(r["v"]) for r in rows],
        [float(r["minutes"]) for r in rows],
    )


def backtest(
    db: ProphitDB,
    stats: tuple[str, ...] = STATS,
    min_ev_pct: float = 5.0,
    prior_games: float = 6.0,
    min_history: int = MIN_HISTORY,
    through_date: Optional[str] = None,
    collect: bool = False,
) -> dict[str, Any]:
    """Walk every historical player-game and grade the model.

    Returns per-stat and overall records for two rules:
      - 'favored': always take the side the model PMF favors (all plays)
      - 'ev_gated': only bets where model EV >= min_ev_pct at -110
    With collect=True also returns 'bets_detail': every play's favored
    side, its EV at -110, and its graded result — used by the calibrator
    to sweep EV thresholds without re-running the backtest.
    """
    store = FeatureStore(db)
    dist = DistributionEngine()
    ev = EVCalculator(min_ev_pct=min_ev_pct)

    games = db.query(
        """
        SELECT player_id, game_date, minutes FROM fact_player_games
        WHERE minutes >= ?
        ORDER BY game_date
        """,
        (MINUTES_FLOOR,),
    )

    def blank() -> dict:
        return {
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "units": 0.0,
            "favored_n": 0,
            "favored_hits": 0,
            "favored_pushes": 0,
        }

    per_stat = {s: blank() for s in stats}
    overall = blank()
    detail: list[dict] = []

    for game in games:
        pid, game_date = game["player_id"], game["game_date"]
        if through_date and game_date >= through_date:
            continue
        feats = store.rolling_features(pid, game_date)
        minutes = next(
            (
                float(feats[k])
                for k in ("minutes_ewma", "minutes_l7", "minutes_l20")
                if feats.get(k) is not None
            ),
            None,
        )
        if minutes is None:
            continue

        actual_by_stat = None
        for stat in stats:
            history, _ = _trailing_values(db, pid, stat, game_date)
            if len(history) < min_history:
                continue
            line = _round_half(sum(history) / len(history))
            if line <= 0:
                continue

            rate = feats.get(f"{stat}_per_min_l20")
            if rate is None:
                ewma, l7 = feats.get(f"{stat}_ewma"), feats.get("minutes_l7")
                rate = ewma / max(l7, 1.0) if ewma is not None and l7 else None
            if rate is None:
                continue
            mu_raw = rate * minutes
            n = len(history)
            mu = (n * mu_raw + prior_games * line) / (n + prior_games)

            pmf = dist.neg_binomial_pmf(mu, history=history)
            ev_res = ev.evaluate_line(pmf, line, -110, -110)

            if actual_by_stat is None:
                actual_by_stat = {}
            if stat not in actual_by_stat:
                parts = COMBO_STATS.get(stat, (stat,))
                expr = "+".join(f"COALESCE({s},0)" for s in parts)
                row = db.query_one(
                    f"SELECT {expr} AS v FROM fact_player_games "
                    "WHERE player_id = ? AND game_date = ?",
                    (pid, game_date),
                )
                actual_by_stat[stat] = float(row["v"]) if row else None
            actual = actual_by_stat[stat]
            if actual is None:
                continue

            push = actual == line
            favored = "over" if ev_res["p_over"] >= 0.5 else "under"
            hit = (actual > line) if favored == "over" else (actual < line)
            for bucket in (per_stat[stat], overall):
                bucket["favored_n"] += 1
                if push:
                    bucket["favored_pushes"] += 1
                else:
                    bucket["favored_hits"] += int(hit)

            if collect:
                fav_ev = (
                    ev_res["ev_over_pct"]
                    if favored == "over"
                    else ev_res["ev_under_pct"]
                )
                detail.append(
                    {
                        "stat": stat,
                        "side": favored,
                        "ev_pct": fav_ev,
                        "result": ("push" if push else ("win" if hit else "loss")),
                        "units": 0.0 if push else (0.9091 if hit else -1.0),
                    }
                )

            side = ev_res["recommended_side"]
            if side is None:
                continue
            won = (side == "over") == (actual > line)
            for bucket in (per_stat[stat], overall):
                bucket["bets"] += 1
                if push:
                    bucket["pushes"] += 1
                elif won:
                    bucket["wins"] += 1
                    bucket["units"] += 0.9091
                else:
                    bucket["losses"] += 1
                    bucket["units"] -= 1.0

    def finish(b: dict) -> dict:
        n_decided = b["wins"] + b["losses"]
        fav_decided = b["favored_n"] - b["favored_pushes"]
        return {
            "bets": b["bets"],
            "wins": b["wins"],
            "losses": b["losses"],
            "pushes": b["pushes"],
            "win_pct": (round(100.0 * b["wins"] / n_decided, 1) if n_decided else None),
            "units": round(b["units"], 2),
            "roi_pct": (
                round(100.0 * b["units"] / b["bets"], 1) if b["bets"] else None
            ),
            "favored_n": fav_decided,
            "favored_hit_pct": (
                round(100.0 * b["favored_hits"] / fav_decided, 1)
                if fav_decided
                else None
            ),
        }

    result = {
        "method": (
            "trailing-mean proxy: line = pre-game L20 mean rounded to .5, "
            "flat -110; model graded leak-free per date"
        ),
        "breakeven_pct": 52.4,
        "per_stat": {s: finish(per_stat[s]) for s in stats},
        "overall": finish(overall),
    }
    if collect:
        result["bets_detail"] = detail
    return result
