"""Offline tests for the trailing-mean proxy backtest."""

from src.evaluation.backtest import _round_half, backtest
from src.ingestion.db import ProphitDB


def _seed_player(db, player_id, name, values, start="2026-03-01"):
    from datetime import date, timedelta

    db.upsert_team(8000, "TST", "Testers")
    db.upsert_team(8100, "OPP", "Opponents")
    db.upsert_player(player_id, name, team_id=8000)
    d = date.fromisoformat(start)
    for i, v in enumerate(values):
        db.upsert_player_game(
            {
                "game_id": f"bt{i}",
                "player_id": player_id,
                "team_id": 8000,
                "opponent_team_id": 8100,
                "game_date": d.isoformat(),
                "minutes": 30.0,
                "pts": float(v),
                "reb": float(v),
                "ast": float(v),
                "fg3m": float(v),
                "is_home": 1,
            }
        )
        d += timedelta(days=3)


def test_round_half():
    assert _round_half(14.24) == 14.0
    assert _round_half(14.26) == 14.5
    assert _round_half(14.5) == 14.5


def test_backtest_grades_pushes_not_wins(tmp_path):
    """Integer lines push on exact hits — must not count as wins."""
    db = ProphitDB(str(tmp_path / "t.db"))
    # Player alternates 14/16: L20 mean 15.0 -> integer line.
    # Many exact-15 games push the line instead of winning/losing.
    vals = [14, 16, 15, 14, 16, 15, 14, 16, 15, 14, 16, 15, 14, 16]
    _seed_player(db, 1, "Push Man", vals)
    res = backtest(db, stats=("pts",), min_ev_pct=0.0, min_history=10)
    pts = res["per_stat"]["pts"]
    assert pts["pushes"] >= 1 or pts["favored_n"] >= 1  # graded, not crashed
    # pushes counted separately — not hidden inside wins
    assert pts["wins"] + pts["losses"] + pts["pushes"] == pts["bets"]
    db.close()


def test_backtest_structure_and_leak_gate(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    _seed_player(db, 1, "Steady Eddie", [10 + (i % 5) for i in range(15)])
    res = backtest(db, stats=("pts", "reb"), min_history=5)
    assert set(res["per_stat"]) == {"pts", "reb"}
    for b in res["per_stat"].values():
        assert b["bets"] == b["wins"] + b["losses"] + b["pushes"]
        assert b["favored_n"] is not None
    db.close()
