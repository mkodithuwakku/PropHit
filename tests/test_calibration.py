"""Bi-weekly calibration loop: interval gate, EV sweep, market suppression."""

import json
from datetime import datetime, timedelta, timezone

from src.evaluation.calibrate import (
    get_min_ev,
    get_suppressed_markets,
    run_calibration,
)
from src.ingestion.db import ProphitDB
from src.publication.static_builder import SlateGenerator


def _detail(stat, ev, result, n):
    u = {"win": 0.9091, "loss": -1.0, "push": 0.0}[result]
    return [
        {"stat": stat, "side": "over", "ev_pct": ev, "result": result, "units": u}
        for _ in range(n)
    ]


def _fake_backtest(detail):
    def fn(db, **kw):
        return {
            "per_stat": {},
            "overall": {},
            "bets_detail": detail,
        }

    return fn


def test_skips_when_last_run_recent(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    db.set_setting(
        "last_calibration",
        (datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
    )
    res = run_calibration(db, backtest_fn=_fake_backtest([]))
    assert res["ran"] is False
    db.close()


def test_first_run_writes_settings(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    res = run_calibration(db, backtest_fn=_fake_backtest([]))
    assert res["ran"] is True
    assert db.get_setting("last_calibration") is not None
    assert get_suppressed_markets(db) == []
    db.close()


def test_gate_moves_up_clamped(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    # Winners only at high EV -> best gate is 15, but clamp limits to 5+2=7.
    detail = _detail("pts", 14.0, "win", 150) + _detail("pts", 6.0, "loss", 120)
    res = run_calibration(db, backtest_fn=_fake_backtest(detail))
    assert res["min_ev_pct"] == 7.0
    assert get_min_ev(db) == 7.0
    db.close()


def test_suppresses_bleeding_market(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    detail = _detail("pts", 9.0, "loss", 60) + _detail("reb", 9.0, "win", 60)
    res = run_calibration(db, backtest_fn=_fake_backtest(detail))
    assert "pts" in res["suppressed_markets"]
    assert get_suppressed_markets(db) == ["pts"]
    db.close()


def test_unsuppression_when_market_recovers(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    db.set_setting("suppressed_markets", json.dumps(["reb"]))
    detail = _detail("reb", 9.0, "win", 60)
    res = run_calibration(db, backtest_fn=_fake_backtest(detail))
    assert res["suppressed_markets"] == []
    db.close()


def test_generator_honors_suppressed_market(tmp_path):
    """Suppressed market still renders but is never recommended."""
    from datetime import date, timedelta

    db = ProphitDB(str(tmp_path / "t.db"))
    db.upsert_team(8000, "TST")
    db.upsert_team(8100, "OPP")
    db.upsert_player(8001, "Score Man", team_id=8000)
    d = date(2025, 10, 1)
    for i in range(12):
        db.upsert_player_game(
            {
                "game_id": f"g{i}",
                "player_id": 8001,
                "team_id": 8000,
                "opponent_team_id": 8100,
                "game_date": d.isoformat(),
                "minutes": 32.0,
                "pts": 20,
                "reb": 8,
                "ast": 4,
                "fg3m": 2,
                "is_home": 1,
            }
        )
        d += timedelta(days=2)
    db.insert_consensus_line(
        {
            "event_id": "evt1",
            "game_date": "2025-10-24",
            "home_team_abbr": "TST",
            "away_team_abbr": "OPP",
            "player_id": 8001,
            "player_name": "Score Man",
            "market": "player_points",
            "bookmaker": "CONSENSUS",
            "line": 12.5,
            "over_price": -110,
            "under_price": -110,
            "is_consensus": 1,
            "snapshot_time": "2025-10-24T13:00:00Z",
        }
    )

    plays = SlateGenerator(db).generate_plays("2025-10-24")
    assert plays and plays[0]["recommended_side"] == "over"

    db.set_setting("suppressed_markets", '["pts"]')
    plays = SlateGenerator(db).generate_plays("2025-10-24")
    assert plays[0]["market_suppressed"] is True
    assert plays[0]["recommended_side"] is None
    db.close()
