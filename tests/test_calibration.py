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


def _seed_settled_day(db, game_date, pid=1):
    db.upsert_team(8000, "TST")
    db.upsert_team(8100, "OPP")
    db.upsert_player(pid, "Settle Man", team_id=8000)
    db.upsert_player_game(
        {
            "game_id": f"settled-{pid}-{game_date}",
            "player_id": pid,
            "team_id": 8000,
            "opponent_team_id": 8100,
            "game_date": game_date,
            "minutes": 30.0,
            "pts": 18,
        }
    )
    db.insert_consensus_line(
        {
            "event_id": f"evt-{game_date}",
            "game_date": game_date,
            "home_team_abbr": "TST",
            "away_team_abbr": "OPP",
            "player_id": pid,
            "player_name": "Settle Man",
            "market": "player_points",
            "bookmaker": "CONSENSUS",
            "line": 18.5,
            "over_price": -110,
            "under_price": -110,
            "is_consensus": 1,
            "snapshot_time": f"{game_date}T12:00:00Z",
        }
    )


def test_early_trigger_when_enough_days_settled(tmp_path):
    """<14d since last run but >=3 new settled game_dates -> still runs."""
    db = ProphitDB(str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    db.set_setting("last_calibration", (now - timedelta(days=3)).isoformat())
    for back in (1, 2, 3):
        _seed_settled_day(db, (now - timedelta(days=back)).date().isoformat())

    res = run_calibration(db, backtest_fn=_fake_backtest([]), min_new_settled_days=3)
    assert res["ran"] is True
    db.close()


def test_no_early_trigger_below_settled_threshold(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))
    now = datetime.now(timezone.utc)
    db.set_setting("last_calibration", (now - timedelta(days=3)).isoformat())
    for back in (1, 2):
        _seed_settled_day(db, (now - timedelta(days=back)).date().isoformat())

    res = run_calibration(db, backtest_fn=_fake_backtest([]), min_new_settled_days=3)
    assert res["ran"] is False
    db.close()


def test_multi_parameter_sweep_persists_and_clamps(tmp_path):
    """prior_games swept to best ladder value, clamped ±2 per run."""
    db = ProphitDB(str(tmp_path / "t.db"))

    def prior_favoring_backtest(db, **kw):
        pg = kw.get("prior_games")
        score = {10.0: 62.0, 8.0: 55.0}.get(pg, 50.0)
        return {
            "overall": {"favored_hit_pct": score},
            "per_stat": {},
            "bets_detail": [],
        }

    res = run_calibration(db, backtest_fn=prior_favoring_backtest)
    assert res["ran"] is True
    # 10.0 scored best, but the ±2 drift clamp caps the move at 6+2=8
    assert float(db.get_setting("prior_games")) == 8.0
    assert any("prior_games 6.0->8.0" in n for n in res["notes"])
    # flat-scored params stay at their calibrated defaults
    assert float(db.get_setting("ml_blend_weight")) == 0.55
    assert float(db.get_setting("dvp_weight")) == 0.08
    assert float(db.get_setting("mc_blend_weight")) == 0.25
    db.close()


def test_multi_parameter_sweep_updates_weights(tmp_path):
    db = ProphitDB(str(tmp_path / "t.db"))

    def dvp_favoring_backtest(db, **kw):
        score = 58.0 if kw.get("dvp_weight") == 0.12 else 50.0
        return {
            "overall": {"favored_hit_pct": score},
            "per_stat": {},
            "bets_detail": [],
        }

    run_calibration(db, backtest_fn=dvp_favoring_backtest)
    assert float(db.get_setting("dvp_weight")) == 0.12
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
