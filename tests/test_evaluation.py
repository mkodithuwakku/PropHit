"""Model-record verification: settlement math, ROI, accuracy, ID promotion.

All offline — reconstructed recommendations run through the same leak-free
SlateGenerator path used by the morning pipeline.
"""

from datetime import date, timedelta

import pytest

from src.evaluation.record import (
    settled_plays,
    summarize_model_stats,
)
from src.ingestion.consensus_odds import ConsensusOddsIngestion
from src.publication.daily_refresh import _season_for

P_TEAM = 8000
P_OPP = 8100
P_ID = 8001
SNAPSHOT = "2025-10-07T13:00:00Z"


def _game(i, d, reb=12):
    return {
        "game_id": f"s{i}",
        "player_id": P_ID,
        "team_id": P_TEAM,
        "opponent_team_id": P_OPP,
        "game_date": d,
        "minutes": 30.0,
        "pts": 18,
        "reb": reb,
        "ast": 4,
        "fg3m": 1,
        "fga": 12,
        "fta": 4,
        "tov": 2,
        "is_home": 1,
    }


def _line(game_date, market="player_rebounds", line=9.5, pid=P_ID):
    return {
        "event_id": f"evt_{game_date}",
        "game_date": game_date,
        "home_team_abbr": "TST",
        "away_team_abbr": "OPP",
        "player_id": pid,
        "player_name": "Board Man",
        "market": market,
        "bookmaker": "CONSENSUS",
        "line": line,
        "over_price": -110,
        "under_price": -110,
        "is_consensus": 1,
        "snapshot_time": SNAPSHOT,
        "commence_time": f"{game_date}T23:00:00Z",
    }


def _seed(db):
    """24-game history (12 REB avg) from Sep 8; lines on game days
    Oct 7/13/15 with varied actuals: win (14), loss (6), push (12 on 12.0)."""
    db.upsert_team(P_TEAM, "TST")
    db.upsert_team(P_OPP, "OPP")
    db.upsert_player(P_ID, "Board Man", team_id=P_TEAM, position="C")
    specials = {"2025-10-08": 14, "2025-10-14": 6}
    d = date(2025, 9, 8)
    i = 0
    while d <= date(2025, 10, 23):
        db.upsert_player_game(_game(i, d.isoformat(), specials.get(str(d), 12)))
        i += 1
        d += timedelta(days=2)
    db.insert_consensus_line(_line("2025-10-08", line=9.5))
    db.insert_consensus_line(_line("2025-10-14", line=9.5))
    db.insert_consensus_line(_line("2025-10-16", line=12.0))
    return db


@pytest.fixture
def settled_db(db):
    return _seed(db)


def test_settled_plays_grades_results(settled_db):
    records = {r["game_date"]: r for r in settled_plays(settled_db)}
    assert records["2025-10-08"]["result"] == "win"  # 14 > 9.5 over
    assert records["2025-10-08"]["units"] == pytest.approx(0.9091, abs=0.001)
    assert records["2025-10-14"]["result"] == "loss"  # 6 < 9.5 over
    assert records["2025-10-14"]["units"] == -1.0
    assert records["2025-10-16"]["result"] == "push"  # 12 == 12.0
    assert records["2025-10-16"]["units"] == 0.0


def test_summarize_model_stats_roi_math(settled_db):
    stats = summarize_model_stats(settled_db, through_date="2025-10-23")
    rec = stats["record"]
    assert rec["bets"] == 3
    assert rec["wins"] == 1
    assert rec["losses"] == 1
    assert rec["pushes"] == 1
    assert rec["units"] == pytest.approx(-0.09, abs=0.01)
    assert rec["roi_pct"] == pytest.approx(-3.0, abs=0.2)
    assert stats["settled_through"] == "2025-10-16"


def test_accuracy_favored_hit_rate(settled_db):
    stats = summarize_model_stats(settled_db, through_date="2025-10-23")
    acc = stats["accuracy"]
    # 3 evaluated plays; model favored over each time -> 2 hits (14, 12>…wait
    # 12.0 pushes don't count as favored decision) -> decided = 2, hits = 1
    assert acc["sample"] == 2
    assert acc["favored_hit_rate"] == 50.0


def test_empty_db_returns_zeroed_record(db):
    stats = summarize_model_stats(db)
    assert stats["record"]["bets"] == 0
    assert stats["record"]["roi_pct"] is None
    assert stats["accuracy"]["sample"] == 0


def test_season_for():
    assert _season_for("2025-10-23") == "2025-26"
    assert _season_for("2026-03-15") == "2025-26"
    assert _season_for("2026-07-04") == "2026-27"


def test_stub_player_promoted_to_real_id(db, load_fixture):
    """PrizePicks stub -> real NBA ID merges children + dim row."""
    ConsensusOddsIngestion(db).ingest_prizepicks(
        load_fixture("prizepicks_projections.json"), snapshot_time=SNAPSHOT
    )
    stub_id = db.player_id_for_name("Nikola Jokic")
    assert stub_id != 203999  # synthetic stub

    db.upsert_player(203999, "Nikola Jokic", position="C")

    assert db.player_id_for_name("Nikola Jokic") == 203999
    assert (
        db.query_one(
            "SELECT COUNT(*) AS n FROM dim_players "
            "WHERE normalized_name = 'nikola jokic'"
        )["n"]
        == 1
    )  # no duplicate rows
    line = db.query_one(
        "SELECT player_id FROM fact_consensus_lines "
        "WHERE player_name = 'Nikola Jokic' LIMIT 1"
    )
    assert line["player_id"] == 203999  # children repointed
