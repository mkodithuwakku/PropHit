"""M5/M6 verification: slate generation, static bundle, publish dry-run.

The end-to-end path is exercised offline: seeded SQLite -> play cards ->
daily_slate.json + web bundle -> publisher command plan (no git/network).
"""

import json
from datetime import date, timedelta

import pytest

from src.ingestion.db import ProphitDB
from src.publication.daily_refresh import run_refresh
from src.publication.publisher import GitPublisher
from src.publication.static_builder import (
    SlateGenerator,
    StaticSiteBuilder,
)

P_TEAM = 8000
P_OPP = 8100
P_ID = 8001
SNAPSHOT = "2025-10-22T13:00:00Z"
GAME_DATE = "2025-10-22"


def _seed_slate_db(db):
    """Player averaging 14 REB in 30 MIN; consensus line REB 9.5 -110."""
    db.upsert_team(P_TEAM, "TST")
    db.upsert_team(P_OPP, "OPP")
    db.upsert_player(P_ID, "Board Man", team_id=P_TEAM, position="C")
    d = date(2025, 10, 1)
    for i in range(12):
        db.upsert_player_game(
            {
                "game_id": f"s{i}",
                "player_id": P_ID,
                "team_id": P_TEAM,
                "opponent_team_id": P_OPP,
                "game_date": d.isoformat(),
                "minutes": 30.0,
                "pts": 18,
                "reb": 14,
                "ast": 4,
                "fg3m": 1,
                "fga": 12,
                "fta": 4,
                "tov": 2,
                "is_home": 1,
            }
        )
        d += timedelta(days=2)
    db.insert_consensus_line(
        {
            "event_id": "evt_tst_opp",
            "game_date": GAME_DATE,
            "home_team_abbr": "TST",
            "away_team_abbr": "OPP",
            "player_id": P_ID,
            "player_name": "Board Man",
            "market": "player_rebounds",
            "bookmaker": "CONSENSUS",
            "line": 9.5,
            "over_price": -110,
            "under_price": -110,
            "is_consensus": 1,
            "snapshot_time": SNAPSHOT,
            "commence_time": "2025-10-23T00:00:00Z",
        }
    )
    return db


@pytest.fixture
def slate_db(db):
    return _seed_slate_db(db)


# --- slate generation ---------------------------------------------------------


def test_generate_plays_end_to_end(slate_db):
    plays = SlateGenerator(slate_db).generate_plays(GAME_DATE)
    assert len(plays) == 1
    play = plays[0]

    assert play["player_name"] == "Board Man"
    assert play["team_abbr"] == "TST"
    assert play["opponent_abbr"] == "OPP"
    assert play["is_home"] is True
    assert play["stat"] == "reb"
    assert play["line"] == 9.5
    # 14 reb/game on ~30 min, shrunk toward line: (12*14 + 6*9.5)/18 = 12.5
    assert play["model_mean"] == pytest.approx(12.5, abs=0.5)
    assert play["p_over"] > 0.6
    assert play["recommended_side"] == "over"
    assert play["recommended_ev_pct"] > 5.0

    # factor tray fields the UI renders
    f = play["factors"]
    assert f["projected_minutes"] == pytest.approx(30.0, abs=1.0)
    assert f["rest_label"] == "Back-to-back"  # played 10-21, game 10-22
    assert f["rest_days"] == 0

    # milestone ladder triggered (edge ~2.5 >= 2.0) with even-number rungs
    m = play["milestone"]
    assert m["triggered"] is True
    assert [r["k"] for r in m["milestones"]] == [10, 12, 14]
    assert all("fair_american" in r for r in m["milestones"])


def test_generate_plays_combo_stat(slate_db):
    slate_db.insert_consensus_line(
        {
            "event_id": "evt_tst_opp",
            "game_date": GAME_DATE,
            "home_team_abbr": "TST",
            "away_team_abbr": "OPP",
            "player_id": P_ID,
            "player_name": "Board Man",
            "market": "player_points_rebounds_assists",
            "bookmaker": "CONSENSUS",
            "line": 30.5,
            "over_price": -115,
            "under_price": -105,
            "is_consensus": 1,
            "snapshot_time": SNAPSHOT,
            "commence_time": "2025-10-23T00:00:00Z",
        }
    )
    plays = {p["market"]: p for p in SlateGenerator(slate_db).generate_plays(GAME_DATE)}
    pra = plays["player_points_rebounds_assists"]
    assert pra["stat"] == "pra"
    # PRA mean ≈ 18 + 14 + 4 = 36 -> shrunk (12*36 + 6*30.5)/18 ≈ 34.2
    assert pra["model_mean"] == pytest.approx(34.2, abs=1.0)


def test_generate_plays_skips_unknown_players(slate_db):
    # stub player (as ingestion would create) with zero game history
    slate_db.upsert_player(999999, "Ghost Player", team_id=P_TEAM)
    slate_db.insert_consensus_line(
        {
            "event_id": "evt_tst_opp",
            "game_date": GAME_DATE,
            "home_team_abbr": "TST",
            "away_team_abbr": "OPP",
            "player_id": 999999,
            "player_name": "Ghost Player",
            "market": "player_points",
            "bookmaker": "CONSENSUS",
            "line": 10.5,
            "over_price": -110,
            "under_price": -110,
            "is_consensus": 1,
            "snapshot_time": SNAPSHOT,
        }
    )
    names = [
        p["player_name"] for p in SlateGenerator(slate_db).generate_plays(GAME_DATE)
    ]
    assert "Ghost Player" not in names


def test_ordinal_labels():
    assert SlateGenerator._ordinal(1) == "1st"
    assert SlateGenerator._ordinal(22) == "22nd"
    assert SlateGenerator._ordinal(23) == "23rd"
    assert SlateGenerator._ordinal(28) == "28th"


# --- static bundle -------------------------------------------------------------


def test_static_builder_writes_bundle(tmp_path, slate_db):
    plays = SlateGenerator(slate_db).generate_plays(GAME_DATE)
    out = tmp_path / "site"
    result = StaticSiteBuilder(out).build({"date": GAME_DATE, "plays": plays})

    assert (out / "daily_slate.json").exists()
    for asset in ("index.html", "styles.css", "app.js"):
        assert (out / asset).exists()
    assert result["total_plays"] == 1
    assert result["recommended_count"] == 1
    assert result["milestone_count"] == 1

    slate = json.loads((out / "daily_slate.json").read_text())
    assert slate["date"] == GAME_DATE
    assert slate["plays"][0]["player_name"] == "Board Man"
    # required card fields the JS renders
    for key in (
        "player_name",
        "line",
        "model_mean",
        "p_over",
        "ev_over_pct",
        "factors",
        "milestone",
    ):
        assert key in slate["plays"][0]


def test_index_html_is_mobile_first(tmp_path):
    StaticSiteBuilder(tmp_path / "site").build({"date": GAME_DATE, "plays": []})
    html = (tmp_path / "site" / "index.html").read_text()
    assert 'name="viewport"' in html
    assert "width=device-width" in html
    assert "daily_slate.json" not in html or "app.js" in html
    css = (tmp_path / "site" / "styles.css").read_text()
    assert "max-width: 430px" in css  # 390-430px primary layout (SRD 7.1)


# --- daily refresh & publish -----------------------------------------------------


def test_daily_refresh_dry_run_end_to_end(tmp_path, slate_db):
    db_path = tmp_path / "prophit.db"
    seeded = _seed_slate_db(ProphitDB(db_path))
    seeded.close()

    result = run_refresh(
        db_path=db_path,
        out_dir=tmp_path / "site",
        game_date=GAME_DATE,
        publish=True,
        dry_run=True,
    )
    assert result["plays"] == 1
    assert result["recommended"] == 1
    assert set(result["files"]) == {
        "daily_slate.json",
        "index.html",
        "styles.css",
        "app.js",
    }
    # publish was requested but dry-run -> command plan only, no git ops
    assert result["publish"]["dry_run"] is True
    assert any("gh-pages" in c for c in result["publish"]["commands"])


def test_publisher_dry_run_no_side_effects(tmp_path):
    pub = GitPublisher(repo_path=tmp_path)
    plan = pub.publish(tmp_path / "site", "test msg", dry_run=True)
    assert plan["dry_run"] is True
    assert plan["branch"] == "gh-pages"
    assert any("git push" in c for c in plan["commands"])
