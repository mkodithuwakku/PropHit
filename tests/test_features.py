"""M2 verification: rolling features, DvP lookups, vacated usage.

Every test runs offline against a temporary SQLite DB. Leakage tests prove
features for a target date never see same-day or future games.
"""

import pytest

from src.features.feature_store import (
    FeatureStore,
    usage_volume,
    walk_forward_splits,
)
from src.features.vacated_usage import VacatedUsageEngine
from src.ingestion.injuries import InjuryEntry

TEAM_ID = 1000
OPP_ID = 2000
PLAYER_ID = 1001

PTS_SEQUENCE = [10, 12, 14, 16, 18, 20, 22, 24, 26, 28]  # oldest -> newest
GAME_DATES = [
    "2025-10-01",
    "2025-10-03",
    "2025-10-05",
    "2025-10-07",
    "2025-10-09",
    "2025-10-11",
    "2025-10-13",
    "2025-10-15",
    "2025-10-17",
    "2025-10-19",
]


def seed_player(db, player_id, name, team_id, team_abbr):
    db.upsert_team(team_id, team_abbr)
    db.upsert_player(player_id, name, team_id=team_id)


def seed_game(db, player_id, game_date, team_id, opp_id=OPP_ID, **stats):
    if opp_id is not None:
        db.upsert_team(opp_id, f"OPP{opp_id}")
    row = {
        "game_id": f"g-{player_id}-{game_date}",
        "player_id": player_id,
        "team_id": team_id,
        "opponent_team_id": opp_id,
        "game_date": game_date,
        "minutes": 30,
        "pts": 0,
        "reb": 0,
        "ast": 0,
        "fg3m": 0,
        "stl": 0,
        "blk": 0,
        "tov": 0,
        "fga": 0,
        "fta": 0,
        "oreb": 0,
        "dreb": 0,
        "plus_minus": 0.0,
    }
    row.update(stats)
    db.upsert_player_game(row)


@pytest.fixture
def seeded_db(db):
    """10 games for PLAYER_ID with an increasing PTS series, constant 34 MIN."""
    seed_player(db, PLAYER_ID, "Alpha Tester", TEAM_ID, "AAA")
    for date, pts in zip(GAME_DATES, PTS_SEQUENCE):
        seed_game(
            db,
            PLAYER_ID,
            date,
            TEAM_ID,
            minutes=34,
            pts=pts,
            fga=10,
            fta=4,
            tov=2,
        )
    return db


# --- rolling windows & EWMA -------------------------------------------------


def test_rolling_window_means(seeded_db):
    feats = FeatureStore(seeded_db).rolling_features(PLAYER_ID, "2025-10-20")
    # most-recent-first series: 28, 26, 24, ...
    assert feats["pts_l3"] == pytest.approx(26.0)
    assert feats["pts_l7"] == pytest.approx(22.0)
    assert feats["pts_l20"] == pytest.approx(19.0)
    assert feats["minutes_l3"] == pytest.approx(34.0)
    assert feats["games_available"] == 10


def test_ewma_weights_recent_games(seeded_db):
    feats = FeatureStore(seeded_db).rolling_features(PLAYER_ID, "2025-10-20")
    recent_first = sorted(PTS_SEQUENCE, reverse=True)
    weights = [0.5 ** (i / 3.0) for i in range(len(recent_first))]
    expected = sum(v * w for v, w in zip(recent_first, weights)) / sum(weights)
    assert feats["pts_ewma"] == pytest.approx(expected)
    # EWMA sits between the hot 3-game mean and the full-sample mean
    assert feats["pts_l20"] < feats["pts_ewma"] < feats["pts_l3"]


def test_per_minute_rates_and_usage_volume(seeded_db):
    feats = FeatureStore(seeded_db).rolling_features(PLAYER_ID, "2025-10-20")
    assert feats["pts_per_min_l20"] == pytest.approx(190 / 340)
    # usage volume per game = fga + 0.44*fta + tov = 10 + 1.76 + 2
    assert feats["usage_volume_l7"] == pytest.approx(13.76)


def test_usage_volume_fta_coefficient():
    game = {"fga": 4, "fta": 10, "tov": 3}
    assert usage_volume(game) == pytest.approx(4 + 4.4 + 3)


# --- zero-leakage guarantees -------------------------------------------------


def test_same_day_game_is_excluded(seeded_db):
    store = FeatureStore(seeded_db)
    before = store.rolling_features(PLAYER_ID, "2025-10-20")

    # a monster game ON the target date must not change anything
    seed_game(seeded_db, PLAYER_ID, "2025-10-20", TEAM_ID, minutes=48, pts=99)
    after = store.rolling_features(PLAYER_ID, "2025-10-20")

    assert after == before
    assert after["pts_l3"] == pytest.approx(26.0)


def test_future_game_is_excluded(seeded_db):
    store = FeatureStore(seeded_db)
    target = "2025-10-10"
    seed_game(
        seeded_db, PLAYER_ID, "2025-10-11", TEAM_ID, minutes=48, pts=99
    )  # future relative to target
    feats = store.rolling_features(PLAYER_ID, target)
    # only games strictly before 10-10: dates 1,3,5,7,9 -> pts 18,16,14,12,10
    assert feats["games_available"] == 5
    assert feats["pts_l3"] == pytest.approx(16.0)
    assert feats["pts_ewma"] is not None


def test_games_before_ordering_and_limit(seeded_db):
    store = FeatureStore(seeded_db)
    games = store.games_before(PLAYER_ID, "2025-10-20", limit=3)
    assert [g["game_date"] for g in games] == [
        "2025-10-19",
        "2025-10-17",
        "2025-10-15",
    ]


def test_walk_forward_splits_are_leak_free():
    games = [
        {"game_id": f"g{i}", "game_date": f"2025-10-{i:02d}"} for i in range(1, 16)
    ]
    splits = list(walk_forward_splits(games, min_train_games=10))
    assert len(splits) == 5
    for train, test in splits:
        test_date = test["game_date"]
        assert all(g["game_date"] < test_date for g in train)
    # expanding window: each train set grows by one
    sizes = [len(train) for train, _ in splits]
    assert sizes == [10, 11, 12, 13, 14]


# --- rest & schedule context -------------------------------------------------


def test_rest_flags_back_to_back(seeded_db):
    feats = FeatureStore(seeded_db).rolling_features(PLAYER_ID, "2025-10-20")
    assert feats["days_rest"] == 0  # played 10-19, game 10-20
    assert feats["is_b2b"] == 1
    assert feats["games_last_4d"] == 2  # 10-17 and 10-19
    assert feats["is_3_in_4"] == 0


def test_rest_flags_normal_and_3_in_4(seeded_db):
    store = FeatureStore(seeded_db)
    rested = store.rolling_features(PLAYER_ID, "2025-10-23")
    assert rested["days_rest"] == 3
    assert rested["is_b2b"] == 0

    # add games on 10-18 and 10-19 -> three games in last 4 days before 10-20
    seed_game(seeded_db, PLAYER_ID, "2025-10-18", TEAM_ID)
    assert store.rolling_features(PLAYER_ID, "2025-10-20")["is_3_in_4"] == 1


def test_no_games_returns_empty_features(db):
    feats = FeatureStore(db).rolling_features(9999, "2025-10-20")
    assert feats["games_available"] == 0
    assert feats["pts_l3"] is None
    assert feats["days_rest"] is None
    assert feats["is_b2b"] == 0


# --- blowout index & altitude ------------------------------------------------


def test_blowout_index(seeded_db):
    ctx = FeatureStore(seeded_db).game_context("2025-10-20", spread=-8.5, total=224.0)
    assert ctx["blowout_index"] == pytest.approx(8.5 * 224.0 / 220.0)
    assert ctx["vegas_spread"] == -8.5
    assert ctx["vegas_total"] == 224.0


def test_altitude_flag(seeded_db):
    seeded_db.upsert_team(3000, "DEN")
    store = FeatureStore(seeded_db)
    away = store.game_context("2025-10-20", opponent_team_id=3000, is_home=False)
    assert away["altitude_game"] == 1
    home = store.game_context("2025-10-20", opponent_team_id=3000, is_home=True)
    assert home["altitude_game"] == 0


# --- DvP matchup vulnerability ----------------------------------------------


def _seed_defense_snapshot(db, snapshot_date, opp_pts_by_team):
    for team_id, opp_pts in opp_pts_by_team.items():
        db.upsert_team(team_id, f"D{team_id}")
        db.upsert_team_defense(
            {
                "team_id": team_id,
                "snapshot_date": snapshot_date,
                "opp_pts": opp_pts,
                "opp_reb": 40 + team_id % 10,
                "opp_ast": 24.0,
                "opp_fg3m": 12.0,
            }
        )


def test_matchup_vulnerability_percentile(seeded_db):
    opp_pts = {2000: 100.0, 2001: 105.0, 2002: 110.0, 2003: 115.0, 2004: 120.0}
    _seed_defense_snapshot(seeded_db, "2025-10-19", opp_pts)
    store = FeatureStore(seeded_db)

    # worst defense (allows most) -> softest matchup
    assert store.matchup_vulnerability(2004, "pts", "2025-10-20") == pytest.approx(1.0)
    assert store.matchup_vulnerability(2002, "pts", "2025-10-20") == pytest.approx(0.6)
    # best defense -> toughest matchup
    assert store.matchup_vulnerability(2000, "pts", "2025-10-20") == pytest.approx(0.2)


def test_matchup_vulnerability_uses_only_prior_snapshots(seeded_db):
    _seed_defense_snapshot(seeded_db, "2025-10-19", {2004: 120.0, 2000: 100.0})
    _seed_defense_snapshot(
        seeded_db, "2025-10-25", {2004: 90.0, 2000: 130.0}
    )  # future snapshot
    store = FeatureStore(seeded_db)
    # on 10-20 only the 10-19 snapshot may be visible
    assert store.matchup_vulnerability(2004, "pts", "2025-10-20") == pytest.approx(1.0)


def test_defense_snapshot_latest_before_date(seeded_db):
    _seed_defense_snapshot(seeded_db, "2025-10-15", {2004: 118.0})
    _seed_defense_snapshot(seeded_db, "2025-10-19", {2004: 120.0})
    snap = FeatureStore(seeded_db).defense_snapshot(2004, "2025-10-20")
    assert snap["snapshot_date"] == "2025-10-19"
    assert snap["opp_pts"] == 120.0


# --- vacated usage -----------------------------------------------------------

TEAM2 = 5000
PA, PB, PC = 5001, 5002, 5003


@pytest.fixture
def rotation_db(db):
    """Team TST: A(30m/usg12), B(20m/usg8), C(10m/usg4), 5 games each."""
    seed_player(db, PA, "Player A", TEAM2, "TST")
    seed_player(db, PB, "Player B", TEAM2, "TST")
    seed_player(db, PC, "Player C", TEAM2, "TST")
    dates = ["2025-10-13", "2025-10-15", "2025-10-17", "2025-10-19"]
    for d in dates:
        seed_game(db, PA, d, TEAM2, minutes=30, fga=12)
        seed_game(db, PB, d, TEAM2, minutes=20, fga=8)
        seed_game(db, PC, d, TEAM2, minutes=10, fga=4)
    return db


def test_team_player_baselines(rotation_db):
    engine = VacatedUsageEngine(rotation_db)
    base = engine.team_player_baselines(TEAM2, "2025-10-20")
    assert base[PA]["minutes_pg"] == pytest.approx(30.0)
    assert base[PA]["usage_pg"] == pytest.approx(12.0)
    assert base[PC]["minutes_pg"] == pytest.approx(10.0)


def test_vacated_usage_redistribution(rotation_db):
    engine = VacatedUsageEngine(rotation_db)
    result = engine.redistribute(TEAM2, [PC], "2025-10-20")

    assert result["vacated_minutes"] == pytest.approx(10.0)
    assert result["vacated_usage"] == pytest.approx(4.0)
    assert result["out_player_ids"] == [PC]

    adj_a = result["adjustments"][PA]
    # minutes follow minutes share: 30/(30+20); usage follows usage share: 12/20
    assert adj_a["extra_minutes"] == pytest.approx(10.0 * 30 / 50)
    assert adj_a["adj_minutes_pg"] == pytest.approx(36.0)
    assert adj_a["extra_usage"] == pytest.approx(4.0 * 12 / 20)
    assert adj_a["usage_boost_pct"] == pytest.approx(0.20)

    adj_b = result["adjustments"][PB]
    assert adj_b["extra_minutes"] == pytest.approx(10.0 * 20 / 50)
    assert adj_b["extra_usage"] == pytest.approx(4.0 * 8 / 20)

    # conservation: everything vacated is fully redistributed
    total_extra_min = sum(a["extra_minutes"] for a in result["adjustments"].values())
    total_extra_usg = sum(a["extra_usage"] for a in result["adjustments"].values())
    assert total_extra_min == pytest.approx(result["vacated_minutes"])
    assert total_extra_usg == pytest.approx(result["vacated_usage"])


def test_vacated_usage_is_leak_free(rotation_db):
    engine = VacatedUsageEngine(rotation_db)
    baseline = engine.redistribute(TEAM2, [PC], "2025-10-20")

    # monster games ON the target date must not change redistribution
    seed_game(rotation_db, PC, "2025-10-20", TEAM2, minutes=48, fga=40)
    seed_game(rotation_db, PA, "2025-10-20", TEAM2, minutes=48, fga=40)
    after = engine.redistribute(TEAM2, [PC], "2025-10-20")

    assert after["vacated_minutes"] == baseline["vacated_minutes"]
    assert after["adjustments"] == baseline["adjustments"]


def test_resolve_out_players_from_injury_report(rotation_db):
    engine = VacatedUsageEngine(rotation_db)
    entries = [
        InjuryEntry(player_name="Player C", status="out", team_abbr="TST"),
        InjuryEntry(player_name="Player B", status="questionable", team_abbr="TST"),
        InjuryEntry(player_name="Player A", status="out", team_abbr="DEN"),
    ]
    out_ids = engine.resolve_out_players(entries, TEAM2)
    assert out_ids == {PC}  # only OUT on this team; DEN entry ignored


def test_redistribute_with_unknown_out_player(rotation_db):
    engine = VacatedUsageEngine(rotation_db)
    result = engine.redistribute(TEAM2, [999999], "2025-10-20")
    assert result["vacated_minutes"] == 0.0
    assert result["adjustments"][PA]["extra_minutes"] == 0.0


# --- advanced efficiency stats -------------------------------------------------


@pytest.fixture
def rich_db(db):
    """10 fully-statted games for PLAYER_ID (constant 34 MIN)."""
    seed_player(db, PLAYER_ID, "Rich Tester", TEAM_ID, "AAA")
    for i, d in enumerate(GAME_DATES):
        seed_game(
            db,
            PLAYER_ID,
            d,
            TEAM_ID,
            minutes=34,
            pts=10 + 2 * i,
            fgm=4 + (i % 3),
            fga=10,
            fg3m=2,
            fg3a=5,
            ftm=3,
            fta=4,
            oreb=3,
            dreb=9,
            reb=12,
            ast=7,
            stl=1,
            blk=1,
            tov=2,
            pf=2,
            plus_minus=4.0,
        )
    return db


def test_ts_pct_efg_and_rate_ratios(rich_db):
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    # L7 sums: pts 28+26+24+22+20+18+16=154, fga 70, fta 28
    assert feats["ts_pct_l7"] == pytest.approx(154 / (2 * (70 + 0.44 * 28)))
    assert feats["ts_pct_l20"] == pytest.approx(190 / (2 * (100 + 0.44 * 40)))
    assert feats["ts_pct_ewma"] is not None
    # L3: 78 pts, denom 2*(30 + 0.44*12)
    l3_ts = 78 / (2 * (30 + 0.44 * 12))
    assert feats["ts_pct_l3"] == pytest.approx(l3_ts)
    assert feats["ts_regression_delta"] == pytest.approx(l3_ts - feats["ts_pct_l20"])
    # efg L20: (fgm_sum + 0.5*20) / 100 ; fgm per game = 4+(i%3) -> sum 4*10+15=55? compute directly
    fgm_sum = sum(4 + (i % 3) for i in range(10))
    assert feats["efg_pct_l20"] == pytest.approx((fgm_sum + 0.5 * 20) / 100)
    assert feats["ftr_l20"] == pytest.approx(40 / 100)
    assert feats["fg3a_rate_l20"] == pytest.approx(50 / 100)
    assert feats["fg3a_per_min_l20"] == pytest.approx(50 / 340)
    assert feats["fga_per_min_l20"] == pytest.approx(100 / 340)
    assert feats["fta_per_min_l20"] == pytest.approx(40 / 340)


def test_ts_pct_neutral_when_no_shots(db):
    seed_player(db, PLAYER_ID, "No Shot", TEAM_ID, "AAA")
    seed_game(db, PLAYER_ID, "2025-10-19", TEAM_ID, minutes=12, fga=0, fta=0)
    feats = FeatureStore(db).rolling_features(PLAYER_ID, "2025-10-20")
    assert feats["ts_pct_l7"] == pytest.approx(0.55)
    assert feats["ts_pct_l20"] == pytest.approx(0.55)
    assert feats["efg_pct_l20"] is None  # zero FGA -> undefined, not neutral


def test_extended_per_minute_rates(rich_db):
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    for stat, total in (
        ("stl", 10),
        ("blk", 10),
        ("tov", 20),
        ("oreb", 30),
        ("dreb", 90),
        ("fga", 100),
        ("fta", 40),
    ):
        assert feats[f"{stat}_per_min_l20"] == pytest.approx(total / 340)
        assert feats[f"{stat}_per_min_ewma"] is not None


# --- on-floor shares (canonical team fallback) ----------------------------------


def test_on_floor_shares_use_canonical_baseline(rich_db):
    """Single-player team -> canonical 240min/88fga/22fta/14tov/44reb team."""
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    # usage = 10 + 0.44*4 + 2 = 13.76; team usage = 88 + 9.68 + 14 = 111.68
    exp_usg = 100 * 13.76 * 48 / (34 * 111.68)
    assert feats["usg_pct_l20"] == pytest.approx(exp_usg)
    assert feats["usg_pct_l7"] == pytest.approx(exp_usg)
    assert feats["usg_pct_ewma"] == pytest.approx(exp_usg)
    # reb share: 100*12*48/(34*44); oreb: 100*3*48/(34*10); dreb: 100*9*48/(34*34)
    assert feats["reb_pct_l20"] == pytest.approx(100 * 12 * 48 / (34 * 44))
    assert feats["oreb_pct_l20"] == pytest.approx(100 * 3 * 48 / (34 * 10))
    assert feats["dreb_pct_l20"] == pytest.approx(100 * 9 * 48 / (34 * 34))
    # ast share of teammate FGM: team fgm 42 - player fgm(4+i%3)
    fgm_sum = sum(4 + (i % 3) for i in range(10))
    exp_ast = sum(100 * 7 * 48 / (34 * (42 - (4 + (i % 3)))) for i in range(10)) / 10
    assert feats["ast_pct_l20"] == pytest.approx(exp_ast)
    assert feats["tov_pct_l20"] == pytest.approx(100 * 20 / (13.76 * 10))
    assert feats["ast_to_tov_l20"] == pytest.approx(70 / 20)


def test_role_expansion_and_minutes_trend(rich_db):
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    # constant workload -> zero deltas
    assert feats["role_expansion_delta"] == pytest.approx(0.0)
    assert feats["minutes_trend_delta"] == pytest.approx(0.0)
    # add a heavy recent stretch: 40 min, 20 fga on the last 3 dates
    for d in GAME_DATES[-3:]:
        seed_game(rich_db, PLAYER_ID, d, TEAM_ID, minutes=40, fga=20)
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    assert feats["minutes_trend_delta"] > 0
    assert feats["role_expansion_delta"] > 0


# --- composite impact & volatility ----------------------------------------------


def test_game_score_pie_and_volatility(rich_db):
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    assert feats["game_score_l7"] is not None
    assert feats["game_score_l20"] is not None
    assert feats["game_score_ewma"] is not None
    # pts rise over the series -> recent game scores exceed long-run mean
    assert feats["game_score_l7"] > feats["game_score_l20"]
    assert feats["minutes_std_l20"] == pytest.approx(0.0)  # constant 34
    assert feats["pts_cv_l20"] > 0
    assert feats["reb_cv_l20"] == pytest.approx(0.0)  # constant 12
    assert feats["foul_rate_per36_l20"] == pytest.approx(36 * 20 / 340)
    assert feats["plus_minus_per_min_l20"] == pytest.approx(40 / 340)
    assert feats["pie_l20"] > 0


def test_minutes_std_varies(rich_db):
    for i, d in enumerate(GAME_DATES):
        seed_game(rich_db, PLAYER_ID, d, TEAM_ID, minutes=20 + 2 * i)
    feats = FeatureStore(rich_db).rolling_features(PLAYER_ID, "2025-10-20")
    import statistics as _st

    assert feats["minutes_std_l20"] == pytest.approx(
        _st.stdev([20 + 2 * i for i in range(10)])
    )


def test_advanced_stats_are_leak_free(rich_db):
    store = FeatureStore(rich_db)
    before = store.rolling_features(PLAYER_ID, "2025-10-20")
    seed_game(
        rich_db,
        PLAYER_ID,
        "2025-10-20",
        TEAM_ID,
        minutes=48,
        pts=99,
        fgm=20,
        fga=30,
        stl=9,
        pf=6,
        plus_minus=40.0,
    )
    after = store.rolling_features(PLAYER_ID, "2025-10-20")
    assert after == before


# --- team pace & expected pace factor -------------------------------------------


def test_team_pace_estimate(rich_db):
    # single-player "team": poss = 10 + 0.44*4 - 3 + 2 = 10.76 per game
    pace = FeatureStore(rich_db).team_pace(TEAM_ID, "2025-10-20")
    assert pace == pytest.approx(10 + 0.44 * 4 - 3 + 2)


def test_expected_pace_factor_neutral_for_synthetic(rich_db):
    assert (
        FeatureStore(rich_db).expected_pace_factor(TEAM_ID, OPP_ID, "2025-10-20") == 1.0
    )


def test_expected_pace_factor_multi_team(db):
    """Three teams (3 players each) with different possession volumes."""
    for team_id, fga in ((7001, 15), (7002, 8), (7003, 12)):
        for p in range(3):
            pid = team_id * 10 + p
            seed_player(db, pid, f"P{pid}", team_id, f"T{team_id}")
            for d in ["2025-10-13", "2025-10-15", "2025-10-17", "2025-10-19"]:
                seed_game(
                    db,
                    pid,
                    d,
                    team_id,
                    game_id=f"tg-{team_id}-{d}",  # teammates share a game
                    minutes=40,
                    fga=fga,
                    fta=2,
                    oreb=1,
                    tov=2,
                )
    store = FeatureStore(db)
    # team poss/game: 3*fga + 0.44*6 - 3 + 6
    assert store.team_pace(7001, "2025-10-20") == pytest.approx(45 + 2.64 - 3 + 6)
    fast = store.expected_pace_factor(7001, 7003, "2025-10-20")
    slow = store.expected_pace_factor(7002, 7003, "2025-10-20")
    assert 0.92 <= slow < fast <= 1.08


def test_expected_pace_factor_leak_free(db):
    """A monster same-day game must not change the pace factor."""
    for team_id in (8001, 8002, 8003):
        for p in range(3):
            pid = team_id * 10 + p
            seed_player(db, pid, f"P{pid}", team_id, f"T{team_id}")
            for d in ["2025-10-15", "2025-10-17", "2025-10-19"]:
                seed_game(
                    db,
                    pid,
                    d,
                    team_id,
                    game_id=f"lg-{team_id}-{d}",
                    minutes=40,
                    fga=10 + (team_id % 3),
                    fta=2,
                    tov=2,
                )
    store = FeatureStore(db)
    before = store.expected_pace_factor(8001, 8002, "2025-10-20")
    for p in range(3):
        seed_game(
            db,
            8001 * 10 + p,
            "2025-10-20",
            8001,
            game_id="lg-8001-x",
            minutes=48,
            fga=60,
            tov=30,
        )
    assert store.expected_pace_factor(8001, 8002, "2025-10-20") == before


# --- position-aware & combo matchup vulnerability --------------------------------


def test_matchup_vulnerability_position_scope(db):
    for team_id, all_pts, c_pts in ((2000, 100.0, 140.0), (2001, 110.0, 130.0)):
        db.upsert_team(team_id, f"D{team_id}")
        db.upsert_team_defense(
            {
                "team_id": team_id,
                "snapshot_date": "2025-10-19",
                "opp_pts": all_pts,
                "position_scope": "ALL",
            }
        )
        db.upsert_team_defense(
            {
                "team_id": team_id,
                "snapshot_date": "2025-10-19",
                "opp_pts": c_pts,
                "position_scope": "C",
            }
        )
    store = FeatureStore(db)
    assert store.matchup_vulnerability(2001, "pts", "2025-10-20") == pytest.approx(1.0)
    # C scope flips the ordering: 2001 allows 130 < 2000's 140
    assert store.matchup_vulnerability(
        2001, "pts", "2025-10-20", position="C"
    ) == pytest.approx(0.5)
    # unseeded scope falls back to ALL
    assert store.matchup_vulnerability(
        2001, "pts", "2025-10-20", position="PG"
    ) == pytest.approx(1.0)


def test_matchup_vulnerability_combos(seeded_db):
    _seed_defense_snapshot(seeded_db, "2025-10-19", {2000: 100.0, 2004: 120.0})
    store = FeatureStore(seeded_db)
    components = [
        store.matchup_vulnerability(2004, s, "2025-10-20")
        for s in ("pts", "reb", "ast")
    ]
    combo = store.matchup_vulnerability(2004, "pts_reb_ast", "2025-10-20")
    assert combo == pytest.approx(sum(components) / 3)
    assert store.matchup_vulnerability(2004, "pra", "2025-10-20") == pytest.approx(
        combo
    )


def test_derived_stl_vulnerability(db):
    """stl/blk/tov have no defense column -> derived from game logs."""
    seed_player(db, 9101, "S1", 9100, "T9100")
    seed_player(db, 9102, "S2", 9200, "T9200")
    seed_player(db, 9103, "S3", 9300, "T9300")
    dates = ["2025-10-13", "2025-10-15", "2025-10-17"]
    for d in dates:
        seed_game(db, 9101, d, 9100, opp_id=8001, stl=0, minutes=30)
        seed_game(db, 9102, d, 9200, opp_id=8002, stl=4, minutes=30)
        seed_game(db, 9103, d, 9300, opp_id=8003, stl=8, minutes=30)
    store = FeatureStore(db)
    assert store.matchup_vulnerability(8003, "stl", "2025-10-20") == pytest.approx(1.0)
    assert store.matchup_vulnerability(8001, "stl", "2025-10-20") == pytest.approx(
        1 / 3
    )
    # same-day games are invisible to the derivation
    seed_game(db, 9103, "2025-10-20", 9300, opp_id=8003, stl=99, minutes=48)
    assert store.matchup_vulnerability(8003, "stl", "2025-10-20") == pytest.approx(1.0)


def test_scoped_vulnerability_falls_back_to_all(db):
    """Opponent missing from a position scope -> ALL-scope percentile."""
    db.upsert_team(2000, "D0")
    db.upsert_team(2001, "D1")
    db.upsert_team_defense(
        {
            "team_id": 2000,
            "snapshot_date": "2025-10-19",
            "opp_pts": 100.0,
            "position_scope": "ALL",
        }
    )
    db.upsert_team_defense(
        {
            "team_id": 2001,
            "snapshot_date": "2025-10-19",
            "opp_pts": 110.0,
            "position_scope": "ALL",
        }
    )
    # C scope exists for the snapshot but only covers team 2000
    db.upsert_team_defense(
        {
            "team_id": 2000,
            "snapshot_date": "2025-10-19",
            "opp_pts": 140.0,
            "position_scope": "C",
        }
    )
    store = FeatureStore(db)
    # 2001 missing from C scope -> falls back to ALL (softest matchup)
    assert store.matchup_vulnerability(
        2001, "pts", "2025-10-20", position="C"
    ) == pytest.approx(1.0)
    # 2000 is present in C scope -> resolves there
    assert store.matchup_vulnerability(
        2000, "pts", "2025-10-20", position="C"
    ) == pytest.approx(1.0)
