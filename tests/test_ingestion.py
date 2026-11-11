"""M1 verification: ingestion, parsing, and storage work 100% offline.

pytest-socket blocks all socket usage (see pytest.ini + conftest), so any
accidental live call to nba_api, PrizePicks, or Underdog fails loudly.
"""

import pytest

from src.ingestion.consensus_odds import (
    CONSENSUS_BOOKMAKER,
    ConsensusOddsIngestion,
    PrizePicksClient,
    StatPickClient,
    UnderdogClient,
)
from src.ingestion.injuries import parse_injury_entries
from src.ingestion.nba_stats import (
    ESPNBoxScoreClient,
    NBAStatsClient,
    PlayerGameLogIngestion,
    TeamDefenseIngestion,
    nba_date_param,
    parse_game_date,
    parse_matchup,
    parse_minutes,
)

SNAPSHOT = "2025-10-21T13:35:00Z"


# --- pure parsing helpers ---------------------------------------------------


def test_parse_minutes_formats():
    assert parse_minutes("34:22") == pytest.approx(34 + 22 / 60)
    assert parse_minutes(36.5) == 36.5
    assert parse_minutes("12") == 12.0
    assert parse_minutes(None) is None
    assert parse_minutes("") is None


def test_parse_matchup_home_away():
    assert parse_matchup("DEN @ LAL") == ("LAL", False)
    assert parse_matchup("DEN vs. UTA") == ("UTA", True)
    assert parse_matchup(None) == (None, None)


def test_parse_game_date_formats():
    assert parse_game_date("OCT 22, 2025") == "2025-10-22"
    assert parse_game_date("2025-10-26") == "2025-10-26"
    assert parse_game_date("10/22/2025") == "2025-10-22"
    assert parse_game_date(None) is None


# --- schema -----------------------------------------------------------------


def test_schema_creates_all_spec_tables(db):
    tables = {
        row["name"]
        for row in db.query("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    assert {
        "dim_players",
        "dim_teams",
        "fact_player_games",
        "fact_team_defense_daily",
        "fact_consensus_lines",
        "fact_model_evaluations",
    } <= tables


# --- NBA stats ingestion ----------------------------------------------------


def test_ingest_player_game_log_fixture(db, load_fixture):
    payload = load_fixture("nba_player_game_log.json")
    ingestion = PlayerGameLogIngestion(db)

    written = ingestion.ingest_rows(payload["PlayerGameLog"])

    assert written == 3  # the row missing Game_ID is skipped
    games = db.query("SELECT * FROM fact_player_games ORDER BY game_date")
    assert len(games) == 3

    opener = games[0]
    assert opener["player_id"] == 203999
    assert opener["game_date"] == "2025-10-22"
    assert opener["is_home"] == 0  # DEN @ LAL
    assert opener["minutes"] == pytest.approx(34 + 22 / 60)
    assert opener["pts"] == 28
    assert opener["reb"] == 15
    assert opener["ast"] == 9

    home_game = games[1]
    assert home_game["is_home"] == 1  # DEN vs. UTA
    assert home_game["minutes"] == pytest.approx(31 + 5 / 60)

    numeric_min_game = games[2]
    assert numeric_min_game["minutes"] == 36.5


def test_ingest_game_log_populates_dimensions(db, load_fixture):
    payload = load_fixture("nba_player_game_log.json")
    PlayerGameLogIngestion(db).ingest_rows(payload["PlayerGameLog"])

    player = db.query_one("SELECT * FROM dim_players WHERE player_id = 203999")
    assert player["full_name"] == "Nikola Jokic"
    assert player["normalized_name"] == "nikola jokic"

    denver = db.query_one("SELECT * FROM dim_teams WHERE abbreviation = 'DEN'")
    assert denver["team_id"] == 1610612743

    # Opponents resolved from MATCHUP strings get deterministic synthetic IDs
    for abbr in ("LAL", "UTA", "PHX"):
        assert db.query_one(
            "SELECT team_id FROM dim_teams WHERE abbreviation = ?", (abbr,)
        )


def test_game_log_ingestion_is_idempotent(db, load_fixture):
    rows = load_fixture("nba_player_game_log.json")["PlayerGameLog"]
    ingestion = PlayerGameLogIngestion(db)
    ingestion.ingest_rows(rows)
    ingestion.ingest_rows(rows)
    assert db.query_one("SELECT COUNT(*) AS n FROM fact_player_games")["n"] == 3


def test_nba_client_calls_endpoint_and_retries(db, load_fixture):
    payload = load_fixture("nba_player_game_log.json")
    calls = []
    attempts = {"n": 0}

    def flaky_caller(endpoint_name, params):
        calls.append((endpoint_name, params))
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise TimeoutError("simulated transient failure")
        return payload

    sleeps = []
    client = NBAStatsClient(endpoint_caller=flaky_caller, sleep_fn=sleeps.append)
    ingestion = PlayerGameLogIngestion(db, client)

    written = ingestion.ingest_player_season(203999, "Nikola Jokic", "2025-26")

    assert written == 3
    endpoint_name, params = calls[0]
    assert endpoint_name == "PlayerGameLog"
    assert params["player_id"] == 203999
    assert params["season"] == "2025-26"
    assert len(calls) == 2  # retried after the failure
    assert sleeps  # backoff sleep occurred between attempts


def test_nba_client_raises_after_max_retries():
    def always_fail(endpoint_name, params):
        raise ConnectionError("down")

    client = NBAStatsClient(
        max_retries=2,
        endpoint_caller=always_fail,
        sleep_fn=lambda s: None,
    )
    with pytest.raises(RuntimeError, match="PlayerGameLog"):
        client.get_player_game_log(203999, "2025-26")


# --- team defense ingestion -------------------------------------------------


def test_ingest_team_defense_fixture(db, load_fixture):
    rows = load_fixture("nba_league_defense.json")["LeagueDashPtTeamDefend"]
    written = TeamDefenseIngestion(db).ingest_rows(rows, "2025-10-21")

    assert written == 2
    uta = db.query_one(
        """
        SELECT * FROM fact_team_defense_daily
        WHERE team_id = ? AND snapshot_date = '2025-10-21'
        """,
        (1610612762,),
    )
    assert uta["def_rating"] == 118.9
    assert uta["def_rating_rank"] == 29
    assert uta["position_scope"] == "Overall"


# --- consensus odds ingestion: PrizePicks primary, Underdog fallback --------
# Both sources are 100% free and unauthenticated — no API keys anywhere.


def test_ingest_prizepicks_fixture(db, load_fixture):
    payload = load_fixture("prizepicks_projections.json")
    written = ConsensusOddsIngestion(db).ingest_prizepicks(
        payload, snapshot_time=SNAPSHOT
    )

    # 7 valid projections (null line_score is skipped) x book+consensus = 14
    assert written == 14

    lines = db.query("SELECT * FROM fact_consensus_lines")
    assert len(lines) == 14
    assert all(row["snapshot_time"] == SNAPSHOT for row in lines)


def test_prizepicks_normalizes_stat_types_lines_prices(db, load_fixture):
    payload = load_fixture("prizepicks_projections.json")
    ConsensusOddsIngestion(db).ingest_prizepicks(payload, snapshot_time=SNAPSHOT)

    def consensus(player_name, market):
        return db.query_one(
            """
            SELECT * FROM fact_consensus_lines
            WHERE player_name = ? AND market = ? AND bookmaker = ?
            """,
            (player_name, market, CONSENSUS_BOOKMAKER),
        )

    jokic_reb = consensus("Nikola Jokic", "player_rebounds")
    assert jokic_reb["is_consensus"] == 1
    assert jokic_reb["line"] == 12.5
    assert jokic_reb["over_price"] == -110  # flat PrizePicks payout
    assert jokic_reb["under_price"] == -110
    assert jokic_reb["event_id"] == "nba-evt-1"
    assert jokic_reb["game_date"] == "2025-10-23"

    assert consensus("Jayson Tatum", "player_points")["line"] == 26.5
    assert consensus("Stephen Curry", "player_threes")["line"] == 4.5
    assert (
        consensus("Giannis Antetokounmpo", "player_points_rebounds_assists")["line"]
        == 45.5
    )
    assert consensus("Nikola Jokic", "player_blocks")["line"] == 1.5
    # unmapped stat types still persist under a slugged market key
    assert consensus("Nikola Jokic", "player_dunks")["line"] == 1.5


def test_prizepicks_matchup_descriptions(db, load_fixture):
    payload = load_fixture("prizepicks_projections.json")
    ConsensusOddsIngestion(db).ingest_prizepicks(payload, snapshot_time=SNAPSHOT)

    def teams(player_name):
        return db.query_one(
            """
            SELECT home_team_abbr, away_team_abbr FROM fact_consensus_lines
            WHERE player_name = ? AND is_consensus = 1 LIMIT 1
            """,
            (player_name,),
        )

    # "UTA" (opponent only) -> away=UTA, home unknown
    row = teams("Nikola Jokic")
    assert row["home_team_abbr"] is None
    assert row["away_team_abbr"] == "UTA"
    # "vs PHX" -> player's team (LAL) is home
    row = teams("LeBron James")
    assert row["home_team_abbr"] == "LAL"
    assert row["away_team_abbr"] == "PHX"
    # "GSW @ POR" -> explicit home/away
    row = teams("Stephen Curry")
    assert row["home_team_abbr"] == "POR"
    assert row["away_team_abbr"] == "GSW"


def test_prizepicks_links_known_players_and_sets_team(db, load_fixture):
    PlayerGameLogIngestion(db).ingest_rows(
        load_fixture("nba_player_game_log.json")["PlayerGameLog"]
    )
    ConsensusOddsIngestion(db).ingest_prizepicks(
        load_fixture("prizepicks_projections.json"), snapshot_time=SNAPSHOT
    )

    jokic = db.query_one("""
        SELECT player_id FROM fact_consensus_lines
        WHERE player_name = 'Nikola Jokic' AND is_consensus = 1
          AND market = 'player_rebounds'
        """)
    assert jokic["player_id"] == 203999  # matched to dim_players

    lebron = db.query_one(
        "SELECT * FROM dim_players WHERE normalized_name = 'lebron james'"
    )
    assert lebron is not None  # stub created for unknown players
    lal = db.team_id_for_abbr("LAL")
    assert lebron["team_id"] == lal  # team from PrizePicks included attrs


def test_prizepicks_ingestion_is_idempotent(db, load_fixture):
    payload = load_fixture("prizepicks_projections.json")
    ingestion = ConsensusOddsIngestion(db)
    ingestion.ingest_prizepicks(payload, snapshot_time=SNAPSHOT)
    ingestion.ingest_prizepicks(payload, snapshot_time=SNAPSHOT)
    assert db.query_one("SELECT COUNT(*) AS n FROM fact_consensus_lines")["n"] == 14


def test_ingest_underdog_fixture(db, load_fixture):
    payload = load_fixture("underdog_lines.json")
    written = ConsensusOddsIngestion(db).ingest_underdog(
        payload, snapshot_time=SNAPSHOT
    )

    assert written == 4  # 2 lines x (bookmaker + consensus rows)

    jokic = db.query_one("""
        SELECT * FROM fact_consensus_lines
        WHERE player_name = 'Nikola Jokic' AND market = 'player_rebounds'
          AND bookmaker = 'underdog'
        """)
    assert jokic["line"] == 12.5
    assert jokic["over_price"] == -118  # 1.85x payout -> -118
    assert jokic["under_price"] == -105  # 1.95x payout -> -105
    assert jokic["home_team_abbr"] == "UTA"
    assert jokic["away_team_abbr"] == "DEN"
    assert jokic["game_date"] == "2025-10-23"


def test_ingest_statpick_fixture(db, load_fixture):
    payload = load_fixture("statpick_slate.json")
    written = ConsensusOddsIngestion(db).ingest_statpick(
        payload, snapshot_time=SNAPSHOT
    )

    # 4 valid headline props (null line skipped) x book+consensus = 8
    assert written == 8

    giannis = db.query_one("""
        SELECT * FROM fact_consensus_lines
        WHERE player_name = 'Giannis Antetokounmpo' AND is_consensus = 1
        """)
    assert giannis["market"] == "player_points"
    assert giannis["line"] == 28.5
    assert giannis["over_price"] == -122  # real consensus prices, not flat
    assert giannis["under_price"] == -108
    assert giannis["home_team_abbr"] == "MIA"
    assert giannis["away_team_abbr"] == "DET"
    assert giannis["game_date"] == "2026-10-23"
    assert giannis["event_id"] == "sp-21717940"
    assert giannis["player_id"] == 203507  # real NBA id from headshotUrl

    vuc = db.query_one("""
        SELECT market FROM fact_consensus_lines
        WHERE player_name = 'Nikola Vucevic' AND is_consensus = 1
        """)
    assert vuc["market"] == "player_points_rebounds_assists"

    # stub (no headshot) still resolves via normalized name
    bam = db.query_one(
        "SELECT player_id FROM fact_consensus_lines "
        "WHERE player_name = 'Bam Adebayo' AND is_consensus = 1"
    )
    assert bam["player_id"] == db.player_id_for_name("Bam Adebayo")


def test_morning_pull_prefers_statpick(db, load_fixture):
    slate = load_fixture("statpick_slate.json")
    projections = load_fixture("prizepicks_projections.json")
    sp_session = _FakeSession([slate])
    pp_session = _FakeSession([projections])

    result = ConsensusOddsIngestion(
        db,
        statpick=StatPickClient(session=sp_session, sleep_fn=lambda s: None),
        primary=PrizePicksClient(session=pp_session, sleep_fn=lambda s: None),
    ).ingest_morning_pull()

    assert result["source"] == "statpick"
    assert result["written"] == 8
    assert len(pp_session.calls) == 0  # downstream sources untouched


def test_morning_pull_chain_falls_through_failures(db, load_fixture):
    projections = load_fixture("prizepicks_projections.json")
    sp_session = _FakeSession([], fail_first=10)
    pp_session = _FakeSession([projections])

    result = ConsensusOddsIngestion(
        db,
        statpick=StatPickClient(
            max_retries=2, session=sp_session, sleep_fn=lambda s: None
        ),
        primary=PrizePicksClient(session=pp_session, sleep_fn=lambda s: None),
    ).ingest_morning_pull()

    assert result["source"] == "prizepicks"
    assert result["written"] == 14
    assert "statpick" in result["skipped_sources"]


# --- clients with injected session -------------------------------------------


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class _FakeSession:
    """Duck-typed requests.Session returning queued fixture payloads."""

    def __init__(self, payloads, fail_first=0):
        self._payloads = list(payloads)
        self._fail_first = fail_first
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append({"url": url, "params": params})
        if self._fail_first > 0:
            self._fail_first -= 1
            import requests

            raise requests.ConnectionError("simulated failure")
        return _FakeResponse(self._payloads.pop(0))


def test_prizepicks_client_morning_pull_offline(db, load_fixture):
    projections = load_fixture("prizepicks_projections.json")
    session = _FakeSession([projections])
    client = PrizePicksClient(session=session, sleep_fn=lambda s: None)

    result = ConsensusOddsIngestion(db, primary=client).ingest_morning_pull()

    assert result["source"] == "prizepicks"
    assert result["written"] == 14
    assert session.calls[0]["url"].endswith("/projections")
    params = session.calls[0]["params"]
    assert params["league_id"] == 7
    # zero-auth: no API key, token, or auth header anywhere
    assert "apiKey" not in params
    assert "api_key" not in params
    assert "Authorization" not in params


def test_prizepicks_client_retries_on_failure(load_fixture):
    projections = load_fixture("prizepicks_projections.json")
    session = _FakeSession([projections], fail_first=1)
    client = PrizePicksClient(session=session, sleep_fn=lambda s: None)
    assert client.get_projections() == projections
    assert len(session.calls) == 2


def test_morning_pull_falls_back_to_underdog(db, load_fixture):
    underdog = load_fixture("underdog_lines.json")
    primary_session = _FakeSession([], fail_first=10)
    fallback_session = _FakeSession([underdog])
    primary = PrizePicksClient(
        max_retries=2, session=primary_session, sleep_fn=lambda s: None
    )
    fallback = UnderdogClient(session=fallback_session, sleep_fn=lambda s: None)

    result = ConsensusOddsIngestion(
        db, primary=primary, fallback=fallback
    ).ingest_morning_pull()

    assert result["source"] == "underdog"
    assert result["written"] == 4
    assert "prizepicks" in result["skipped_sources"]
    assert len(fallback_session.calls) == 1
    assert fallback_session.calls[0]["url"].endswith("/beta/v3/over_under_lines")


# --- injury report parsing --------------------------------------------------


def test_parse_injury_entries(load_fixture):
    raw = load_fixture("injury_report.json")
    entries = parse_injury_entries(raw)

    assert len(entries) == 3  # blank player + invalid status skipped
    by_name = {e.player_name: e for e in entries}
    assert by_name["Jamal Murray"].status == "out"
    assert by_name["Anthony Davis"].status == "questionable"
    assert by_name["LeBron James"].team_abbr == "LAL"


# --- ingestion audit --------------------------------------------------------


def test_ingestion_runs_are_logged(db, load_fixture):
    PlayerGameLogIngestion(db).ingest_rows(
        load_fixture("nba_player_game_log.json")["PlayerGameLog"]
    )
    ConsensusOddsIngestion(db).ingest_prizepicks(
        load_fixture("prizepicks_projections.json"), snapshot_time=SNAPSHOT
    )
    runs = db.query(
        "SELECT source, records_written FROM ingestion_runs ORDER BY run_id"
    )
    sources = {r["source"]: r["records_written"] for r in runs}
    assert sources["nba_stats.player_game_log"] == 3
    assert sources["consensus_odds.prizepicks"] == 14


# --- upsert conflict coverage -------------------------------------------------


def _seed_game(db, player_id, game_date, team_id, opp_id, game_id=None, **stats):
    db.upsert_team(team_id, f"T{team_id}")
    db.upsert_team(opp_id, f"O{opp_id}")
    db.upsert_player(player_id, f"Player {player_id}", team_id=team_id)
    row = {
        "game_id": game_id or f"g-{player_id}-{game_date}-{team_id}",
        "player_id": player_id,
        "team_id": team_id,
        "opponent_team_id": opp_id,
        "game_date": game_date,
    }
    row.update(stats)
    db.upsert_player_game(row)


def test_upsert_player_game_updates_all_box_columns(db):
    db.upsert_team(100, "T100")
    db.upsert_team(200, "O200")
    db.upsert_player(7, "Test Player", team_id=100)
    base = {
        "team_id": 100,
        "opponent_team_id": 200,
        "season_id": "22025",
        "matchup": "T100 @ O200",
        "is_home": 0,
        "wl": "W",
        "minutes": 30,
        "pts": 20,
        "fgm": 8,
        "fga": 15,
        "stl": 1,
        "blk": 0,
        "tov": 2,
        "pf": 3,
        "plus_minus": 5.0,
        "touches": 61.0,
    }
    db.upsert_player_game(
        {**base, "game_id": "g1", "player_id": 7, "game_date": "2025-10-22"}
    )

    refreshed = {
        **base,
        "game_id": "g1",
        "player_id": 7,
        "game_date": "2025-10-22",
        "pts": 35,
        "stl": 4,
        "blk": 2,
        "tov": 5,
        "pf": 1,
        "plus_minus": -3.0,
        "wl": "L",
    }
    refreshed.pop("touches")  # tracking feed absent -> keep prior value
    db.upsert_player_game(refreshed)

    row = db.query_one("SELECT * FROM fact_player_games WHERE game_id = 'g1'")
    assert row["pts"] == 35
    assert row["stl"] == 4
    assert row["blk"] == 2
    assert row["tov"] == 5
    assert row["pf"] == 1
    assert row["plus_minus"] == -3.0
    assert row["wl"] == "L"
    assert row["season_id"] == "22025"
    assert row["touches"] == 61.0  # COALESCE preserved the tracked value


def test_upsert_team_defense_updates_opp_columns(db):
    db.upsert_team(100, "T100")
    row = {
        "team_id": 100,
        "snapshot_date": "2025-10-21",
        "def_rating": 110.0,
        "opp_pts": 105.0,
        "opp_reb": 40.0,
        "opp_ast": 24.0,
        "opp_fg3m": 12.0,
        "paint_pts_rank": 5,
    }
    db.upsert_team_defense(row)
    db.upsert_team_defense(
        {
            **row,
            "opp_pts": 120.0,
            "opp_reb": 48.5,
            "paint_pts_rank": 28,
            "season_id": "22025",
        }
    )
    snap = db.query_one("SELECT * FROM fact_team_defense_daily WHERE team_id = 100")
    assert snap["opp_pts"] == 120.0
    assert snap["opp_reb"] == 48.5
    assert snap["paint_pts_rank"] == 28
    assert snap["season_id"] == "22025"


# --- incremental watermarks ---------------------------------------------------


def test_latest_game_date_watermark(db):
    _seed_game(db, 7, "2025-04-10", 100, 200)  # prior-season tail
    _seed_game(db, 7, "2025-10-20", 100, 200)
    _seed_game(db, 7, "2025-10-22", 100, 200)
    ingestion = PlayerGameLogIngestion(db)

    assert ingestion.latest_game_date() == "2025-10-22"
    assert ingestion.latest_game_date("2025-10-21") == "2025-10-22"
    assert ingestion.latest_game_date("2025-11-01") is None
    # season label '2025-26' bounds to games on/after 2025-08-01
    assert ingestion.latest_game_date("2025-26") == "2025-10-22"
    assert nba_date_param("2025-10-22") == "10/22/2025"
    assert nba_date_param(None) is None


# --- zero-auth ESPN box-score fallback ----------------------------------------


def _espn_scoreboard(event_id="4011", completed=True):
    return {
        "events": [
            {
                "id": event_id,
                "status": {"type": {"completed": completed, "state": "post"}},
                "competitions": [
                    {
                        "competitors": [
                            {
                                "homeAway": "home",
                                "team": {"abbreviation": "LAL"},
                                "score": "110",
                                "winner": True,
                            },
                            {
                                "homeAway": "away",
                                "team": {"abbreviation": "DEN"},
                                "score": "105",
                                "winner": False,
                            },
                        ]
                    }
                ],
            },
            {  # not yet completed -> skipped
                "id": "9999",
                "status": {"type": {"completed": False, "state": "pre"}},
                "competitions": [{"competitors": []}],
            },
        ]
    }


_ESPN_KEYS = [
    "min",
    "fg",
    "3pt",
    "ft",
    "oreb",
    "dreb",
    "reb",
    "ast",
    "stl",
    "blk",
    "to",
    "pf",
    "+/-",
    "pts",
]


def _espn_summary():
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"abbreviation": "DEN"},
                    "statistics": [
                        {
                            "keys": _ESPN_KEYS,
                            "athletes": [
                                {
                                    "athlete": {"displayName": "Nikola Jokic"},
                                    "stats": [
                                        "34:22",
                                        "11-18",
                                        "2-4",
                                        "4-5",
                                        "3",
                                        "12",
                                        "15",
                                        "9",
                                        "1",
                                        "1",
                                        "3",
                                        "2",
                                        "+8",
                                        "28",
                                    ],
                                },
                                {
                                    "athlete": {"displayName": "Bench Warmer"},
                                    "didNotPlay": True,
                                    "stats": [],
                                },
                            ],
                        }
                    ],
                },
                {
                    "team": {"abbreviation": "LAL"},
                    "statistics": [
                        {
                            "keys": _ESPN_KEYS,
                            "athletes": [
                                {
                                    "athlete": {"displayName": "LeBron James"},
                                    "stats": [
                                        "36:00",
                                        "10-20",
                                        "3-8",
                                        "7-9",
                                        "2",
                                        "7",
                                        "9",
                                        "8",
                                        "2",
                                        "0",
                                        "2",
                                        "4",
                                        "+5",
                                        "30",
                                    ],
                                }
                            ],
                        }
                    ],
                },
            ]
        }
    }


def test_espn_boxscore_client_fallback(db):
    session = _FakeSession([_espn_scoreboard(), _espn_summary()])
    client = ESPNBoxScoreClient(session=session, sleep_fn=lambda s: None)

    rows = client.fetch_missing_dates(["2025-10-22"])

    assert len(session.calls) == 2  # scoreboard + one completed-event summary
    assert "/scoreboard" in session.calls[0]["url"]
    assert session.calls[0]["params"]["dates"] == "20251022"
    assert "/summary" in session.calls[1]["url"]

    assert len(rows) == 2  # DNP athlete skipped, non-completed event skipped
    written = PlayerGameLogIngestion(db).ingest_rows(rows)
    assert written == 2

    jokic = db.query_one(
        "SELECT g.* FROM fact_player_games g"
        " JOIN dim_players p ON p.player_id = g.player_id"
        " WHERE p.normalized_name = 'nikola jokic'"
    )
    assert jokic["pts"] == 28
    assert jokic["fgm"] == 11 and jokic["fga"] == 18
    assert jokic["fg3m"] == 2 and jokic["fg3a"] == 4
    assert jokic["ftm"] == 4 and jokic["fta"] == 5
    assert jokic["minutes"] == pytest.approx(34 + 22 / 60)
    assert jokic["plus_minus"] == 8.0
    assert jokic["tov"] == 3 and jokic["pf"] == 2
    assert jokic["wl"] == "L"  # DEN lost 105-110
    assert jokic["is_home"] == 0  # DEN @ LAL
    assert jokic["opponent_team_id"] == db.team_id_for_abbr("LAL")

    # home team block -> "LAL vs. DEN" so is_home/opponent resolve right
    lebron = db.query_one(
        "SELECT g.* FROM fact_player_games g"
        " JOIN dim_players p ON p.player_id = g.player_id"
        " WHERE p.normalized_name = 'lebron james'"
    )
    assert lebron["is_home"] == 1
    assert lebron["wl"] == "W"
    assert lebron["opponent_team_id"] == db.team_id_for_abbr("DEN")


# --- local team-defense derivation (zero API calls) ---------------------------


def _seed_dvp_game(db, pid, d, team_id, opp_id, game_id, position, **stats):
    db.upsert_team(team_id, f"T{team_id}")
    db.upsert_team(opp_id, f"O{opp_id}")
    db.upsert_player(pid, f"Player {pid}", team_id=team_id, position=position)
    _seed_game(db, pid, d, team_id, opp_id, game_id=game_id, **stats)


def test_derive_from_games_overall_and_position_scopes(db):
    dates = ["2025-10-01", "2025-10-03", "2025-10-05", "2025-10-07", "2025-10-09"]
    for i, d in enumerate(dates):
        _seed_dvp_game(
            db,
            11,
            d,
            100,
            200,
            f"ev{i}",
            "PG",
            pts=20,
            reb=8,
            ast=5,
            fg3m=2,
            fgm=8,
            fga=14,
            fg3a=6,
            fta=4,
            oreb=2,
            tov=3,
        )
        _seed_dvp_game(
            db,
            12,
            d,
            100,
            200,
            f"ev{i}",
            "C",
            pts=10,
            reb=4,
            ast=3,
            fg3m=1,
            fgm=4,
            fga=8,
            fg3a=3,
            fta=2,
            oreb=1,
            tov=2,
        )
        _seed_dvp_game(
            db,
            21,
            d,
            200,
            100,
            f"ev{i}",
            "SF",
            pts=15,
            reb=6,
            ast=4,
            fg3m=3,
            fgm=6,
            fga=12,
            fg3a=7,
            fta=3,
            oreb=1,
            tov=4,
        )

    written = TeamDefenseIngestion(db).derive_from_games("2025-10-20")

    snap = db.query_one(
        "SELECT * FROM fact_team_defense_daily"
        " WHERE team_id = 200 AND position_scope = 'ALL'"
    )
    assert snap["opp_pts"] == pytest.approx(30.0)  # 20 + 10 per game
    assert snap["opp_reb"] == pytest.approx(12.0)
    assert snap["opp_ast"] == pytest.approx(8.0)
    assert snap["opp_fg3m"] == pytest.approx(3.0)
    # paint pts allowed = 2 * (fgm - fg3m) = 2 * (12 - 3) = 18
    assert snap["paint_pts_allowed"] == pytest.approx(18.0)
    # softer than team 100 (which only allows 15 ppg) -> rank 2
    assert snap["def_rating_rank"] == 2

    pg = db.query_one(
        "SELECT * FROM fact_team_defense_daily"
        " WHERE team_id = 200 AND position_scope = 'PG'"
    )
    assert pg["opp_pts"] == pytest.approx(20.0)
    g_scope = db.query_one(
        "SELECT * FROM fact_team_defense_daily"
        " WHERE team_id = 200 AND position_scope = 'G'"
    )
    assert g_scope["opp_pts"] == pytest.approx(20.0)
    c_scope = db.query_one(
        "SELECT * FROM fact_team_defense_daily"
        " WHERE team_id = 200 AND position_scope = 'C'"
    )
    assert c_scope["opp_pts"] == pytest.approx(10.0)
    assert written >= 2 + 5  # 2 ALL rows + G/F/C + PG/SF scoped rows


def test_derive_from_games_season_fallback(db):
    dates = ["2025-03-10", "2025-03-12", "2025-03-14"]
    for i, d in enumerate(dates):
        _seed_dvp_game(db, 31, d, 100, 200, f"old{i}", "SG", pts=18, fga=10)
    # only 3 in-season games after 2025-08-01 -> falls back to all history
    written = TeamDefenseIngestion(db).derive_from_games(
        "2025-10-20", season_start="2025-08-01"
    )
    assert written >= 2
    snap = db.query_one(
        "SELECT * FROM fact_team_defense_daily"
        " WHERE team_id = 200 AND position_scope = 'ALL'"
    )
    assert snap["opp_pts"] == pytest.approx(18.0)


# --- daily refresh ingestion guards --------------------------------------------


def test_ingest_nba_skips_seeded_and_same_day(db):
    from src.publication.daily_refresh import _ingest_nba_and_defense

    calls = []

    def caller(endpoint, params):
        calls.append((endpoint, params))
        return {"LeagueGameLog": []}

    nba = NBAStatsClient(endpoint_caller=caller, sleep_fn=lambda s: None)
    _seed_game(db, 7, "2025-04-10", 100, 200)  # prior-season tail seeded
    _seed_game(db, 7, "2025-10-24", 100, 200)  # current season thru yesterday

    result = _ingest_nba_and_defense(db, "2025-10-25", nba)

    assert result["prior_season_cached"] is True
    assert result["game_log_cached"] is True
    assert result["game_rows"] == 0
    assert calls == []  # zero nba_api calls
    assert result["defense_rows"] >= 1  # derived locally, zero API calls


def test_ingest_nba_incremental_date_from(db):
    from src.publication.daily_refresh import _ingest_nba_and_defense

    calls = []

    def caller(endpoint, params):
        calls.append((endpoint, params))
        return {"LeagueGameLog": []}

    nba = NBAStatsClient(endpoint_caller=caller, sleep_fn=lambda s: None)
    _seed_game(db, 7, "2025-04-10", 100, 200)
    _seed_game(db, 7, "2025-10-20", 100, 200)

    _ingest_nba_and_defense(db, "2025-10-25", nba)

    league_calls = [c for c in calls if c[0] == "LeagueGameLog"]
    assert len(league_calls) == 1  # prior tail skipped, one incremental call
    assert league_calls[0][1]["date_from_nullable"] == "10/20/2025"


def test_ingest_nba_falls_back_to_espn(db):
    from src.publication.daily_refresh import _ingest_nba_and_defense

    def failing(endpoint, params):
        raise ConnectionError("simulated stats.nba.com block")

    nba = NBAStatsClient(
        max_retries=1, endpoint_caller=failing, sleep_fn=lambda s: None
    )
    _seed_game(db, 7, "2025-10-21", 100, 200)  # only missing 10-22
    session = _FakeSession([_espn_scoreboard(), _espn_summary()])
    espn = ESPNBoxScoreClient(session=session, sleep_fn=lambda s: None)

    result = _ingest_nba_and_defense(db, "2025-10-23", nba, espn=espn)

    assert "prior_season_error" in result  # prior tail fetch also failed
    assert "nba_stats_error" in result
    assert result["game_rows"] == 2  # ESPN backfilled the missing date
    assert session.calls[0]["params"]["dates"] == "20251022"
