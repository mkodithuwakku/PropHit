"""M1 verification: ingestion, parsing, and storage work 100% offline.

pytest-socket blocks all socket usage (see pytest.ini + conftest), so any
accidental live call to nba_api, PrizePicks, or Underdog fails loudly.
"""

import pytest

from src.ingestion.consensus_odds import (
    CONSENSUS_BOOKMAKER,
    ConsensusOddsIngestion,
    PrizePicksClient,
    UnderdogClient,
)
from src.ingestion.injuries import parse_injury_entries
from src.ingestion.nba_stats import (
    NBAStatsClient,
    PlayerGameLogIngestion,
    TeamDefenseIngestion,
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
    assert result["primary_error"]
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
