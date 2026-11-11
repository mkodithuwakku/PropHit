"""Automated morning pipeline runner (Milestone 6, RUNBOOK_DAILY_REFRESH).

Steps: (optionally) ingest live data -> generate evaluated plays ->
write static bundle -> optionally publish to GitHub Pages.

Offline/dry-run mode skips all network and git side effects while still
producing the build artifacts — the M6 verification gate.

Usage:
    .venv/bin/python -m src.publication.daily_refresh \\
        --db data/prophit.db --out site --date 2025-10-22 --dry-run
"""

from __future__ import annotations

import argparse
from datetime import date as _date
from datetime import timedelta
from pathlib import Path
from typing import Optional

from src.ingestion.db import ProphitDB
from src.publication.publisher import GitPublisher
from src.publication.static_builder import SlateGenerator, StaticSiteBuilder


def _season_for(game_date: str) -> str:
    """'2025-10-23' -> '2025-26' (NBA season runs Oct-Jun)."""
    year, month = int(game_date[:4]), int(game_date[5:7])
    start = year if month >= 7 else year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def _ingest_nba_and_defense(
    db: ProphitDB,
    target_date: str,
    nba,
    espn=None,
) -> dict:
    """Incremental box-score + local defense refresh.

    1. Prior-season tail is only fetched when the seeded window is empty.
    2. Current season uses an incremental ``date_from`` watermark and is
       skipped entirely when the DB already covers yesterday.
    3. On nba_api failure the missing dates are backfilled via the
       zero-auth ESPN box-score endpoints (up to 5 days).
    4. Team defense is derived locally from fact_player_games — zero
       LeagueDashPtTeamDefend calls.
    """
    from src.ingestion.nba_stats import (
        ESPNBoxScoreClient,
        PlayerGameLogIngestion,
        TeamDefenseIngestion,
        nba_date_param,
    )

    result: dict = {}
    season = _season_for(target_date)
    start_year = int(season[:4])
    log_ingestion = PlayerGameLogIngestion(db, nba)

    prior = db.query_one(
        "SELECT COUNT(*) AS n FROM fact_player_games "
        "WHERE game_date >= ? AND game_date < ?",
        (f"{start_year}-03-01", f"{start_year}-07-01"),
    )
    if prior["n"] == 0:
        # Tail of last regular season so early-season slates still have
        # L20 rolling context (SeasonType filter excludes playoffs).
        prev = f"{start_year - 1}-{int(season[-2:]) - 1:02d}"
        try:
            result["prior_season_rows"] = log_ingestion.ingest_rows(
                nba.get_league_game_log(prev, date_from=f"03/01/{start_year}")
            )
        except Exception as exc:
            result["prior_season_error"] = str(exc)
    else:
        result["prior_season_rows"] = 0
        result["prior_season_cached"] = True

    yesterday = (_date.fromisoformat(target_date) - timedelta(days=1)).isoformat()
    max_game_date = log_ingestion.latest_game_date(season)
    if max_game_date and max_game_date >= yesterday:
        result["game_rows"] = 0
        result["game_log_cached"] = True
    else:
        try:
            result["game_rows"] = log_ingestion.ingest_rows(
                nba.get_league_game_log(season, date_from=nba_date_param(max_game_date))
            )
        except Exception as exc:
            result["nba_stats_error"] = str(exc)
            result["game_rows"] = _espn_backfill(
                db,
                log_ingestion,
                espn or ESPNBoxScoreClient(),
                max_game_date,
                yesterday,
                result,
            )

    try:
        result["defense_rows"] = TeamDefenseIngestion(db).derive_from_games(
            snapshot_date=target_date,
            season_start=f"{start_year}-08-01",
        )
    except Exception as exc:
        result["defense_error"] = str(exc)
    return result


def _espn_backfill(
    db: ProphitDB,
    log_ingestion,
    espn,
    max_game_date: Optional[str],
    yesterday: str,
    result: dict,
) -> int:
    """Fetch up to 5 missing days of box scores from ESPN's public API."""
    end = _date.fromisoformat(yesterday)
    if max_game_date:
        start = _date.fromisoformat(max_game_date) + timedelta(days=1)
    else:
        start = end - timedelta(days=4)
    dates = []
    current = start
    while current <= end and len(dates) < 5:
        dates.append(current.isoformat())
        current += timedelta(days=1)
    try:
        rows = espn.fetch_missing_dates(dates)
        return log_ingestion.ingest_rows(rows)
    except Exception as exc:
        result["espn_error"] = str(exc)
        return 0


def maybe_ingest_live(
    db: ProphitDB,
    game_date: Optional[str] = None,
    nba_client=None,
    espn_client=None,
) -> dict:
    """Live pull via zero-auth sources — no keys or secrets needed.

    1. Consensus lines: statpick.ai public prop API (primary),
       PrizePicks public projections, then Underdog Fantasy public
       lines (automatic fallback chain).
    2. Player box-score history: incremental LeagueGameLog via nba_api
       (free), with a zero-auth ESPN box-score backfill on failure.
    3. Team defense derived locally — no LeagueDashPtTeamDefend calls.
    """
    from src.ingestion.consensus_odds import (
        ConsensusOddsIngestion,
        PrizePicksClient,
        StatPickClient,
        UnderdogClient,
    )
    from src.ingestion.nba_stats import NBAStatsClient

    target_date = str(game_date or _date.today().isoformat())[:10]
    result: dict = {"ingested": False}
    try:
        result.update(
            ConsensusOddsIngestion(
                db,
                statpick=StatPickClient(),
                primary=PrizePicksClient(),
                fallback=UnderdogClient(),
            ).ingest_morning_pull()
        )
        result["ingested"] = True
    except Exception as exc:
        return {**result, "reason": f"consensus: {exc}"}

    result.update(
        _ingest_nba_and_defense(
            db,
            target_date,
            nba_client or NBAStatsClient(),
            espn=espn_client,
        )
    )
    return result


def run_refresh(
    db_path: "str | Path",
    out_dir: "str | Path" = "site",
    game_date: Optional[str] = None,
    min_ev_pct: float = 5.0,
    live: bool = False,
    publish: bool = False,
    dry_run: bool = True,
) -> dict:
    """Execute the morning refresh. Returns a summary of what happened."""
    game_date_explicit = game_date is not None
    game_date = str(game_date or _date.today().isoformat())[:10]
    db = ProphitDB(db_path)
    try:
        ingest = maybe_ingest_live(db, game_date) if live else {"ingested": False}
        if not game_date_explicit:
            has_lines = db.query_one(
                "SELECT 1 FROM fact_consensus_lines WHERE is_consensus = 1 AND game_date = ? LIMIT 1",
                (game_date,),
            )
            if not has_lines:
                latest_slate = db.query_one(
                    "SELECT MAX(game_date) AS d FROM fact_consensus_lines WHERE is_consensus = 1"
                )
                if latest_slate and latest_slate["d"]:
                    game_date = str(latest_slate["d"])[:10]

        from src.evaluation.calibrate import run_calibration
        from src.evaluation.record import summarize_model_stats

        # Bi-weekly self-calibration: skips itself when <14d since last run
        # unless 3+ new game dates have settled (continuous tuning).
        calibration = run_calibration(db, min_new_settled_days=3)

        generator = SlateGenerator(db, min_ev_pct=min_ev_pct)
        plays = generator.generate_plays(game_date)
        slate = {
            "date": game_date,
            "plays": plays,
            "model_stats": summarize_model_stats(db, generator.min_ev_pct, game_date),
            "calibration": calibration,
        }
        build = StaticSiteBuilder(out_dir).build(slate)

        publish_result = None
        if publish:
            publish_result = GitPublisher().publish(
                out_dir,
                message=f"Daily slate {game_date}",
                dry_run=dry_run,
            )

        return {
            "date": game_date,
            "ingestion": ingest,
            "plays": len(plays),
            "recommended": build["recommended_count"],
            "milestones": build["milestone_count"],
            "out_dir": build["out_dir"],
            "files": build["files"],
            "publish": publish_result,
            "calibration": calibration,
        }
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prophit daily refresh")
    parser.add_argument("--db", default="data/prophit.db")
    parser.add_argument("--out", default="site")
    parser.add_argument("--date", default=None)
    parser.add_argument("--min-ev", type=float, default=5.0)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--dry-run", action="store_true", default=True)
    args = parser.parse_args()

    result = run_refresh(
        db_path=args.db,
        out_dir=args.out,
        game_date=args.date,
        min_ev_pct=args.min_ev,
        live=args.live,
        publish=args.publish,
        dry_run=args.dry_run,
    )
    print(result)


if __name__ == "__main__":
    main()
