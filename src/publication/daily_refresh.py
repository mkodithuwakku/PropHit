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


def maybe_ingest_live(db: ProphitDB, game_date: Optional[str] = None) -> dict:
    """Live pull via zero-auth sources — no keys or secrets needed.

    1. Consensus lines: statpick.ai public prop API (primary),
       PrizePicks public projections, then Underdog Fantasy public
       lines (automatic fallback chain).
    2. Player box-score history: one league-wide LeagueGameLog call via
       nba_api (free), which seeds every player appearing on the board —
       stub players get promoted to their real NBA IDs automatically.
    3. League defense snapshot for DvP matchup factors.
    """
    from src.ingestion.consensus_odds import (
        ConsensusOddsIngestion,
        PrizePicksClient,
        StatPickClient,
        UnderdogClient,
    )
    from src.ingestion.nba_stats import (
        NBAStatsClient,
        PlayerGameLogIngestion,
        TeamDefenseIngestion,
    )

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

    try:
        season = _season_for(game_date or _date.today().isoformat())
        nba = NBAStatsClient()
        log_ingestion = PlayerGameLogIngestion(db, nba)
        result["game_rows"] = log_ingestion.ingest_rows(nba.get_league_game_log(season))
        # Tail of last regular season so early-season slates still have
        # L20 rolling context (SeasonType filter excludes playoffs).
        prev = f"{int(season[:4]) - 1}-{int(season[-2:]) - 1:02d}"
        result["prior_season_rows"] = log_ingestion.ingest_rows(
            nba.get_league_game_log(prev, date_from=f"03/01/{int(season[:4])}")
        )
        result["defense_rows"] = TeamDefenseIngestion(db, nba).ingest_rows(
            nba.get_league_dash_team_defense(season),
            snapshot_date=game_date or _date.today().isoformat()[:10],
        )
    except Exception as exc:
        result["nba_stats_error"] = str(exc)

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
    game_date = str(game_date or _date.today().isoformat())[:10]
    db = ProphitDB(db_path)
    try:
        ingest = maybe_ingest_live(db, game_date) if live else {"ingested": False}

        from src.evaluation.record import summarize_model_stats

        plays = SlateGenerator(db, min_ev_pct=min_ev_pct).generate_plays(game_date)
        slate = {
            "date": game_date,
            "plays": plays,
            "model_stats": summarize_model_stats(db, min_ev_pct, game_date),
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
