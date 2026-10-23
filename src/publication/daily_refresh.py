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
import os
from datetime import date as _date
from pathlib import Path
from typing import Optional

from src.ingestion.db import ProphitDB
from src.publication.publisher import GitPublisher
from src.publication.static_builder import SlateGenerator, StaticSiteBuilder


def maybe_ingest_live(db: ProphitDB) -> dict:
    """Live ingestion — skipped unless THE_ODDS_API_KEY is configured."""
    if not os.environ.get("THE_ODDS_API_KEY"):
        return {"ingested": False, "reason": "THE_ODDS_API_KEY not set"}
    # Live pulls land here (nba_stats/consensus_odds adapters, M1).
    return {"ingested": False, "reason": "live pull not enabled in this run"}


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
        ingest = maybe_ingest_live(db) if live else {"ingested": False}

        plays = SlateGenerator(db, min_ev_pct=min_ev_pct).generate_plays(game_date)
        slate = {"date": game_date, "plays": plays}
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
