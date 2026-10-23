"""NBA injury report ingestion.

Parses official morning injury report entries into normalized
``InjuryEntry`` records. Feature-store consumers (M2) use these to compute
vacated usage/minutes; persistence lands with that milestone per SRD 4.1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

VALID_STATUSES = {"out", "questionable", "doubtful", "probable", "available"}


@dataclass(frozen=True)
class InjuryEntry:
    player_name: str
    status: str
    team_abbr: Optional[str] = None
    reason: Optional[str] = None
    report_date: Optional[str] = None


def parse_injury_entries(raw_entries: list[dict]) -> list[InjuryEntry]:
    """Normalize raw report rows into InjuryEntry records.

    Accepts dicts with keys like ``player``/``player_name``, ``status``,
    ``team``/``team_abbr``, ``reason``/``injury``, ``date``/``report_date``.
    Rows with an unknown player or unrecognized status are skipped.
    """
    entries = []
    for raw in raw_entries:
        player = raw.get("player") or raw.get("player_name")
        status = str(raw.get("status") or "").strip().lower()
        if not player or status not in VALID_STATUSES:
            continue
        entries.append(
            InjuryEntry(
                player_name=str(player).strip(),
                status=status,
                team_abbr=raw.get("team") or raw.get("team_abbr"),
                reason=raw.get("reason") or raw.get("injury"),
                report_date=raw.get("date") or raw.get("report_date"),
            )
        )
    return entries
