"""Bi-weekly self-calibration loop (runs inside the daily refresh).

Every `interval_days` (default 14) the morning pipeline calls
`run_calibration`, which re-grades model projections against the
trailing-mean backtest and applies bounded corrections, all persisted
in the `model_settings` table so every change is visible and revertable:

- `min_ev_pct`   — swept over a fixed ladder, best gate adopted, but
                   clamped to +/-2 points per calibration run so the
                   system can't whipsaw on noisy samples.
- `suppressed_markets` — JSON list of stats whose graded ROI is below
                   the suppression floor over enough bets. Suppressed
                   markets still appear on the slate (flagged) but are
                   never recommended. They un-suppress automatically
                   when graded ROI recovers.

Guardrails: minimum bet counts before any action, per-run clamp on the
gate, and every write carries a human-readable reason that surfaces on
the dashboard slate JSON.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from src.ingestion.db import ProphitDB

INTERVAL_DAYS = 14
EV_LADDER = (5.0, 8.0, 10.0, 12.0, 15.0)
MAX_GATE_STEP = 2.0
MIN_BETS_FOR_TUNING = 100
SUPPRESS_ROI_PCT = -5.0
UNSUPPRESS_ROI_PCT = 3.0
MIN_BETS_FOR_SUPPRESSION = 50
DEFAULT_MIN_EV = 5.0


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def get_min_ev(db: ProphitDB, fallback: float = DEFAULT_MIN_EV) -> float:
    """Effective EV gate: calibrated value if present, else fallback."""
    raw = db.get_setting("min_ev_pct")
    try:
        return float(raw) if raw is not None else fallback
    except ValueError:
        return fallback


def get_suppressed_markets(db: ProphitDB) -> list[str]:
    raw = db.get_setting("suppressed_markets")
    try:
        val = json.loads(raw) if raw else []
        return [str(v) for v in val]
    except (ValueError, TypeError):
        return []


def _sweep_gate(detail: list[dict], gate: float) -> dict:
    bets = [d for d in detail if d["ev_pct"] is not None and d["ev_pct"] >= gate]
    units = sum(d["units"] for d in bets)
    return {
        "gate": gate,
        "bets": len(bets),
        "units": round(units, 2),
        "roi_pct": round(100.0 * units / len(bets), 1) if bets else None,
    }


def run_calibration(
    db: ProphitDB,
    *,
    interval_days: int = INTERVAL_DAYS,
    now: Optional[datetime] = None,
    backtest_fn: Optional[Callable[..., dict]] = None,
) -> dict[str, Any]:
    """Run the bi-weekly calibration if due. Returns a report dict.

    `backtest_fn` is injectable for tests; production uses the real
    trailing-mean backtest with detail collection.
    """
    now = now or datetime.now(timezone.utc)
    last = _parse_dt(db.get_setting("last_calibration"))
    if last and now - last < timedelta(days=interval_days):
        return {
            "ran": False,
            "reason": f"last run {last.date().isoformat()} < {interval_days}d ago",
            "min_ev_pct": get_min_ev(db),
            "suppressed_markets": get_suppressed_markets(db),
        }

    if backtest_fn is None:
        from src.evaluation.backtest import backtest as backtest_fn

    res = backtest_fn(db, min_ev_pct=0.0, collect=True)
    detail = res.get("bets_detail", [])
    notes: list[str] = []

    # --- min_ev gate sweep --------------------------------------------------
    current_gate = get_min_ev(db)
    sweeps = [_sweep_gate(detail, g) for g in EV_LADDER]
    eligible = [s for s in sweeps if s["bets"] >= MIN_BETS_FOR_TUNING]
    best = max(eligible, key=lambda s: s["units"], default=None)
    new_gate = current_gate
    if best and best["gate"] != current_gate:
        target = best["gate"]
        lo, hi = current_gate - MAX_GATE_STEP, current_gate + MAX_GATE_STEP
        new_gate = min(max(target, lo), hi)
        cur = _sweep_gate(detail, current_gate)
        best_roi = best["roi_pct"] or -999
        cur_roi = cur["roi_pct"] or -999
        if new_gate != current_gate and best_roi > cur_roi + 1.0:
            db.set_setting(
                "min_ev_pct",
                new_gate,
                f"calibration: {best['roi_pct']}% ROI @gate {best['gate']} "
                f"vs {cur['roi_pct']}% @{current_gate} "
                f"({best['bets']} bets, clamped ±{MAX_GATE_STEP})",
            )
            notes.append(f"min_ev {current_gate}->{new_gate}")
        else:
            new_gate = current_gate

    # --- per-market suppression ---------------------------------------------
    suppressed = set(get_suppressed_markets(db))
    for stat in sorted({d["stat"] for d in detail}):
        stat_bets = [
            d
            for d in detail
            if d["stat"] == stat and d["ev_pct"] is not None and d["ev_pct"] >= new_gate
        ]
        units = sum(d["units"] for d in stat_bets)
        roi = 100.0 * units / len(stat_bets) if stat_bets else None
        if (
            len(stat_bets) >= MIN_BETS_FOR_SUPPRESSION
            and roi is not None
            and roi <= SUPPRESS_ROI_PCT
            and stat not in suppressed
        ):
            suppressed.add(stat)
            notes.append(f"suppressed {stat} ({roi:.1f}% ROI)")
        elif (
            stat in suppressed
            and len(stat_bets) >= MIN_BETS_FOR_SUPPRESSION
            and roi is not None
            and roi >= UNSUPPRESS_ROI_PCT
        ):
            suppressed.discard(stat)
            notes.append(f"unsuppressed {stat} ({roi:+.1f}% ROI)")

    suppressed_list = sorted(suppressed)
    db.set_setting(
        "suppressed_markets", json.dumps(suppressed_list), "calibration sweep"
    )
    db.set_setting("last_calibration", now.isoformat(), "bi-weekly run")
    db.set_setting(
        "calibration_notes",
        json.dumps({"sweeps": sweeps, "notes": notes}),
        "bi-weekly run",
    )

    return {
        "ran": True,
        "date": now.date().isoformat(),
        "min_ev_pct": new_gate,
        "suppressed_markets": suppressed_list,
        "gate_sweep": sweeps,
        "notes": notes,
        "bets_analyzed": len(detail),
    }
