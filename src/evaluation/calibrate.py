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

# Continuous in-season tuning ladders (Phase 4). Values are swept per
# calibration run against the offline backtest and persisted in
# model_settings; prior_games drifts at most ±2.0 per run.
PARAM_LADDERS = {
    "prior_games": (4.0, 6.0, 8.0, 10.0),
    "ml_blend_weight": (0.25, 0.40, 0.55, 0.70),
    "dvp_weight": (0.04, 0.08, 0.12),
    "mc_blend_weight": (0.15, 0.25, 0.35),
}
PARAM_DEFAULTS = {
    "prior_games": 6.0,
    "ml_blend_weight": 0.55,
    "dvp_weight": 0.08,
    "mc_blend_weight": 0.25,
}
PARAM_MAX_DRIFT = {"prior_games": 2.0}


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


def _setting_float(db: ProphitDB, key: str, fallback: float) -> float:
    try:
        raw = db.get_setting(key)
        return float(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _new_settled_days(db: ProphitDB, last: datetime, now: datetime) -> int:
    """Distinct game_dates settled (played) since ``last`` and before today."""
    lo, hi = last.date().isoformat(), now.date().isoformat()
    row = db.query_one(
        "SELECT COUNT(DISTINCT d) AS n FROM ("
        " SELECT game_date AS d FROM fact_consensus_lines"
        "  WHERE game_date >= ? AND game_date < ?"
        " UNION"
        " SELECT game_date AS d FROM fact_player_games"
        "  WHERE game_date >= ? AND game_date < ?"
        ")",
        (lo, hi, lo, hi),
    )
    return int(row["n"] if row else 0)


def _param_objective(res: dict) -> Optional[float]:
    """Scalar score for a swept backtest: favored-side hit rate first,
    then total units from the collected detail."""
    overall = res.get("overall") or {}
    score = overall.get("favored_hit_pct")
    if score is not None:
        return float(score)
    detail = res.get("bets_detail") or []
    if detail:
        return sum(d.get("units") or 0.0 for d in detail)
    return None


def run_calibration(
    db: ProphitDB,
    *,
    interval_days: int = INTERVAL_DAYS,
    now: Optional[datetime] = None,
    backtest_fn: Optional[Callable[..., dict]] = None,
    min_new_settled_days: Optional[int] = None,
) -> dict[str, Any]:
    """Run the bi-weekly calibration if due. Returns a report dict.

    `backtest_fn` is injectable for tests; production uses the real
    trailing-mean backtest with detail collection.

    ``min_new_settled_days`` triggers an early run when at least that
    many new distinct game_dates have settled since last_calibration,
    even inside the normal interval window (daily_refresh passes 3).
    """
    now = now or datetime.now(timezone.utc)
    last = _parse_dt(db.get_setting("last_calibration"))
    if last and now - last < timedelta(days=interval_days):
        settled = (
            _new_settled_days(db, last, now) if min_new_settled_days is not None else 0
        )
        if min_new_settled_days is None or settled < min_new_settled_days:
            return {
                "ran": False,
                "reason": f"last run {last.date().isoformat()} < {interval_days}d ago",
                "min_ev_pct": get_min_ev(db),
                "suppressed_markets": get_suppressed_markets(db),
            }

    if backtest_fn is None:
        from src.evaluation.backtest import backtest as _real_bt
        from src.features.feature_store import FeatureStore

        shared_store = FeatureStore(db)

        def backtest_fn(db_arg, **kw):
            return _real_bt(db_arg, store=shared_store, **kw)

    tuned = {
        name: _setting_float(db, name, default)
        for name, default in PARAM_DEFAULTS.items()
    }
    res = backtest_fn(
        db,
        min_ev_pct=0.0,
        collect=True,
        prior_games=tuned["prior_games"],
        dvp_weight=tuned["dvp_weight"],
        ml_blend_weight=tuned["ml_blend_weight"],
        mc_blend_weight=tuned["mc_blend_weight"],
    )
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

    # --- continuous multi-parameter sweeps ----------------------------------
    for name, ladder in PARAM_LADDERS.items():
        current = tuned[name]
        scored = []
        for cand in ladder:
            cand_res = backtest_fn(db, min_ev_pct=0.0, collect=True, **{name: cand})
            scored.append((cand, _param_objective(cand_res)))
        scored_ok = [x for x in scored if x[1] is not None]
        best = max(scored_ok, key=lambda x: x[1], default=None)
        cur_score = next((s for c, s in scored if c == current and s is not None), None)
        target = current
        if best and best[0] != current and (cur_score is None or best[1] > cur_score):
            target = best[0]
        drift = PARAM_MAX_DRIFT.get(name)
        if drift is not None:
            target = min(max(target, current - drift), current + drift)
        if target != current:
            db.set_setting(
                name,
                target,
                f"calibration: {current}->{target} (swept {ladder})",
            )
            notes.append(f"{name} {current}->{target}")
        else:
            db.set_setting(name, current, "calibration: unchanged")

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
