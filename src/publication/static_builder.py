"""Static site generation & daily slate builder (Milestone 5).

``SlateGenerator`` runs the end-to-end morning evaluation: latest consensus
lines -> leak-free features -> stat projections -> PMF -> +EV / milestone
ladder -> play cards. ``StaticSiteBuilder`` writes ``daily_slate.json`` and
copies the vanilla web bundle into the deployable site directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import date as _date
from pathlib import Path
from typing import Optional

from src.decision.ev_calculator import EVCalculator
from src.decision.milestone_engine import MilestoneEngine
from src.features.feature_store import FeatureStore
from src.ingestion.db import ProphitDB, utcnow_iso
from src.models.distribution import DistributionEngine

MARKET_TO_STAT = {
    "player_points": "pts",
    "player_rebounds": "reb",
    "player_assists": "ast",
    "player_threes": "fg3m",
    "player_points_rebounds_assists": "pra",
    "player_points_rebounds": "pr",
    "player_points_assists": "pa",
    "player_rebounds_assists": "ra",
    "player_blocks": "blk",
    "player_steals": "stl",
    "player_turnovers": "tov",
}
COMBO_STATS = {
    "pra": ("pts", "reb", "ast"),
    "pr": ("pts", "reb"),
    "pa": ("pts", "ast"),
    "ra": ("reb", "ast"),
}
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


def _setting_float(db: ProphitDB, key: str, fallback: float) -> float:
    try:
        raw = db.get_setting(key)
        return float(raw) if raw is not None else fallback
    except (TypeError, ValueError):
        return fallback


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(value, lo), hi)


class SlateGenerator:
    """Produces evaluated play cards for a game date from the local DB."""

    MIN_ML_TRAIN_SAMPLES = 60
    ML_SAMPLE_CAP = 400
    MIN_STAT_MODEL_SAMPLES = 30

    def __init__(
        self,
        db: ProphitDB,
        min_ev_pct: float = 5.0,
        history_games: int = 20,
        prior_games: float = 6.0,
    ):
        self.db = db
        self.store = FeatureStore(db)
        self.dist = DistributionEngine()
        # Calibration overrides (model_settings table) take precedence over
        # the constructor default so bi-weekly tuning applies automatically.
        from src.evaluation.calibrate import (
            get_min_ev,
            get_suppressed_markets,
        )

        effective_ev = get_min_ev(db, fallback=min_ev_pct)
        self.ev = EVCalculator(min_ev_pct=effective_ev)
        self.ladder = MilestoneEngine()
        self.history_games = history_games
        self.min_ev_pct = effective_ev
        self.suppressed_markets = set(get_suppressed_markets(db))
        # Empirical-Bayes shrinkage: with thin box-score history the market
        # line is the best prior, so model means are pulled toward it.
        # weight_model = n / (n + prior_games); at n=20 the model gets ~77%.
        self.prior_games = _setting_float(db, "prior_games", prior_games)
        # Stage 1/2 ML blend + contextual multipliers (calibrated settings).
        self.ml_weight = _setting_float(db, "ml_blend_weight", 0.55)
        self.dvp_weight = _setting_float(db, "dvp_weight", 0.08)
        self.mc_weight = _setting_float(db, "mc_blend_weight", 0.25)
        self._minutes_model = None
        self._stat_models: dict = {}
        self._ml_date: Optional[str] = None
        self._multi_team: Optional[bool] = None

    def latest_consensus_lines(self, game_date: str) -> list:
        """Most recent consensus snapshot per event/player/market,
        restricted to the slate's game_date."""
        return self.db.query(
            """
            SELECT l.* FROM fact_consensus_lines l
            JOIN (
                SELECT event_id, player_name, market, MAX(snapshot_time) AS s
                FROM fact_consensus_lines
                WHERE is_consensus = 1 AND game_date = ?
                GROUP BY event_id, player_name, market
            ) latest
              ON latest.event_id = l.event_id
             AND latest.player_name = l.player_name
             AND latest.market = l.market
             AND latest.s = l.snapshot_time
            WHERE l.is_consensus = 1 AND l.game_date = ?
            """,
            (str(game_date)[:10], str(game_date)[:10]),
        )

    def _stat_history(self, player_id: int, stat: str, game_date: str) -> list[float]:
        stats = COMBO_STATS.get(stat, (stat,))
        cols = ", ".join(f"COALESCE({s}, 0) AS {s}" for s in stats)
        rows = self.db.query(
            f"SELECT {cols} FROM fact_player_games "
            "WHERE player_id = ? AND game_date < ? "
            f"ORDER BY game_date DESC LIMIT {self.history_games}",
            (player_id, str(game_date)[:10]),
        )
        return [float(sum(r[s] or 0 for s in stats)) for r in rows]

    # --- Stage 1/2 ML models -------------------------------------------------

    def _ensure_ml_models(
        self, game_date: str, player_ids: Optional[list[int]] = None
    ) -> None:
        """Lazily fit the minutes + per-stat models once per slate.

        All training rows are strictly before ``game_date`` (leak-free).
        Skipped entirely below MIN_ML_TRAIN_SAMPLES so small fixtures
        fast-path without fitting noise; sample lists are capped at
        ML_SAMPLE_CAP so production fits stay sub-second.
        """
        game_date = str(game_date)[:10]
        if self._ml_date == game_date:
            return
        self._ml_date = game_date
        self._minutes_model = None
        self._stat_models = {}

        n_rows = self.db.query_one(
            "SELECT COUNT(*) AS n FROM fact_player_games WHERE game_date < ?",
            (game_date,),
        )["n"]
        if n_rows < self.MIN_ML_TRAIN_SAMPLES:
            return

        from src.models.minutes_model import MinutesModel, build_minutes_samples
        from src.models.stat_models import build_rate_samples, build_stat_model

        if not player_ids or len(player_ids) < 10:
            db_pids = [
                r["player_id"]
                for r in self.db.query(
                    "SELECT DISTINCT player_id FROM fact_player_games "
                    "WHERE game_date < ? LIMIT 25",
                    (game_date,),
                )
            ]
            player_ids = sorted(set(player_ids or []) | set(db_pids))
        if not player_ids:
            return
        # Bound the player set: rolling_features is cached per (pid, date,
        # row_count), so the minutes + 4 stat builders share each player's
        # computed feature dicts with zero duplicate SQL work.
        player_ids = list(player_ids)[:25]
        per_player_cap = 12 if len(player_ids) >= 10 else None

        try:
            samples = [
                s
                for s in build_minutes_samples(
                    self.db,
                    player_ids,
                    store=self.store,
                    max_games_per_player=per_player_cap,
                )
                if s["game_date"] < game_date
            ][-self.ML_SAMPLE_CAP :]
            if len(samples) >= self.MIN_ML_TRAIN_SAMPLES:
                self._minutes_model = MinutesModel().fit(samples)
        except Exception:
            self._minutes_model = None

        for stat in ("pts", "reb", "ast", "fg3m"):
            try:
                rs = [
                    s
                    for s in build_rate_samples(
                        self.db,
                        stat,
                        player_ids,
                        include_advanced=True,
                        store=self.store,
                        max_games_per_player=per_player_cap,
                    )
                    if s["game_date"] < game_date
                ][-self.ML_SAMPLE_CAP :]
                if len(rs) >= self.MIN_STAT_MODEL_SAMPLES:
                    self._stat_models[stat] = build_stat_model(stat).fit(rs)
            except Exception:
                continue

    def _projected_minutes(self, feats: dict) -> Optional[float]:
        min_base = None
        for key in ("minutes_ewma", "minutes_l7", "minutes_l20", "minutes_l3"):
            if feats.get(key) is not None:
                min_base = float(feats[key])
                break
        if min_base is None:
            return None
        if self._minutes_model is not None:
            try:
                min_ml = _clamp(
                    self._minutes_model.predict(feats),
                    0.5 * min_base,
                    1.45 * min_base,
                )
                return self.ml_weight * min_ml + (1 - self.ml_weight) * min_base
            except Exception:
                pass
        return min_base

    def _multi_team_data(self) -> bool:
        if self._multi_team is None:
            row = self.db.query_one(
                "SELECT COUNT(DISTINCT team_id) AS n FROM fact_player_games"
            )
            self._multi_team = bool(row and row["n"] >= 2)
        return self._multi_team

    def _context_multiplier(self, feats: dict, stat: str, ctx: dict) -> float:
        """DvP + pace + efficiency/rest adjustments, clamped [0.85, 1.18]."""
        vuln = (
            self.store.matchup_vulnerability(
                ctx["opp_id"],
                stat,
                ctx["game_date"],
                position=ctx.get("position"),
            )
            if ctx.get("opp_id") is not None
            else None
        )
        mult = 1.0 + self.dvp_weight * ((vuln if vuln is not None else 0.5) - 0.5)
        pace = self.store.expected_pace_factor(
            ctx.get("team_id"), ctx.get("opp_id"), ctx["game_date"]
        )
        mult *= pace
        if stat in ("pts", "fg3m"):
            mult += -0.15 * (feats.get("ts_regression_delta") or 0.0)
            role = _clamp(feats.get("role_expansion_delta") or 0.0, -0.2, 0.2)
            mult += 0.10 * role
        # Schedule context only when the minutes model isn't active (the
        # fitted model already internalizes rest) and real multi-team data
        # exists — keeps single-team fixtures neutral.
        if self._minutes_model is None and self._multi_team_data():
            if feats.get("is_b2b"):
                mult -= 0.025
            if feats.get("is_3_in_4"):
                mult -= 0.015
            if feats.get("altitude_game"):
                mult -= 0.02
            if (feats.get("days_rest") or 0) >= 3:
                mult += 0.015
        return _clamp(mult, 0.85, 1.18)

    def _projected_mean(
        self, feats: dict, stat: str, minutes: float, ctx: Optional[dict] = None
    ) -> Optional[float]:
        from src.models.stat_models import _rate_features

        ctx = ctx or {}
        stats = COMBO_STATS.get(stat, (stat,))
        total = 0.0
        for s in stats:
            rate = feats.get(f"{s}_per_min_l20")
            if rate is None:
                ewma, l7 = feats.get(f"{s}_ewma"), feats.get("minutes_l7")
                if ewma is not None and l7:
                    rate = ewma / max(l7, 1.0)
            if rate is None:
                return None
            mu_s = rate * minutes
            model = self._stat_models.get(s)
            if model is not None:
                try:
                    rate_feats = _rate_features(feats, s, include_advanced=True)
                    count_ml = model.predict_count(rate_feats, minutes)
                    mu_s = self.ml_weight * count_ml + (1 - self.ml_weight) * mu_s
                except Exception:
                    pass
            total += mu_s * self._context_multiplier(feats, s, ctx)
        return total

    def _factor_block(
        self,
        feats: dict,
        opp_id: Optional[int],
        stat: str,
        game_date: str,
        minutes: float,
        position: Optional[str] = None,
        team_id: Optional[int] = None,
    ) -> dict:
        vuln = (
            self.store.matchup_vulnerability(
                opp_id,
                stat if stat in ("pts", "reb", "ast", "fg3m") else "pts",
                game_date,
                position=position,
            )
            if opp_id is not None
            else None
        )
        dvp_rank = round(vuln * 29) + 1 if vuln is not None else None
        days_rest = feats.get("days_rest")
        ts_pct = feats.get("ts_pct_l7")
        usg_pct = feats.get("usg_pct_l20")
        game_score = feats.get("game_score_l7")
        return {
            "projected_minutes": round(minutes, 1),
            "dvp_rank": dvp_rank,
            "dvp_label": (
                f"Opponent {self._ordinal(dvp_rank)} vs {stat.upper()}"
                if dvp_rank
                else None
            ),
            "rest_days": days_rest,
            "rest_label": (
                "Back-to-back"
                if feats.get("is_b2b")
                else (
                    f"{days_rest} day{'s' if days_rest != 1 else ''} rest"
                    if days_rest is not None
                    else None
                )
            ),
            "altitude": bool(feats.get("altitude_game", 0)),
            "ts_pct": round(ts_pct, 3) if ts_pct is not None else None,
            "usg_pct": round(usg_pct, 1) if usg_pct is not None else None,
            "pace_factor": round(
                self.store.expected_pace_factor(team_id, opp_id, game_date), 3
            ),
            "ml_active": bool(self._minutes_model or self._stat_models),
            "game_score": round(game_score, 1) if game_score is not None else None,
        }

    def _inactive_this_season(self, player_id: int, game_date: str) -> bool:
        """True if the player's team has played this season but he has not
        (injury/return ramp the market prices in but our stats can't see)."""
        year = int(str(game_date)[:4])
        month = int(str(game_date)[5:7])
        season_start = f"{year if month >= 7 else year - 1}-10-01"
        player_row = self.db.query_one(
            "SELECT team_id FROM dim_players WHERE player_id = ?",
            (player_id,),
        )
        team_id = player_row["team_id"] if player_row else None
        if not team_id:
            return False
        team_played = self.db.query_one(
            "SELECT 1 FROM fact_player_games g JOIN dim_players p "
            "ON p.player_id = g.player_id "
            "WHERE p.team_id = ? AND g.game_date >= ? LIMIT 1",
            (team_id, season_start),
        )
        if not team_played:
            return False
        return (
            self.db.query_one(
                "SELECT 1 FROM fact_player_games "
                "WHERE player_id = ? AND game_date >= ? LIMIT 1",
                (player_id, season_start),
            )
            is None
        )

    @staticmethod
    def _confidence(
        feats: dict,
        stat: str,
        history_games: int,
        inactive_this_season: bool = False,
    ) -> tuple[bool, list[str]]:
        """Gate recommendations on projection trustworthiness.

        Low confidence when (a) the box-score window is thin (<8 games),
        (b) recent form diverges sharply from long-run form — the
        signature of a role change (trade, injury return, bench->starter)
        where last-season rates are stale relative to the market — or
        (c) the player has no games this season while his team does.
        """
        reasons = []
        if history_games < 8:
            reasons.append(f"thin sample ({history_games} games)")
        if inactive_this_season:
            reasons.append("no games played this season")

        parts = COMBO_STATS.get(stat, (stat,))
        recent = sum(feats.get(f"{s}_l3") or 0.0 for s in parts)
        long_run = sum(feats.get(f"{s}_l20") or 0.0 for s in parts)
        if (
            feats.get("minutes_l3") is not None
            and long_run > 0
            and abs(recent - long_run) / long_run > 0.30
        ):
            reasons.append("role/form shift detected")

        return bool(reasons), reasons

    @staticmethod
    def _ordinal(n: int) -> str:
        if 10 <= n % 100 <= 20:
            return f"{n}th"
        return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"

    def generate_plays(
        self, game_date: Optional[str] = None, injury_entries=None
    ) -> list[dict]:
        """Evaluate every latest consensus line into a play card."""
        game_date = str(game_date or _date.today().isoformat())[:10]
        lines = self.latest_consensus_lines(game_date)
        slate_pids = sorted(
            {l["player_id"] for l in lines if l["player_id"] is not None}
        )
        self._ensure_ml_models(game_date, player_ids=slate_pids or None)

        # Vacated usage: resolved once per team when an injury report is
        # supplied — out teammates push minutes (and usage) to the rest.
        vac_engine = None
        vacated_by_team: dict = {}
        out_players: set = set()
        if injury_entries:
            from src.features.vacated_usage import VacatedUsageEngine

            vac_engine = VacatedUsageEngine(self.db)

        def vacated(team_id):
            if team_id not in vacated_by_team:
                out_ids = (
                    vac_engine.resolve_out_players(injury_entries, team_id)
                    if vac_engine and team_id
                    else set()
                )
                out_players.update(out_ids)
                adj = (
                    vac_engine.redistribute(team_id, sorted(out_ids), game_date)[
                        "adjustments"
                    ]
                    if out_ids
                    else {}
                )
                vacated_by_team[team_id] = adj
            return vacated_by_team[team_id]

        plays = []
        for line_row in lines:
            stat = MARKET_TO_STAT.get(line_row["market"])
            player_id = line_row["player_id"]
            if stat is None or player_id is None:
                continue

            player = self.db.query_one(
                "SELECT p.*, t.abbreviation AS team_abbr FROM dim_players p "
                "LEFT JOIN dim_teams t ON t.team_id = p.team_id "
                "WHERE p.player_id = ?",
                (player_id,),
            )
            team_abbr = line_row["player_team_abbr"] or (
                player["team_abbr"] if player else None
            )
            if (
                line_row["home_team_abbr"]
                and line_row["away_team_abbr"]
                and team_abbr
                not in (line_row["home_team_abbr"], line_row["away_team_abbr"])
            ):
                # An inconsistent or obsolete source row cannot identify a
                # valid player matchup. Do not publish a guessed opponent.
                continue
            opp_abbr = (
                line_row["away_team_abbr"]
                if team_abbr == line_row["home_team_abbr"]
                else (line_row["home_team_abbr"] or line_row["away_team_abbr"])
            )
            opp_id = self.db.team_id_for_abbr(opp_abbr) if opp_abbr else None
            is_home = bool(team_abbr and team_abbr == line_row["home_team_abbr"])
            team_id = player["team_id"] if player else None
            position = player["position"] if player else None

            feats = self.store.rolling_features(player_id, game_date)
            ctx_game = self.store.game_context(
                game_date, opponent_team_id=opp_id, is_home=is_home
            )
            feats["is_home"] = int(is_home)
            feats["altitude_game"] = ctx_game["altitude_game"]

            minutes = self._projected_minutes(feats)
            if minutes is None:
                continue
            vac_map = vacated(team_id) if team_id else {}
            if player_id in out_players:
                continue  # listed OUT on the injury report
            vac_adj = vac_map.get(player_id)
            if vac_adj:
                minutes += vac_adj["extra_minutes"]

            ctx = {
                "opp_id": opp_id,
                "team_id": team_id,
                "game_date": game_date,
                "position": position,
            }
            mu = self._projected_mean(feats, stat, minutes, ctx)
            if mu is None:
                continue

            history = self._stat_history(player_id, stat, game_date)
            if line_row["line"] is not None and self.prior_games > 0:
                n = len(history)
                mu = (n * mu + self.prior_games * line_row["line"]) / (
                    n + self.prior_games
                )
            low_conf, conf_reasons = self._confidence(
                feats,
                stat,
                len(history),
                inactive_this_season=self._inactive_this_season(player_id, game_date),
            )
            minutes_std = feats.get("minutes_std_l20") or 0.0
            mc_w = self.mc_weight if len(history) >= 15 else 0.0
            pmf = (
                self.dist.blended_pmf(
                    mu,
                    history=history or None,
                    minutes_mean=minutes,
                    minutes_std=minutes_std,
                    rate_mean=mu / max(minutes, 1.0),
                    weight_mc=mc_w,
                )
                if mc_w > 0
                else self.dist.neg_binomial_pmf(mu, history=history or None)
            )
            ev = self.ev.evaluate_line(
                pmf,
                line_row["line"],
                over_price=line_row["over_price"],
                under_price=line_row["under_price"],
            )
            ladder = self.ladder.build_ladder(
                pmf,
                consensus_line=line_row["line"],
                model_mean=mu,
                player_name=line_row["player_name"],
                market=line_row["market"],
            )
            if low_conf:
                ladder = {**ladder, "triggered": False}
            suppressed = stat in self.suppressed_markets
            if suppressed:
                ladder = {**ladder, "triggered": False}

            plays.append(
                {
                    "player_name": line_row["player_name"],
                    "player_id": player_id,
                    "position": player["position"] if player else None,
                    "team_abbr": team_abbr,
                    "event_id": line_row["event_id"],
                    "game_date": line_row["game_date"],
                    "home_team_abbr": line_row["home_team_abbr"],
                    "away_team_abbr": line_row["away_team_abbr"],
                    "commence_time": line_row["commence_time"],
                    "opponent_abbr": opp_abbr,
                    "is_home": bool(is_home),
                    "market": line_row["market"],
                    "stat": stat,
                    "line": line_row["line"],
                    "history_games": len(history),
                    "model_mean": round(mu, 2),
                    "model_edge": round(mu - line_row["line"], 2),
                    "p_over": round(ev["p_over"], 4),
                    "p_under": round(ev["p_under"], 4),
                    "over_price": line_row["over_price"],
                    "under_price": line_row["under_price"],
                    "ev_over_pct": ev["ev_over_pct"],
                    "ev_under_pct": ev["ev_under_pct"],
                    "fair_over_american": ev["fair_over_american"],
                    "recommended_side": (
                        None if (low_conf or suppressed) else ev["recommended_side"]
                    ),
                    "recommended_ev_pct": (
                        None if (low_conf or suppressed) else ev["recommended_ev_pct"]
                    ),
                    "low_confidence": low_conf,
                    "confidence_reasons": conf_reasons,
                    "market_suppressed": suppressed,
                    "factors": self._factor_block(
                        feats,
                        opp_id,
                        stat,
                        game_date,
                        minutes,
                        position=position,
                        team_id=team_id,
                    ),
                    "milestone": {
                        "triggered": ladder["triggered"],
                        "base_over": ladder["base_over"],
                        "milestones": ladder["milestones"],
                    },
                }
            )

        plays.sort(
            key=lambda p: max(p["ev_over_pct"] or -999, p["ev_under_pct"] or -999),
            reverse=True,
        )
        return plays


class StaticSiteBuilder:
    """Writes daily_slate.json + copies the vanilla web bundle."""

    def __init__(self, out_dir: "str | Path" = "site"):
        self.out_dir = Path(out_dir)

    def build(self, slate: dict) -> dict:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        slate = dict(slate)
        slate["generated_at"] = utcnow_iso()
        slate["total_plays"] = len(slate.get("plays", []))
        slate["recommended_count"] = sum(
            1 for p in slate["plays"] if p.get("recommended_side")
        )
        slate["milestone_count"] = sum(
            1 for p in slate["plays"] if p.get("milestone", {}).get("triggered")
        )

        (self.out_dir / "daily_slate.json").write_text(json.dumps(slate, indent=2))
        for asset in ("index.html", "styles.css", "app.js"):
            shutil.copy2(WEB_DIR / asset, self.out_dir / asset)
        return {
            "out_dir": str(self.out_dir),
            "files": sorted(p.name for p in self.out_dir.iterdir()),
            "total_plays": slate["total_plays"],
            "recommended_count": slate["recommended_count"],
            "milestone_count": slate["milestone_count"],
        }
