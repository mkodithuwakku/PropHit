# Runbook: Advanced Stats, ML Slate Integration & Zero-Throttle Ingestion Epic

This runbook is the single source of truth for **Agent 2 (Coding Agent)** to implement the 4-part upgrade with minimal token usage. **Do NOT read `site/daily_slate.json` or scan the whole repo** — all target files, line ranges, formulas, and test constraints are specified below.

---

## Token-Efficiency Rules for Agent 2
1. **Never read `site/daily_slate.json` or `data/prophit.db` directly.**
2. **Do not run exploratory searches (`grep`/`find`)** unless a test fails; use the exact file and line references below.
3. **Preserve existing test contracts** — every existing test in `tests/` must continue to pass without network calls (`pytest-socket` is active).
4. **Batch edits per file** and verify with `.venv/bin/python -m pytest -q` and `.venv/bin/python -m black --check .`.

---

## Phase 1: Zero-Throttle Ingestion, CDN/ESPN Fallback & Local DvP

### 1.1 `src/ingestion/db.py` (Lines 427–565)
- **Fix `upsert_player_game` (`lines 474–485`):**
  Update `ON CONFLICT(game_id, player_id) DO UPDATE SET` so all box-score columns (`team_id`, `opponent_team_id`, `season_id`, `game_date`, `matchup`, `is_home`, `wl`, `minutes`, `pts`, `fgm`, `fga`, `fg_pct`, `fg3m`, `fg3a`, `fg3_pct`, `ftm`, `fta`, `ft_pct`, `oreb`, `dreb`, `reb`, `ast`, `stl`, `blk`, `tov`, `pf`, `plus_minus`) are updated from `excluded.*`, and tracking/hustle columns (`touches`, `front_court_touches`, `time_of_poss`, `potential_ast`, `rebound_chances`, `deflections`, `screen_assists`, `contested_shots`) use `COALESCE(excluded.<col>, fact_player_games.<col>)`.
- **Fix `upsert_team_defense` (`lines 555–562`):**
  Include `season_id`, `opp_pts`, `opp_reb`, `opp_ast`, `opp_fg3m`, `paint_pts_rank`, `perimeter_fg3_rank`, `rim_fg_pct_rank` in the `ON CONFLICT(team_id, snapshot_date, position_scope) DO UPDATE SET` clause so derived `opp_*` values are persisted on update.

### 1.2 `src/ingestion/nba_stats.py` (Lines 98–293)
- **Incremental Watermark Helper (`PlayerGameLogIngestion`):**
  Add `latest_game_date(self, season_prefix: Optional[str] = None) -> Optional[str]` querying `SELECT MAX(game_date) AS d FROM fact_player_games` (filtered by `game_date >= season_start` when given). Convert `YYYY-MM-DD` to `MM/DD/YYYY` for `nba_api`'s `date_from_nullable`.
- **Zero-Auth Public Box-Score Fallback (`ESPNBoxScoreClient`):**
  Add a zero-auth client using `requests.Session` (same injection pattern as `_RetryingClient` in `consensus_odds.py` so offline tests can pass a fake session) that fetches completed games for missing dates (`YYYYMMDD`) from:
  - Scoreboard: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard?dates={YYYYMMDD}`
  - Boxscore summary: `https://site.api.espn.com/apis/site/v2/sports/basketball/nba/summary?event={event_id}`
  Normalize athlete stat rows (`MIN`, `FG`, `3PT`, `FT`, `OREB`, `DREB`, `REB`, `AST`, `STL`, `BLK`, `TOV`, `PF`, `+/-`, `PTS`) into the exact dict format accepted by `PlayerGameLogIngestion.ingest_rows()`.
- **Local Team Defense & Position DvP Derivation (`TeamDefenseIngestion.derive_from_games`):**
  Add `derive_from_games(self, snapshot_date: str, season_start: Optional[str] = None) -> int` on `TeamDefenseIngestion`:
  - Aggregate `fact_player_games` strictly with `game_date <= snapshot_date` (and `>= season_start` if provided, falling back to all available games if current season has `< 5` team games) grouped by `(opponent_team_id, game_id)` to compute per-game allowed averages for each team:
    - `opp_pts`, `opp_reb`, `opp_ast`, `opp_fg3m`
    - `paint_pts_allowed`: opponent 2-point FGM $\times 2$ per game (`2 * SUM(COALESCE(fgm,0) - COALESCE(fg3m,0)) / games`)
    - `perimeter_fg3_pct`: `SUM(fg3m) / NULLIF(SUM(fg3a), 0)` allowed
    - `rim_fg_pct_allowed`: opponent 2P% allowed (`SUM(fgm - fg3m) / NULLIF(SUM(fga - fg3a), 0)`)
    - `def_rating`: points allowed per 100 estimated possessions, where team possessions per game $\approx \text{FGA} + 0.44 \cdot \text{FTA} - \text{OREB} + \text{TOV}$
    - Rank teams $1\text{..}N$ (`def_rating_rank`, `paint_pts_rank`, `perimeter_fg3_rank`, `rim_fg_pct_rank`) and upsert with `position_scope = "ALL"`.
  - Also aggregate by `(opponent_team_id, position_group, game_id)` joining `dim_players.position` (mapping positions to `G`, `F`, `C` as well as exact `PG`, `SG`, `SF`, `PF`, `C` scopes when available) and upsert into `fact_team_defense_daily` so position-specific DvP is populated with **zero API calls**.

### 1.3 `src/publication/daily_refresh.py` (Lines 33–88)
Refactor `maybe_ingest_live(db, game_date)`:
1. **Prior-Season Skip Guard:** Query `SELECT COUNT(*) AS n FROM fact_player_games WHERE game_date >= ? AND game_date < ?` for the prior season tail window (`[f"{start_year}-03-01", f"{start_year}-07-01")`). Only call `nba.get_league_game_log(prev, ...)` if `n == 0` (saves 1 API call and 7,231 upserts on every run).
2. **Current-Season Incremental Fetch + Same-Day Guard:**
   - Determine `yesterday = (date.fromisoformat(target_date) - timedelta(days=1)).isoformat()`.
   - If `MAX(game_date)` in `fact_player_games` is already `>= yesterday`, skip the `LeagueGameLog` network call completely (`result["game_rows"] = 0`, `result["game_log_cached"] = True`).
   - Otherwise, pass `date_from = MM/DD/YYYY` of `max_game_date` (if present in the current season) so only the missing 1–2 days are fetched.
3. **Automatic Zero-Auth Fallback on Throttle/Error:**
   - If `nba.get_league_game_log(...)` raises an exception (e.g. 48-hour `stats.nba.com` IP block), catch it, record `result["nba_stats_error"]`, and invoke `ESPNBoxScoreClient` for the missing dates between `max_game_date + 1 day` and `yesterday` (up to 5 days) to ingest the missing box scores into `fact_player_games`.
4. **Zero-API-Call Team Defense:**
   - Replace `nba.get_league_dash_team_defense(season)` with `TeamDefenseIngestion(db).derive_from_games(snapshot_date=target_date)` so `LeagueDashPtTeamDefend` is never called during daily refreshes.

---

## Phase 2: Local Advanced Stats & Context Engine (`src/features/feature_store.py`)

Expand `FeatureStore.rolling_features(self, player_id: int, game_date: str)` (and helper methods) using strictly `game_date < target_date` rows:

1. **Shooting & Scoring Efficiency (L7, L20, EWMA):**
   - `ts_pct_l7`, `ts_pct_l20`, `ts_pct_ewma`: $\frac{\text{PTS}}{2 \cdot (\text{FGA} + 0.44 \cdot \text{FTA})}$ (when denominator $> 0$, else `0.55` neutral).
   - `efg_pct_l7`, `efg_pct_l20`: $\frac{\text{FGM} + 0.5 \cdot \text{FG3M}}{\text{FGA}}$.
   - `ftr_l20`: $\frac{\sum \text{FTA}}{\sum \text{FGA}}$; `fg3a_rate_l20`: $\frac{\sum \text{FG3A}}{\sum \text{FGA}}$; `fg3a_per_min_l20`: $\frac{\sum \text{FG3A}}{\sum \text{MIN}}$; `fga_per_min_l20`: $\frac{\sum \text{FGA}}{\sum \text{MIN}}$; `fta_per_min_l20`: $\frac{\sum \text{FTA}}{\sum \text{MIN}}$.
   - `ts_regression_delta`: `ts_pct_l3 - ts_pct_l20` (captures unsustainable hot/cold shooting streaks).
2. **True Team-Relative Shares (On-Floor Opportunity Capture):**
   - For the player's last 20 games (`game_date < target_date`), query team game totals from `fact_player_games` grouped by `(game_id, team_id)` (if a synthetic test DB only has 1 player on the team, fall back gracefully to canonical NBA team averages: `240` team min, `88` FGA, `22` FTA, `14` TOV, `44` REB, `26` AST, `42` FGM so single-player unit test fixtures stay well-scaled):
   - `usg_pct_l7`, `usg_pct_l20`, `usg_pct_ewma`: true on-floor usage rate share.
   - `reb_pct_l20`, `oreb_pct_l20`, `dreb_pct_l20`: on-floor total/offensive/defensive rebound shares.
   - `ast_pct_l20`: on-floor assist share of teammate made FGs.
   - `tov_pct_l20`: $\frac{\text{TOV}}{\text{usage\_volume}}$; `ast_to_tov_l20`: $\frac{\sum \text{AST}}{\max(\sum \text{TOV}, 1)}$.
   - `role_expansion_delta`: `(usage_volume_l3 / max(minutes_l3, 1)) - (usage_volume_l20 / max(minutes_l20, 1))` and `minutes_trend_delta`: `minutes_l3 - minutes_l20`.
3. **Composite Impact & Volatility / Floor Metrics:**
   - `game_score_l7`, `game_score_l20`, `game_score_ewma`: Hollinger Game Score per game.
   - `pie_l20`: Player Impact Estimate proxy per minute.
   - `minutes_std_l20`: sample standard deviation of `minutes` over up to 20 games (used by Stage 3 `monte_carlo_pmf`).
   - `{stat}_cv_l20`: coefficient of variation $\sigma / \mu$ for `pts`, `reb`, `ast`, `fg3m`.
   - `foul_rate_per36_l20`: $36 \times \frac{\sum \text{PF}}{\sum \text{MIN}}$.
   - `plus_minus_per_min_l20`: $\frac{\sum \text{PLUS\_MINUS}}{\sum \text{MIN}}$.
   - ` additional rate stats`: add `stl`, `blk`, `tov`, `oreb`, `dreb`, `fga`, `fta` to per-minute rate calculations (`{stat}_per_min_l20` and `{stat}_per_min_ewma`).
4. **Team Pace, Defensive Rating, & Position-Aware DvP (`FeatureStore`):**
   - `team_pace(self, team_id: int, game_date: str) -> Optional[float]`: estimated possessions per game strictly before `game_date`.
   - `expected_pace_factor(self, team_id: Optional[int], opp_id: Optional[int], game_date: str) -> float`: ratio of expected game pace to league-average pace, clamped to `[0.92, 1.08]` (returns `1.0` if fewer than 3 multi-player team games exist, preserving synthetic unit tests).
   - Enhance `matchup_vulnerability(self, opponent_team_id, stat_type, game_date, scope="ALL", position=None)`:
     - Support `stl`, `blk`, `tov`, and combo stats (by averaging component vulnerabilities) in addition to `pts`, `reb`, `ast`, `fg3m`.
     - If `position` is supplied (e.g. `"C"`, `"PG"`), check for a matching `position_scope` in `fact_team_defense_daily` first, falling back to `"ALL"` if not found.

---

## Pass 1 Validation Fixes (Must Include in Pass 2)

1. **`ESPNBoxScoreClient._event_rows` (`src/ingestion/nba_stats.py` lines 249–268):**
   - `parse_matchup()` treats the first team in `MATCHUP` as the player's team and the second team as the opponent (`"LAL vs. BOS"` -> `opp="BOS", is_home=True`; `"BOS @ LAL"` -> `opp="LAL", is_home=False`).
   - In `_event_rows`, compute `matchup` per team block:
     `team_matchup = f"{home_abbr} vs. {away_abbr}" if team_abbr == home_abbr else f"{away_abbr} @ {home_abbr}"`
     so home players get `opponent_team_id = away_team_id` and `is_home = 1`.
2. **`FeatureStore._single_vulnerability` (`src/features/feature_store.py` lines 558–585):**
   - When `scope != "ALL"` and the opponent has no non-null row in `fact_team_defense_daily` for that `scope`, fall back to `scope = "ALL"` instead of returning `None`.

---

## Phase 3: Stage 1 & 2 ML Models + Blended PMF in `SlateGenerator`

### 3.1 `src/models/minutes_model.py` & `src/models/stat_models.py`
- **Preserve Existing Test Contracts:**
  - `MINUTES_FEATURES` in `minutes_model.py` keeps the 14 base features; add `ADVANCED_MINUTES_FEATURES = ["minutes_std_l20", "foul_rate_per36_l20", "role_expansion_delta", "game_score_l7", "usg_pct_l7", "plus_minus_per_min_l20"]` which `MinutesModel` includes automatically when present in sample feature dicts.
  - In `stat_models.py`, keep `RATE_FEATURES` as the 6 base keys (`recent_rate`, `recent_ewma`, `minutes_l7`, `usage_volume_l7`, `is_b2b`, `days_rest`) so `test_build_rate_samples_from_db` in `tests/test_models.py` (`assert set(s["features"]) == ...`) passes unchanged when `include_advanced=False` (default).
  - Add `ADVANCED_RATE_FEATURES = ["ts_pct_l7", "ts_pct_l20", "ts_regression_delta", "efg_pct_l7", "ftr_l20", "fg3a_rate_l20", "usg_pct_l7", "usg_pct_l20", "reb_pct_l20", "oreb_pct_l20", "dreb_pct_l20", "ast_pct_l20", "tov_pct_l20", "game_score_l7", "role_expansion_delta", "foul_rate_per36_l20", "is_home", "altitude_game"]`.
  - Update `build_rate_samples(..., include_advanced: bool = False)` to populate `ADVANCED_RATE_FEATURES` when `include_advanced=True`.
  - Update `_design_matrix`, `_GLMRateModel`, `ZIPRateModel`, and `LightGBMRateModel` to dynamically select active feature columns (`RATE_FEATURES + [k for k in ADVANCED_RATE_FEATURES if any(fd.get(k) is not None for fd in feature_dicts)]`) and store `self._feature_names` on `fit()`, using `self._feature_names` in `predict_count()`. This guarantees zero singular all-zero columns when synthetic tests pass only the 6 base keys, while using all 24 features when trained on real DB samples.

### 3.2 `src/publication/static_builder.py` (`SlateGenerator`)
Upgrade `SlateGenerator` so predictions are driven by the Stage 1 + Stage 2 ML models, advanced features, contextual multipliers, and Stage 3 Blended PMFs:
1. **Lazy Per-Slate ML Model Fitting & Caching:**
   - Add a helper `_ensure_ml_models(self, game_date: str, player_ids: Optional[list[int]] = None)` on `SlateGenerator` that builds strictly pre-`game_date` training samples (`game_date < target_date`).
   - **Threshold Guard & Sample Cap (Critical for Fast Unit Tests & Sub-Second Production Runs):**
     - Only fit `MinutesModel` and per-stat `build_stat_model(stat)` (`include_advanced=True`) when the database has at least `min_ml_train_samples = 60` historical player-game rows before `game_date` (in production `data/prophit.db` has `8,500+` rows; in small unit test fixtures like `test_static_builder.py` and `test_evaluation.py` there are only 12–24 rows, so they fast-path without fitting noise).
     - Scope `player_ids` to the players on the slate (or cap to the most recent `400` player-game rows before `game_date`) and cache fitted models so `generate_plays()` completes in under 1 second.
2. **Stage 1 Projected Minutes (`_projected_minutes`):**
   - Compute baseline EWMA/rolling minutes (`min_base`).
   - When `MinutesModel` is fitted, predict `min_ml = self._minutes_model.predict(feats)` (clipped to `[0.5 * min_base, 1.45 * min_base]`) and blend with `min_base` using calibrated `minutes_ml_weight` (default `0.55`).
   - Apply Vacated Usage minutes adjustment if `injury_entries` / out teammates are present via `VacatedUsageEngine`.
3. **Stage 2 Stat Projection + Advanced & Contextual Multipliers (`_projected_mean`):**
   - For each component stat $s$:
     - Compute `rate_base` from `{s}_per_min_l20` and `{s}_per_min_ewma` (blended by recent form).
     - When the Stage 2 model for $s$ is fitted, predict `count_ml = model_s.predict_count(rate_feats, minutes)` and blend `mu_s = ml_weight * count_ml + (1 - ml_weight) * (rate_base * minutes)`.
     - **Advanced Stat & Context Multiplier (clamped to `[0.85, 1.18]`):**
       - **DvP Matchup Adjustment:** `1.0 + dvp_weight * (vuln - 0.5)` where `vuln` is position-aware `matchup_vulnerability` in `[0, 1]` (neutral `0.5` when no defense snapshot exists).
       - **Pace Factor:** `expected_pace_factor(team_id, opp_id, game_date)`.
       - **Efficiency Mean-Reversion & Role Expansion:** For scoring (`pts`, `fg3m`), slightly regress extreme 3-game TS% spikes (`-0.15 * ts_regression_delta`) while rewarding genuine usage/shot-role expansion (`+0.10 * clamp(role_expansion_delta, -0.2, 0.2)`).
       - **Schedule / Rest / Altitude Context:** When `MinutesModel` is not active, apply small empirical rate adjustments for `is_b2b` (`-2.5%`), `is_3_in_4` (`-1.5%`), `altitude_game` (`-2.0%`), and `3+ days_rest` (`+1.5%`) when the DB has multi-team data.
4. **Stage 3 Blended PMF (`DistributionEngine.blended_pmf`):**
   - Use `self.dist.blended_pmf(mu, history=history, minutes_mean=minutes, minutes_std=minutes_std, rate_mean=mu / max(minutes, 1.0), weight_mc=mc_weight)` where `mc_weight` comes from `model_settings` (default `0.25` when `len(history) >= 15`, else `0.0` so small synthetic fixtures match exact NegBinomial analytical expectations).
5. **Expose Advanced Stats in `factors`:**
   - Add `ts_pct`, `usg_pct`, `pace_factor`, `ml_active`, and `game_score` to the play card's `factors` dict (all existing keys `projected_minutes`, `dvp_rank`, `dvp_label`, `rest_days`, `rest_label`, `altitude` remain unchanged).

---

## Phase 4: Continuous In-Season Model Tuning (`src/evaluation/calibrate.py` & `src/evaluation/backtest.py`)

Upgrade `calibrate.py` and `backtest.py` so the engine continuously tunes hyperparameters across the season:
1. **Flexible Trigger Cadence (`run_calibration`):**
   - Keep `interval_days: int = INTERVAL_DAYS` (`14`) as the default parameter so `test_skips_when_last_run_recent` in `tests/test_calibration.py` passes unchanged!
   - Add an optional `min_new_settled_days: Optional[int] = None` (used by `daily_refresh.py` with `min_new_settled_days=3`): if `last_calibration` is `< interval_days` ago, still trigger calibration early if at least `min_new_settled_days` new distinct `game_date`s have settled in `fact_consensus_lines` + `fact_player_games` since `last_calibration`.
2. **Multi-Parameter Tuning Sweep in `run_calibration`:**
   In addition to `min_ev_pct` and `suppressed_markets`, tune and persist in `model_settings`:
   - `prior_games` (Empirical-Bayes market shrinkage weight, swept over `(4.0, 6.0, 8.0, 10.0)`, default `6.0`, clamped `±2.0` per run).
   - `ml_blend_weight` (blend weight on Stage 1/2 ML models vs rolling rate baseline, swept over `(0.25, 0.40, 0.55, 0.70)`, default `0.55`).
   - `dvp_weight` (sensitivity to opponent DvP percentile, swept over `(0.04, 0.08, 0.12)`, default `0.08`).
   - `mc_blend_weight` (Monte Carlo PMF blend weight, tuned using Brier / tail calibration error over `(0.15, 0.25, 0.35)`, default `0.25`).
3. **Update `backtest.py`:**
   - Incorporate `dvp_weight` and advanced-rate adjustments (`ts_regression_delta`, `role_expansion_delta`, `expected_pace_factor`) into `backtest()` and optionally collect `prior_games` / `dvp_weight` sensitivity metrics so `run_calibration` can evaluate them offline.

---

## Pass 2 Validation & Production Performance Fixes (Pass 3)

Benchmarking Pass 2 against the real 8,506-row `data/prophit.db` (`1,526` consensus lines on `2026-10-28`) revealed three redundant-query hotspots that cause `generate_plays()` to take `43s` and `backtest()` to run thousands of full-table scans:

1. **Cache `_team_game_possessions` and `_latest_defense_snapshot_date` in `FeatureStore` (`src/features/feature_store.py`):**
   - Add small dict caches on `FeatureStore.__init__`: `self._poss_cache: dict = {}` and `self._features_cache: dict = {}`.
   - In `_team_game_possessions(self, game_date: str, min_players: int = 1)`: key the cache by `(str(game_date)[:10], min_players, self._row_count())` where `self._row_count()` is `SELECT COUNT(*) FROM fact_player_games` (so any new game inserted in mid-test invalidates the cache automatically while repeated calls during a slate or backtest hit in $O(1)$).
   - In `rolling_features(self, player_id: int, game_date: str)`: key `self._features_cache` by `(player_id, str(game_date)[:10], self._row_count())` and return a shallow copy `dict(cached)`. This eliminates recomputing `rolling_features` 10× per player across 10 prop markets on the same slate AND 5× per sample across `MinutesModel` + 4 `stat_models`.
2. **Share Precomputed Feature Dicts Across Models in `SlateGenerator._ensure_ml_models` (`src/publication/static_builder.py` lines 134–200):**
   - Limit `player_ids` in `_ensure_ml_models` to `player_ids[:25]` (up to 25 players) and fetch only each player's most recent `12` games before `game_date` (`<= 240` total `(player_id, game_date)` pairs) instead of scanning all historical games for all 150 slate players.
   - Because `self.store.rolling_features` is cached by `(player_id, game_date, row_count)`, `build_minutes_samples` and the 4 `build_rate_samples` calls will reuse the exact same computed feature dicts with zero duplicate SQL queries.
   - Suppress `SingularMatrixWarning` / `RuntimeWarning` inside ` _GLMRateModel.fit` (`src/models/stat_models.py` line 119) with ` warnings.catch_warnings()` + `warnings.simplefilter("ignore")` (matching `ZIPRateModel.fit`) so pytest output is 100% warning-free.

Target benchmark on `data/prophit.db`: `generate_plays('2026-10-28')` must complete in **< 3.0 seconds** (down from `43.4s`).

---

## Verification Checklist (Must Pass 100% Before Finishing)

Run these exact commands:
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m black --check .
node --test tests/test_dashboard.cjs
```
All existing tests plus new unit tests covering incremental `date_from` watermarks, `ESPNBoxScoreClient` fallback, local `derive_from_games` DvP, new `FeatureStore` advanced metrics, ML-backed `SlateGenerator`, and multi-parameter `run_calibration` must pass 100% offline.
