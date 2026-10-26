# Prophit Architecture

Prophit is structured as a six-module offline-testable Python pipeline paired with a zero-framework vanilla JS/CSS mobile dashboard.

## 1. End-to-End Data Flow

```
[1. Ingestion (src/ingestion/)]
  ├─ ConsensusOddsIngestion: statpick.ai (primary + per-player prop pages)
  │    -> PrizePicks -> Underdog fallback (100% free, zero-auth)
  ├─ NBAStatsClient: LeagueGameLog (current + prior season tail) & LeagueDashPtTeamDefend
  └─ Injuries: parse_injury_entries()
         │
         ▼
[2. SQLite Persistence (data/prophit.db via ProphitDB)]
  ├─ Dimensions: dim_teams, dim_players (auto-promotes synthetic stub IDs -> official NBA IDs)
  ├─ Facts: fact_player_games, fact_team_defense_daily, fact_consensus_lines, fact_model_evaluations
  └─ Metadata: ingestion_runs, model_settings
         │
         ▼
[3. Leak-Free Feature Store (src/features/)]
  ├─ FeatureStore: L3/L7/L20 rolling means, EWMA (halflife=3.0), per-min rates (L20),
  │    usage_volume (FGA + 0.44*FTA + TOV), rest/B2B/3-in-4, altitude, blowout_index, DvP percentile
  └─ VacatedUsageEngine: redistributes OUT teammates' minutes & usage by active share
         │
         ▼
[4. Modeling & Distribution Engine (src/models/)]
  ├─ Stage 1 (minutes_model.py): LightGBM Huber regressor (Ridge fallback)
  ├─ Stage 2 (stat_models.py): LightGBM (pts), NegBin GLM (reb), Poisson GLM (ast), ZIP (fg3m)
  ├─ Slate Inference (static_builder.py): EWMA/rolling minutes × per-min rate + Empirical-Bayes
  │    shrinkage toward consensus line (prior_games=6.0)
  └─ Stage 3 (distribution.py): Discrete PMF P(X = k) for k in [0..60] via NegBin (MoM dispersion r)
       and 10,000-trial Monte Carlo
         │
         ▼
[5. Decision & Self-Calibration (src/decision/ & src/evaluation/)]
  ├─ EVCalculator (ev_calculator.py): push-aware P(Over), P(Under), P(Push), fair odds, EV%
  ├─ MilestoneEngine (milestone_engine.py): even-integer alt ladders when model_mean - line >= 2.0
  ├─ Bi-Weekly Calibrator (calibrate.py): sweeps EV gates (±2% clamp) & suppresses losing markets
  └─ Model Record (record.py) & Proxy Backtest (backtest.py): settles past plays & grades W-L-P / ROI
         │
         ▼
[6. Publication & Hosting (src/publication/ + web/ -> site/)]
  ├─ StaticSiteBuilder: writes site/daily_slate.json + copies web/{index.html,styles.css,app.js}
  └─ GitHub Actions: daily_refresh.yml commits site/ & data/prophit.db -> deploy_pages.yml deploys Pages
```

## 2. Module Responsibilities & Key Entrypoints

| Module | File | Key Classes / Functions | Purpose |
| :--- | :--- | :--- | :--- |
| **Persistence** | `src/ingestion/db.py` | `ProphitDB`, `normalize_player_name`, `synthetic_id` | SQLite connection, schema init, foreign-key safe synthetic-to-official ID promotion (`_promote_player_id`, `_promote_team_id`), settings store. |
| **Consensus Odds** | `src/ingestion/consensus_odds.py` | `StatPickClient`, `PrizePicksClient`, `UnderdogClient`, `ConsensusOddsIngestion` | Zero-auth morning odds pull (`ingest_morning_pull`). Normalizes 11 markets into `fact_consensus_lines` (writes both bookmaker row and `CONSENSUS` mirror row). |
| **NBA Stats** | `src/ingestion/nba_stats.py` | `NBAStatsClient`, `PlayerGameLogIngestion`, `TeamDefenseIngestion` | Rate-limited (`0.6s`), retrying `nba_api` wrapper; injectable `endpoint_caller` for 100% offline tests. |
| **Injuries** | `src/ingestion/injuries.py` | `InjuryEntry`, `parse_injury_entries` | Normalizes injury report rows (`out`, `questionable`, `doubtful`, `probable`, `available`). |
| **Feature Store** | `src/features/feature_store.py` | `FeatureStore`, `ewma`, `usage_volume`, `walk_forward_splits` | Strictly `game_date < target_date` queries. Computes rolling windows, EWMA, per-minute rates, DvP vulnerability (`[0, 1]`), and game context. |
| **Vacated Usage** | `src/features/vacated_usage.py` | `VacatedUsageEngine` | Computes 10-game team baselines and redistributes `OUT` players' minutes and usage volume proportionally. |
| **Minutes Model** | `src/models/minutes_model.py` | `MinutesModel`, `build_minutes_samples`, `walk_forward_mae` | Stage 1 LightGBM Huber regressor over 14 pre-game features with automatic NumPy Ridge fallback. |
| **Stat Models** | `src/models/stat_models.py` | `LightGBMRateModel`, `NegativeBinomialRateModel`, `PoissonRateModel`, `ZIPRateModel`, `build_stat_model` | Stage 2 count/rate models using `log(minutes)` offsets (GLMs/ZIP) or minutes feature conditioning (LightGBM). |
| **Distributions** | `src/models/distribution.py` | `DistributionEngine`, `fit_dispersion`, `prob_at_least`, `prob_over_line`, `pmf_mean` | Stage 3 discrete PMF over `k in [0, 60]` via Negative Binomial (collapsing to Poisson when `r >= 1e6`) and seeded Monte Carlo. |
| **EV Calculator** | `src/decision/ev_calculator.py` | `EVCalculator`, `american_to_decimal`, `decimal_to_american`, `prob_to_fair_american`, `expected_value_pct` | Evaluates `p_over`, `p_under`, `p_push` (on integer lines), fair odds, and `recommended_side` when `EV >= min_ev_pct` within odds bounds. |
| **Milestone Engine** | `src/decision/milestone_engine.py` | `MilestoneEngine`, `_base_threshold` | Prices 3 even-integer ladder rungs (`base + 0, +2, +4`) with fair American odds and `+25` alt target margin when `model_mean - line >= 2.0`. |
| **Proxy Backtest** | `src/evaluation/backtest.py` | `backtest` | Leak-free historical grading against synthetic trailing-L20 lines rounded to `0.5` at `-110/-110`. |
| **Self-Calibration** | `src/evaluation/calibrate.py` | `run_calibration`, `get_min_ev`, `get_suppressed_markets` | Bi-weekly (`14d`) tuning of `min_ev_pct` (`±2.0` step clamp) and automatic suppression/unsuppression of markets based on graded ROI. |
| **Model Record** | `src/evaluation/record.py` | `settled_plays`, `summarize_model_stats` | Reconstructs historical slates leak-free, grades against actual box scores, logs to `fact_model_evaluations`, and summarizes W-L-P, units, ROI%, and favored hit rate. |
| **Static Builder** | `src/publication/static_builder.py` | `SlateGenerator`, `StaticSiteBuilder` | Builds play cards with Empirical-Bayes shrinkage, confidence & suppression gating, factor blocks, and milestone ladders; writes `site/daily_slate.json` + web assets. |
| **Daily Refresh** | `src/publication/daily_refresh.py` | `run_refresh`, `maybe_ingest_live`, `main` | CLI entrypoint orchestrating live ingestion, calibration, slate generation, model record summary, and static site build. |
| **Git Publisher** | `src/publication/publisher.py` | `GitPublisher` | Optional `gh-pages` branch publisher using a temporary git worktree (supports side-effect-free `dry_run=True`). |
| **Frontend UI** | `web/index.html`, `web/app.js`, `web/styles.css` | `renderSlate`, `renderPlays`, `cardHtml`, `miniHtml`, `displayedOutcome` | Mobile-first charcoal/plum dashboard with `Browse`, `Best Picks`, and `Games` views. Copied into `site/` on build. |

## 3. Database Schema (`data/prophit.db`)

- `dim_teams`: `team_id (PK)`, `abbreviation (UNIQUE)`, `team_name`, `team_city`, `conference`, `division`, `updated_at`
- `dim_players`: `player_id (PK)`, `full_name`, `normalized_name (UNIQUE)`, `first_name`, `last_name`, `position`, `team_id (FK)`, `height_inches`, `weight_lbs`, `is_active`, `updated_at`
- `fact_player_games`: `(game_id, player_id) PK`, `team_id`, `opponent_team_id`, `season_id`, `game_date`, `matchup`, `is_home`, `wl`, `minutes`, box-score counts (`pts`, `reb`, `ast`, `fg3m`, `stl`, `blk`, `tov`, `fga`, `fta`, `oreb`, `dreb`, `pf`, `plus_minus`), tracking/hustle columns, `ingested_at`
- `fact_team_defense_daily`: `(team_id, snapshot_date, position_scope) PK`, `def_rating`, `def_rating_rank`, `paint_pts_allowed`, `perimeter_fg3_pct`, `rim_fg_pct_allowed`, `opp_pts`, `opp_reb`, `opp_ast`, `opp_fg3m`, `ingested_at`
- `fact_consensus_lines`: `line_id (PK)`, `event_id`, `game_date`, `home_team_abbr`, `away_team_abbr`, `player_team_abbr`, `player_id (FK)`, `player_name`, `market`, `bookmaker`, `line`, `over_price`, `under_price`, `is_consensus`, `snapshot_time`, `commence_time`, `ingested_at`
- `fact_model_evaluations`: `eval_id (PK)`, `eval_time`, `game_id`, `player_id`, `stat_type`, `model_version`, `projected_mean`, `projected_minutes`, `pmf_json`, `actual_value`
- `ingestion_runs`: `run_id (PK)`, `source`, `started_at`, `finished_at`, `records_written`, `status`, `notes`
- `model_settings`: `key (PK)`, `value`, `updated_at`, `reason` (stores `min_ev_pct`, `suppressed_markets`, `last_calibration`, `calibration_notes`)
