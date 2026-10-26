# Model & Decision Engine Specification

This document details the mathematical formulations and inference rules implemented across `src/features/`, `src/models/`, `src/decision/`, `src/evaluation/`, and `src/publication/static_builder.py`.

---

## 1. Leak-Free Feature Engineering (`src/features/feature_store.py`)

All features for a target `game_date` are computed strictly from historical rows satisfying `fact_player_games.game_date < target_date` (or `fact_team_defense_daily.snapshot_date <= target_date`).

- **Rolling Windows (`L3`, `L7`, `L20`):** Unweighted means over the last 3, 7, and 20 games for all box-score stats (`minutes`, `pts`, `reb`, `ast`, `fg3m`, `stl`, `blk`, `tov`, `oreb`, `dreb`, `fga`, `fta`, `pf`, `plus_minus`).
- **EWMA (`{stat}_ewma`):** Exponentially weighted moving average with half-life $h = 3.0$ games (`values[0]` is most recent):
  $$w_i = 0.5^{i / 3.0}, \quad \text{EWMA} = \frac{\sum_i w_i x_i}{\sum_i w_i}$$
- **Per-Minute Rates (`{stat}_per_min_l20`):** Total stat divided by total minutes over the last 20 games for `pts`, `reb`, `ast`, `fg3m`.
- **Usage Volume (`usage_volume_l7`):** 7-game mean of possessions used per game:
  $$\text{Usage Volume} = \text{FGA} + 0.44 \times \text{FTA} + \text{TOV}$$
- **Rest & Schedule Flags:** `days_rest`, `is_b2b` (`days_rest == 0`), `games_last_4d`, `is_3_in_4` (`games_last_4d >= 3`), `altitude_game` (`1` when playing away at `DEN` or `UTA`), and `blowout_index` ($|\text{Spread}| \times \text{Total} / 220$).
- **DvP Matchup Vulnerability (`matchup_vulnerability`):** Opponent generosity percentile in $[0, 1]$ for `pts`, `reb`, `ast`, `fg3m` (`1.0` = softest defense in the league). Mapped to ordinal rank $1\text{..}30$ via $\text{round}(\text{vuln} \times 29) + 1$.
- **Vacated Usage (`src/features/vacated_usage.py`):** When teammates are marked `out`, their 10-game baseline minutes and usage volume are redistributed to active teammates proportional to each active player's share of active team minutes and usage.

---

## 2. Multi-Stage ML Models (`src/models/`)

### 2.1 Stage 1 — Minutes Model (`src/models/minutes_model.py`)
- **Algorithm:** LightGBM Booster (`objective="huber"`, `n_estimators=200`, `learning_rate=0.05`, `num_leaves=15`, `min_child_samples=5`, `seed=42`). Automatically falls back to a closed-form NumPy Ridge regressor (`_RidgeRegressor`) if `lightgbm` is unavailable.
- **Features (`MINUTES_FEATURES`):** `minutes_l3`, `minutes_l7`, `minutes_l20`, `minutes_ewma`, `pts_l7`, `usage_volume_l7`, `fga_l7`, `pf_l7`, `days_rest`, `is_b2b`, `is_3_in_4`, `games_last_4d`, `is_home`, `altitude_game`.
- **Validation Gate (`walk_forward_mae`):** Expanding-window walk-forward MAE must beat the expanding season-average baseline (`beats_baseline = True`).

### 2.2 Stage 2 — Category Rate Models (`src/models/stat_models.py`)
Shared normalized feature vector (`RATE_FEATURES`): `recent_rate`, `recent_ewma`, `minutes_l7`, `usage_volume_l7`, `is_b2b`, `days_rest`.
- **Points (`pts` -> `LightGBMRateModel`):** LightGBM regressor (`objective="poisson"`, `n_estimators=150`) conditioned directly on `RATE_FEATURES + [minutes]`.
- **Rebounds (`reb` -> `NegativeBinomialRateModel`):** `statsmodels` GLM with `NegativeBinomial(alpha=1.0)` family and $\log(\max(\text{minutes}, 1))$ offset.
- **Assists (`ast` -> `PoissonRateModel`):** `statsmodels` GLM with `Poisson()` family and $\log(\max(\text{minutes}, 1))$ offset.
- **Three-Pointers Made (`fg3m` -> `ZIPRateModel`):** `statsmodels` `ZeroInflatedPoisson` with $\log(\max(\text{minutes}, 1))$ offset fit via BFGS.
- **Other Count Markets (`stl`, `blk`, `tov`):** Default to `PoissonRateModel` in `build_stat_model()`.

### 2.3 Morning Slate Inference & Empirical-Bayes Shrinkage (`src/publication/static_builder.py`)
During daily slate generation (`SlateGenerator.generate_plays`), projections support all 11 single and combo markets (`pts`, `reb`, `ast`, `fg3m`, `pra`, `pr`, `pa`, `ra`, `blk`, `stl`, `tov`):
1. **Projected Minutes:** First non-null value from `(minutes_ewma, minutes_l7, minutes_l20, minutes_l3)`.
2. **Raw Mean ($\mu_{\text{raw}}$):** Sum across component stats of $\text{rate}_s \times \text{minutes}$, where $\text{rate}_s$ is `{s}_per_min_l20` (falling back to `{s}_ewma / max(minutes_l7, 1.0)`).
3. **Empirical-Bayes Shrinkage:** To prevent overreacting on small sample sizes $n = \text{len(history)}$ (up to `history_games = 20`), the raw mean is shrunk toward the bookmaker consensus line $L$ with prior weight $w_0 = 6.0$ (`prior_games`):
   $$\hat{\mu} = \frac{n \cdot \mu_{\text{raw}} + w_0 \cdot L}{n + w_0}$$

---

## 3. Stage 3 — Discrete Probability Distribution (`src/models/distribution.py`)

`DistributionEngine` computes the discrete PMF $P(X = k)$ over integer support $k \in [0, 60]$ (normalized to sum to $1.0$):
- **Parametric Negative Binomial (`neg_binomial_pmf`):**
  - Method-of-moments dispersion $r = \text{clip}\left(\frac{\mu^2}{\sigma^2 - \mu}, 0.05, 10^6\right)$ fit on the player's last 20 games for that stat (or combo stat).
  - When sample variance $\sigma^2 \le \mu$ or $n < 2$, $r = 10^6$ (`POISSON_LIMIT_R`) and the PMF evaluates the exact Poisson distribution.
- **Monte Carlo Simulation (`monte_carlo_pmf`):**
  - 10,000 seeded trials drawing $\text{minutes} \sim \max(\mathcal{N}(\mu_{\min}, \sigma_{\min}), 0)$ and optional Gamma rate volatility $\text{rate} \sim \text{Gamma}$, then sampling counts from $\text{Poisson}(\text{minutes} \times \text{rate})$.
- **Blended PMF (`blended_pmf`):** Weighted average of parametric Negative Binomial and Monte Carlo PMFs (default 50/50).

---

## 4. Stage 4 — EV, Push Handling, Confidence & Milestone Ladders

### 4.1 Push-Aware Expected Value (`src/decision/ev_calculator.py`)
Given a consensus line $L$ and American prices:
- $P(\text{Over } L) = \sum_{k = \lfloor L \rfloor + 1}^{60} P(X = k)$
- $P(\text{Push } L) = P(X = L)$ if $L$ is an integer, else $0.0$
- $P(\text{Under } L) = 1 - P(\text{Over } L) - P(\text{Push } L)$
- Percentage Expected Value with decimal price $d$:
  $$\text{EV}\% = 100 \times \left(P_{\text{win}} \cdot d + P_{\text{push}} - 1\right)$$
- A side is recommended if its odds fall within `[min_american_odds, max_american_odds]` and $\text{EV}\% \ge \text{min\_ev\_pct}$ (default `5.0%`, dynamically tuned by calibration).

### 4.2 Confidence & Market Suppression Gates (`src/publication/static_builder.py`)
Even when raw EV exceeds the threshold, `recommended_side` is forced to `None` and `milestone.triggered` is set to `False` if any gate trips:
1. **Thin Sample:** `history_games < 8`.
2. **Inactive This Season:** The player's team has played in the current season (`game_date >= Oct 1`), but the player has 0 games logged this season.
3. **Role / Form Shift:** $|\text{L3} - \text{L20}| / \text{L20} > 0.30$ (signals trade, rotation change, or injury ramp where L20 per-minute rates are stale).
4. **Market Suppressed:** The stat market is listed in `model_settings.suppressed_markets` by the bi-weekly calibrator.

### 4.3 Milestone / Alt-Line Ladders (`src/decision/milestone_engine.py`)
- **Trigger:** $\hat{\mu} - L \ge 2.0$ (`trigger_diff`) and play is neither low-confidence nor market-suppressed.
- **Thresholds:** Starting at the smallest even integer strictly above $L$ (`_base_threshold(L)`), evaluates offsets `(0, +2, +4)` (e.g., $L = 9.5 \to 10+, 12+, 14+$; $L = 12.5 \to 14+, 16+, 18+$).
- **Pricing:** Converts $P(X \ge k)$ to vig-free `fair_american` odds and adds `+25` (`alt_target_american = fair_american + 25`) as the minimum playable sportsbook price.

---

## 5. Self-Calibration & Backtesting (`src/evaluation/`)

- **Trailing-Mean Proxy Backtest (`backtest.py`):** Synthesizes historical book lines from each player's pre-game L20 mean rounded to the nearest `0.5` at `-110/-110` for games with `minutes >= 10.0` and `>= 10` prior games.
- **Bi-Weekly Calibration (`calibrate.py`):** Runs automatically inside `daily_refresh` if `last_calibration` is `>= 14` days old:
  - Sweeps `EV_LADDER = (5.0, 8.0, 10.0, 12.0, 15.0)` over `bets_detail` (requires `>= 100` bets and `> 1.0%` ROI improvement) and clamps `min_ev_pct` adjustments to `±2.0%` per run.
  - Suppresses any stat market with `>= 50` bets and `ROI <= -5.0%`; automatically unsuppresses when `ROI >= +3.0%`.
- **Settled Play Record (`record.py`):** Re-runs `SlateGenerator` leak-free on past dates that have both consensus lines and box-score actuals, logs evaluations to `fact_model_evaluations`, and outputs `model_stats` (`wins`, `losses`, `pushes`, `units`, `roi_pct`, `favored_hit_rate`) to the dashboard header.
