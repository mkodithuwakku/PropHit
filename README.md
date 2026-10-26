# Prophit

Automated NBA player prop and milestone betting engine. Prophit ingests 100% free, zero-auth public data sources (`statpick.ai`, PrizePicks, Underdog Fantasy, and `nba_api` box scores/defensive metrics), computes leak-free pre-game features, projects discrete probability distributions ($P(X = k)$), identifies +EV plays, and prices alternative/milestone ladders (e.g., Bet365 alt lines). Output is a static, mobile-first dashboard deployed daily via GitHub Pages.

See `AGENTS.md` for universal agent rules and `PROPHIT_SPECIFICATION.md` for the full software requirements document.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Verification

All automated tests run 100% offline (`pytest-socket` blocks network access; Node tests use local fixtures).

```bash
.venv/bin/python -m pytest -q          # Python unit + integration suite (100% offline)
.venv/bin/python -m black --check .    # Code formatting check
node --test tests/test_dashboard.cjs   # Frontend UI regression suite (Node 18+)
```

## Common Commands

```bash
# Offline dry-run refresh against existing SQLite data
.venv/bin/python -m src.publication.daily_refresh --db data/prophit.db --out site --dry-run

# Live morning refresh (pulls statpick/PrizePicks/Underdog + nba_api, calibrates, builds site/)
.venv/bin/python -m src.publication.daily_refresh --db data/prophit.db --out site --live

# Convenience wrapper script
.venv/bin/python scripts/daily_morning_run.py --db data/prophit.db --out site --live
```

## Repository Layout

- `src/ingestion/` — Zero-auth consensus odds chain (`consensus_odds.py`), `nba_api` game logs & team defense (`nba_stats.py`), injury parser (`injuries.py`), and SQLite schema/persistence (`db.py`).
- `src/features/` — Leak-free rolling windows (L3/L7/L20), EWMA (`halflife=3`), per-minute rates, usage volume, rest/B2B/3-in-4/altitude/blowout context, DvP percentiles (`feature_store.py`), and OUT teammate redistribution (`vacated_usage.py`).
- `src/models/` — Stage 1 LightGBM Huber minutes regressor with Ridge fallback (`minutes_model.py`), Stage 2 stat-specific rate models (`LightGBMRateModel`, `NegativeBinomialRateModel`, `PoissonRateModel`, `ZIPRateModel` in `stat_models.py`), and Stage 3 Negative Binomial + Monte Carlo PMF engine (`distribution.py`).
- `src/decision/` — American/decimal/implied odds conversions, push-aware +EV evaluator (`ev_calculator.py`), and even-integer milestone ladder pricing (`milestone_engine.py`).
- `src/evaluation/` — Trailing-mean proxy backtester (`backtest.py`), bi-weekly self-calibration loop (`calibrate.py`), and settled play grading / model record tracker (`record.py`).
- `src/publication/` — End-to-end slate generator & static bundle builder (`static_builder.py`), daily pipeline runner (`daily_refresh.py`), and `gh-pages` worktree publisher (`publisher.py`).
- `web/` — Source vanilla HTML/CSS/JS mobile dashboard (`index.html`, `styles.css`, `app.js`).
- `site/` — Built static artifact bundle (`daily_slate.json` + copied `web/` assets) deployed to GitHub Pages.
- `docs/` — System architecture (`ARCHITECTURE.md`), model specification (`MODEL_SPEC.md`), and operational runbooks (`docs/runbooks/`).
- `tests/` — Offline pytest suite, Node dashboard regression tests (`test_dashboard.cjs`), and JSON fixtures (`tests/fixtures/`).

## Dashboard & Modeling Highlights

- **Three Views:**
  - **Browse:** Full-width cards on mobile (390px–430px) and 2-column grid on desktop (`>=700px`). Supports game-picker scoping, search, smart filters (`Top Value`, `Milestones`, `Overs`, `Unders`, `All`), stat tabs, and sorting (`EV`, `Probability`, `Edge`, `Player`).
  - **Best Picks (`digest`):** 3-column compact mini-card grid of recommended plays (`EV` or `Prob` sort). Excludes `0.5` lines so top recommendations do not hinge on a single counted event (`0.5` lines remain accessible in Browse).
  - **Games:** Lists matchups on the slate with pick and prop counts; clicking a game scopes Browse to that matchup.
- **Matchup Integrity:** Each consensus line stores `home_team_abbr`, `away_team_abbr`, and `player_team_abbr`. Matchups and opponents are derived from the line snapshot rather than mutable player roster records, preventing historical season imports or trades from corrupting today's slate. Rows where `player_team_abbr` matches neither team in the matchup are excluded.
- **Probability & Push Semantics:** Recommendation labels, probability bars, and probability sorting always reflect the *recommended side's* win probability (which can be `<50%` on positive-money long shots with +EV). Integer-line push mass `P(X = L)` is tracked explicitly so pushes are never counted as under wins.
- **Empirical-Bayes Shrinkage & Confidence Gating:** Raw rate projections are shrunk toward the consensus line (`prior_games = 6.0`). Low-confidence plays (`<8` history games, `>30%` L3 vs L20 form/role shift, or inactive this season while team has played) and auto-suppressed markets have recommendations and milestone triggers disabled.
