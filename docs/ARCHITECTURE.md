# Prophit Architecture

Five-layer pipeline (see `PROPHIT_SPECIFICATION.md` Section 2):

1. **Ingestion** (`src/ingestion/`) — `nba_api` box scores/tracking/hustle,
   consensus odds (The Odds API), official injury report, schedule context.
   Rate-limited, cached into SQLite (`data/prophit.db`).
2. **Feature Store** (`src/features/`) — rolling EWMA windows (3/7/20 games),
   DvP matchup matrix, vacated usage, rest/altitude factors. Strictly
   pre-tip-off data only.
3. **ML Engine** (`src/models/`) — Stage 1 LightGBM minutes regressor →
   Stage 2 stat-specific rate models (Poisson/NB/ZIP) → Stage 3 discrete
   PMF / Monte Carlo distribution engine.
4. **Decision Engine** (`src/decision/`) — consensus-line +EV comparison and
   milestone/alt-line fair-odds ladder pricing.
5. **Publishing** (`src/publication/` + `web/`) — static `daily_slate.json`
   and mobile-first HTML/CSS/JS dashboard → GitHub Pages.
