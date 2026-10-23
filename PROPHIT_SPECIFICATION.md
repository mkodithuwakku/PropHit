# Software Requirements Document (SRD): Prophit (NBA Player Prop & Milestone Engine)

---

## 1. Executive Summary & Vision

### 1.1 Project Objective
**Prophit** is an automated, machine-learning-driven NBA player proposition and milestone betting engine. The system ingests all available free data sources (official NBA statistics, player tracking, defensive metrics, schedule/rest factors, and morning consensus lines), runs a multi-stage machine learning pipeline to project player performance distributions, identifies high-discrepancy lines against bookmaker consensus, and prices **alternative/milestone lines** (e.g., taking 12+ rebounds when the standard line is 9.5).

The platform builds a responsive, mobile-first static dashboard deployed daily via **GitHub Pages** (mirroring LockIn's zero-maintenance architecture), allowing morning decision-making directly from a mobile device without server hosting costs.

### 1.2 Core Philosophy
- **Morning Workflow:** Designed for early-morning betting runs (8:00 AM – 10:00 AM local time). Avoids fragile real-time pre-tip-off scraping.
- **Distributional Modeling (Not Point Estimates):** Models calculate the entire discrete probability distribution $P(X \ge k)$, allowing exact fair-value pricing for standard and milestone/ladder lines.
- **Zero-Maintenance Hosting:** A single automated pipeline generates static JSON and HTML, pushed to GitHub Pages. Zero servers, zero database costs.
- **Strict Agentic Architecture:** Engineered from Day 1 to be built, tested, and maintained by autonomous agent personas (Coding Agent, Documentation/Prompt Agent).

---

## 2. System Architecture & Data Flow

```mermaid
flowchart TD
    subgraph Data_Acquisition[1. Free Ingestion Layer (Morning Run)]
        NBA[nba_api: Box Scores, Advanced, Tracking, Hustle]
        Odds[PrizePicks Public Projections + Underdog Fallback]
        Injuries[Official NBA Injury Report PDF/JSON]
        Schedule[NBA B2B / Rest / Altitude / Schedule]
    end

    subgraph Storage[2. Persistence & Feature Store]
        DB[(SQLite: Historical Games, Odds, Tracking)]
        FeatEng[Feature Store: Rolling Windows, Matchups, Vacated Usage]
    end

    subgraph MLEngine[3. Multi-Stage ML Pipeline]
        M_Min[Stage 1: Minutes Predictor (LightGBM)]
        M_Rates[Stage 2: Per-Minute Rate Models (GLM / Poisson / XGB)]
        M_Dist[Stage 3: Discrete Probability Engine (Negative Binomial / Monte Carlo)]
    end

    subgraph DecisionEngine[4. Prop & Milestone Evaluator]
        Comp[Consensus Line Comparison (+EV Calculation)]
        Milestone[Milestone / Ladder Extrapolator (P(X >= k) vs Fair Odds)]
        Rank[Confidence & Edge Ranker]
    end

    subgraph Publishing[5. Deployment & UI]
        Export[Static Site Generator (JSON + Responsive HTML/CSS/JS)]
        GH[GitHub Actions -> GitHub Pages]
        Mobile[Mobile Dashboard: Best Bets, Milestone Ladders]
    end

    Data_Acquisition --> Storage
    Storage --> FeatEng
    FeatEng --> MLEngine
    MLEngine --> DecisionEngine
    DecisionEngine --> Export
    Export --> GH
    GH --> Mobile
```

---

## 3. Data Acquisition & Consensus Odds Specifications

### 3.1 Data Sources (100% Free Tier)
1. **NBA Stats via `nba_api` & CDN:**
   - **Player Game Logs:** Box scores from past 3 full seasons + current season.
   - **Player Tracking:** Potential assists, rebound chances, touches, front-court touches, average time of possession per touch.
   - **Hustle Stats:** Deflections, screen assists, contested shots.
   - **Defense vs. Position (DvP):** Team defensive rating, paint points conceded, perimeter 3P% allowed, rim protection FG% allowed.
2. **Schedule & Contextual Data:**
   - Days of rest (B2B, 3-in-4, travel distance, altitude games e.g., DEN/UTA).
   - Game totals & point spreads (Vegas game totals indicate pace and blowout risk).
3. **Consensus Line Aggregator (100% free, zero-auth):**
   - Primary Source: **PrizePicks Public REST API** (`https://api.prizepicks.com/projections?league_id=7` for NBA). Returns the entire evening slate of player props in a single morning JSON payload.
   - Secondary Fallback: **Underdog Fantasy** public over/under lines, used automatically if the primary pull fails.
   - Auth Required: **NONE** — 100% free, unauthenticated, zero secrets or tokens.
   - Target markets: Player Points, Rebounds, Assists, 3-Pointers Made (3PM), and Combos (Pts+Rebs+Asts).
4. **Official Injury Feed:**
   - Official NBA 1:30 PM / morning injury reports.
   - Logic: Identify confirmed OUT, Questionable, and Probable players to compute **Vacated Usage** and **Vacated Minutes**.

### 3.2 Ingestion Schedule
- **Frequency:** Once daily between 7:30 AM and 8:30 AM local time.
- **Rate-Limiting Protection:** Bounded retries, exponential backoff (0.5s–2s between NBA Stats calls), cached responses in local SQLite to eliminate redundant network traffic.

---

## 4. Feature Engineering & Persistence Schema

### 4.1 SQLite Database Schema
The database (`data/prophit.db`) must maintain:

1. `dim_players` & `dim_teams`: IDs, names, positions, physical profiles.
2. `fact_player_games`: Historic box scores, minutes, tracking metrics (past 3 seasons).
3. `fact_team_defense_daily`: Opponent defensive rankings sliced by position and stat type.
4. `fact_consensus_lines`: Historical opening lines, closing lines, and morning consensus lines.
5. `fact_model_evaluations`: Pre-game model outputs saved before game start to verify calibration without data leakage.

### 4.2 Engineered Features
- **Usage Vacuum:** Teammate absent usage percentage redistributed based on historical on/off splits.
- **Rolling Windows:** Weighted EWMA (Exponential Weighted Moving Average) across 3-game, 7-game, and 20-game horizons.
- **Matchup Vulnerability Score:** DvP percentile ranking against player's primary shot profile (e.g., paint-heavy center vs bottom-5 rim defense).
- **Blowout Index:** $|Spread| \times (Total / 220)$. High values trigger down-weighting of 4th-quarter starter minutes.
- **Rest Penalty:** Flags for back-to-backs with travel or high altitude.

---

## 5. Machine Learning Engine Architecture

### 5.1 Multi-Stage Pipeline Design

```
Raw Inputs ──► [Stage 1: Minutes Model] ──► Predicted Min
                      │
                      ▼
               [Stage 2: Rate Models]   ──► Predicted Rates (PTS/min, REB/min, etc.)
                      │
                      ▼
               [Stage 3: Distribution]  ──► Full Probability Mass Function P(X = k)
                      │
                      ▼
               [Stage 4: Milestone EV]  ──► Cumulative P(X >= k) vs Sportsbook Odds
```

#### Stage 1: Minutes Played Model (LightGBM Regressor)
- **Target:** Actual minutes played.
- **Features:** Coach rotation trends (last 5 games), starter/bench status, injury vacated minutes, blowout spread index, foul rate tendencies, rest days.
- **Loss Function:** Huber loss or MAE (robust to garbage-time outliers).

#### Stage 2: Category Rate Models (Stat-Specific)
Separate models tailored to the mathematical distribution of each stat:
1. **Points (Continuous/High Variance):** LightGBM/CatBoost conditioned on projected minutes, true shooting %, usage rate, and opponent defensive efficiency.
2. **Rebounds & Assists (Discrete Count Data):** Negative Binomial GLM / Poisson Regressors modeling expected rate per possession/minute.
3. **Three-Pointers Made (3PM):** Zero-Inflated Poisson (ZIP) modeling catch-and-shoot volume and corner-three frequency allowed.

#### Stage 3: Probability Distribution & Monte Carlo Engine
Rather than simply generating a single projected mean ($\hat{\mu}$), the engine computes the full probability mass function $P(X = k)$ for integer outcomes $k \in [0, 60]$ using:
- **Parametric Formulation:** Negative Binomial distribution parameterized by mean $\mu$ and dispersion parameter $r$ (fit on player's historical consistency).
- **Monte Carlo Simulation (10,000 trials):** Simulating minutes variance $\sim \mathcal{N}(\mu_{min}, \sigma_{min})$ combined with per-minute Poisson rates to capture tail volatility.

---

## 6. Betting Decision, +EV, & Milestone Ladder Engine

### 6.1 Standard Line Comparison (+EV)
For any consensus line $L$ (e.g., Rebounds Over 9.5 at -115):
1. **Calculate Model Win Probability:**
   $$P(\text{Over } L) = \sum_{k = \lceil L \rceil}^{\infty} P(X = k)$$
2. **Calculate Implied Probability from Consensus Odds:**
   $$\text{Implied Prob} = \frac{|\text{American Odds}|}{|\text{American Odds}| + 100} \quad (\text{if negative})$$
3. **Calculate Expected Value (EV):**
   $$\text{EV} = (P_{model} \times \text{Net Payout}) - ((1 - P_{model}) \times \text{Stake})$$
   Flag as a **Recommended Play** if $\text{EV} \ge +5.0\%$ and sample edge meets minimum threshold.

### 6.2 Milestone / Alt-Line Ladder Pricing Engine
Bet365 and major sportsbooks offer alternative lines (e.g., 8+, 10+, 12+, 14+). Often, consensus odds providers only provide the base line (9.5).

**The Engine's Milestone Pricing Formula:**
1. **Trigger:** Whenever Model Projection $\ge \text{Consensus Line} + 2.0$ (e.g., Consensus is 9.5, Model projects 12.2).
2. **Compute Milestone Probabilities:**
   - Calculate $P(X \ge 10)$, $P(X \ge 12)$, $P(X \ge 14)$ directly from the distribution.
3. **Derive Fair American Odds:**
   $$\text{Fair Decimal} = \frac{1}{P(X \ge k)}$$
   $$\text{Fair American} = (\text{Fair Decimal} - 1) \times 100 \quad (\text{if } P \le 0.50)$$
4. **UI Output for the User:**
   The dashboard displays a dedicated **Milestone Alert**:
   > *"Domantas Sabonis vs UTA: Consensus is 12.5. Model predicts 15.4.*
   > - *Over 12.5 Fair: -190 (Model: 65.5%)*
   > - *Milestone 14+ Fair: +125 (Model: 44.4%) — Bet365 Alt Target: +150 or better*
   > - *Milestone 16+ Fair: +260 (Model: 27.8%) — Bet365 Alt Target: +300 or better"*

This empowers the user to immediately recognize profitable alt-lines on Bet365 that are not listed on consensus odds feeds.

---

## 7. Frontend Specification & Mobile-First GitHub Pages

### 7.1 Architecture & Design System
- **Hosting:** GitHub Pages (Static export of HTML, CSS, JavaScript, and pre-computed `daily_slate.json`).
- **Styling:** Vanilla CSS, zero bundle tools (no Webpack, Vite, or React needed).
- **Theme (LockIn Style Guide):**
  - Charcoal base (`#121316`), subtle plum/violet undertones (`#1c1a24`), muted accents.
  - High-contrast text, clear visual chips for Edge % (+EV green, neutral amber, caution red).
- **Responsive Layout:**
  - Viewport-optimized for iOS Safari and Android Chrome (390px–430px primary layout).
  - Sticky header showing today's date, slate summary, and total plays.
  - Quick-filter tabs: **Top Value (+EV)**, **Milestone Ladders**, **Points**, **Rebounds**, **Assists**, **All Plays**.

### 7.2 UI Player Card Components
Each card must display:
- Player name, position, team, opponent, and home/away indicator.
- Consensus line vs Model prediction (e.g., `Line: 7.5 AST | Model: 10.1 AST`).
- Edge indicator: $+8.4\%$ EV / Model Win Prob: $63.2\%$.
- Factor breakdown expandable tray:
  - Projected minutes breakdown.
  - DvP rank (e.g., "Opponent 28th vs Playmakers").
  - Rest status ("3 days rest").
- **Milestone Ladder Expandable Box** (if discrepancy warrants alt-betting).

---

## 8. Multi-Agent Team Framework & Operational Runbooks

To allow autonomous agent development and maintenance, this project establishes a strict **Agent Governance System**.

### 8.1 Agent Roles & Personas

```
┌────────────────────────────────────────────────────────┐
│                   AGENT TEAM STRUCTURE                 │
├──────────────────────────┬─────────────────────────────┤
│ 1. Coding Agent          │ Implements Python backend,   │
│    (Engineering Lead)    │ ML models, SQLite pipeline, │
│                          │ tests, static site builder. │
├──────────────────────────┼─────────────────────────────┤
│ 2. Documentation &       │ Maintains README, runbooks, │
│    Prompt Agent          │ AGENTS.md, architectural    │
│    (Architecture/Review) │ maps, and agent prompts.    │
└──────────────────────────┴─────────────────────────────┘
```

#### Role 1: Coding Agent (`.devin/agents/coding_agent.md`)
- **Mission:** Write idiomatic, highly tested Python and JS code.
- **Constraints:**
  - Never execute non-deterministic live calls during tests (mock all external APIs).
  - Enforce Black formatting and 100% pass rate on test suites.
  - Never commit API keys or hardcode sportsbook URLs.

#### Role 2: Documentation & Prompt Agent (`.devin/agents/doc_prompt_agent.md`)
- **Mission:** Keep all Markdown documentation, agent rule files, and runbooks synchronized with code changes.
- **Constraints:**
  - Update `README.md` and architecture diagrams on any feature addition.
  - Validate that prompt instructions are unambiguous and fit agent context windows.

---

### 8.2 Standard Repository Runbooks

The project will contain standard operation runbooks inside `docs/runbooks/`:

#### `RUNBOOK_DAILY_REFRESH.md`
Step-by-step procedure for the automated morning pipeline:
1. Pull schedule and official morning injury report.
2. Ingest consensus odds for tonight's games.
3. Compute feature vectors for all active players.
4. Execute ML inference (Minutes $\to$ Rates $\to$ Probabilities).
5. Generate `docs/daily_slate.json` and static HTML.
6. Commit and push to `gh-pages` branch.

#### `RUNBOOK_MODEL_RETRAINING.md`
Procedure for weekly/monthly model updates:
1. Extract past 60 days of actual game logs from SQLite.
2. Re-fit LightGBM hyperparameters on walk-forward validation splits (no future data leakage).
3. Evaluate calibration curves (Brier score) and MAE.
4. Save model weights to `models/` directory if Brier score improves.

#### `RUNBOOK_BACKTEST_EVALUATION.md`
Evaluation protocol:
1. Simulate past 1,000 bets with historic opening/closing lines.
2. Calculate ROI at flat 1-unit staking and Kelly-fraction staking.
3. Calculate CLV (Closing Line Value) hit rate.

---

## 9. File Tree & Workspace Structure

```
Prophit/
├── .devin/
│   ├── agents/
│   │   ├── coding_agent.md
│   │   └── doc_prompt_agent.md
│   └── skills/
├── AGENTS.md                          # Universal Agent Instructions & Rules
├── README.md                          # Living Portfolio & System Documentation
├── requirements.txt                   # Production dependencies
├── requirements-dev.txt               # Testing & linting tools
├── config.yaml                        # Thresholds (+EV min, odds limits, API keys)
├── data/
│   └── prophit.db                     # Local SQLite database
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── nba_stats.py               # nba_api wrappers with rate-limiting
│   │   ├── consensus_odds.py          # PrizePicks + Underdog public lines client
│   │   └── injuries.py                # NBA injury report parser
│   ├── features/
│   │   ├── feature_store.py           # Feature transformations & EWMA
│   │   └── vacated_usage.py           # Teammate absence redistribution
│   ├── models/
│   │   ├── minutes_model.py           # Stage 1: LightGBM minutes predictor
│   │   ├── stat_models.py             # Stage 2: PTS, REB, AST, 3PM models
│   │   └── distribution.py            # Stage 3: NegBinomial & Monte Carlo engine
│   ├── decision/
│   │   ├── ev_calculator.py           # Expected value & probability margins
│   │   └── milestone_engine.py        # Ladder & alt-line fair odds pricing
│   └── publication/
│       ├── static_builder.py          # Generates daily_slate.json & web bundle
│       └── publisher.py               # Git commit & push to GitHub Pages
├── web/
│   ├── index.html                     # Responsive mobile-first dashboard
│   ├── styles.css                     # Dark-mode charcoal/plum design system
│   └── app.js                         # Dynamic filtering, sorting, milestone view
├── docs/
│   ├── ARCHITECTURE.md
│   ├── MODEL_SPEC.md
│   └── runbooks/
│       ├── RUNBOOK_DAILY_REFRESH.md
│       ├── RUNBOOK_MODEL_RETRAINING.md
│       └── RUNBOOK_BACKTEST_EVALUATION.md
└── tests/
    ├── test_ingestion.py
    ├── test_features.py
    ├── test_models.py
    ├── test_milestones.py
    └── test_static_builder.py
```

---

## 10. Implementation Plan & Milestones

| Milestone | Deliverable | Verification Gate |
| :--- | :--- | :--- |
| **M1: Foundation & Data Ingestion** | `AGENTS.md`, SQLite schema, `nba_stats.py`, consensus odds ingestion. | Automated tests verifying mocked data ingestion with zero network calls. |
| **M2: Feature Store & Vacated Usage** | Rolling features, DvP matrix, schedule/rest factors, injury redistribution. | Feature generation unit tests & leak-free time-series splitting. |
| **M3: ML Engine (Stage 1 & 2)** | LightGBM Minutes model, Poisson/NegBinomial stat models. | Walk-forward cross-validation showing lower MAE than baseline season averages. |
| **M4: Distribution & Milestone Engine** | Discrete PMF generator, +EV thresholding, milestone ladder calculator. | Test cases verifying fair odds pricing across known statistical distributions. |
| **M5: Mobile Dashboard & Static Export** | Responsive HTML/CSS/JS interface, static site generator. | Node/browser tests verifying responsive layout at 390px viewport width. |
| **M6: Automation & GitHub Pages** | Scheduled morning runner, Git publishing workflow, runbook suite. | Successful end-to-end dry run creating static build artifacts. |

---

## 11. Initial `AGENTS.md` Specification to Bootstrap the Project

```markdown
# AGENTS.md — Universal Agent Instructions for Prophit

## 1. Project Mission
Prophit is an automated NBA player prop and milestone betting engine. It generates morning probability distributions for player stats, identifies +EV lines against consensus bookmakers, and calculates fair pricing for alternative/milestone lines (e.g. Bet365 ladders). The output is published daily as a static dashboard on GitHub Pages.

## 2. Core Architectural Principles
- **Offline & Mocked Verification:** All automated tests must run completely offline without hitting live NBA APIs or odds endpoints. Use fixtures in `tests/fixtures/`.
- **Zero-Leakage Time Travel:** Training and feature engineering must strictly use data known *before* game tip-off time. Never use same-day game results in feature calculations.
- **Distributional Focus:** Never evaluate props using only expected value point estimates ($\hat{\mu}$). Always use the discrete probability mass function $P(X \ge k)$.
- **Mobile-First Roster UI:** The web dashboard must remain lightweight, fast-loading, and optimized for phone screens (390px width). No heavy JavaScript frameworks (React/Vue/Webpack); use clean vanilla JS/CSS.

## 3. Tool & Runtime Rules
- Always run Python commands through `.venv/bin/python`.
- Formatting standard: Black (`.venv/bin/python -m black --check .`).
- Test standard: Pytest (`.venv/bin/python -m pytest -q`).
- Documentation: Keep `README.md` and `docs/runbooks/` strictly synchronized with code changes.
```
