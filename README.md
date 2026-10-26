# Prophit

Automated NBA player prop & milestone betting engine. Prophit ingests free data
sources (NBA stats, player tracking, consensus odds, injury reports), runs a
multi-stage ML pipeline to project full player performance distributions, and
identifies +EV lines and fair-priced milestone/alt-line ladders. Output is a
static mobile-first dashboard published to GitHub Pages.

See `AGENTS.md` for agent governance rules and `PROPHIT_SPECIFICATION.md` for
the full system requirements document.

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Verify

```bash
.venv/bin/python -m pytest -q          # all tests run 100% offline
.venv/bin/python -m black --check .    # formatting
node --test tests/test_dashboard.cjs  # offline UI regressions; Node 18+
```

## Layout

- `src/ingestion/` — NBA stats, consensus odds, and injury report adapters + SQLite persistence
- `src/features/` — feature store, EWMA windows, vacated usage (M2)
- `src/models/` — minutes/rate/distribution models (M3–M4)
- `src/decision/` — +EV calculator and milestone ladder engine (M4)
- `src/publication/` — static site builder & GitHub Pages publisher (M5–M6)
- `web/` — vanilla HTML/CSS/JS mobile dashboard
- `docs/` — architecture notes and operational runbooks
- `tests/` — offline test suite with mocked fixtures

The Games view uses the home/away matchup stored with each consensus line.
Player teams are also saved with the line, so importing older seasons cannot
change today's matchups. See the daily refresh runbook for refresh and display
verification steps.

Best Picks excludes 0.5 lines to avoid bets that hinge on a single counted
event; those lines remain available in Browse. Recommendation labels,
probability bars, and probability sorting use the recommended side's win
probability, which can differ from the more likely side when odds favor it.
Browse shows full-width cards on phones with wrapping player names, separate
recommendation/EV badges, and labeled lines and projections; wider screens
use two columns.
