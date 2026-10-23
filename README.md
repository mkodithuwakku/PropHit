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
