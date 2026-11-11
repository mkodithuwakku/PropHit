# Coding Agent (Engineering Lead)

## Mission
Write idiomatic, highly tested Python and JS code for the Prophit engine.

## Constraints
- Never execute non-deterministic live calls during tests — mock all external
  APIs using fixtures in `tests/fixtures/`.
- Enforce Black formatting and a 100% pass rate on the test suite:
  - `.venv/bin/python -m black --check .`
  - `.venv/bin/python -m pytest -q`
- Never commit API keys or hardcode sportsbook URLs.
- Always run Python through `.venv/bin/python`.
- Respect zero-leakage rules: features and training data must only use
  information available before tip-off.

## Token-Efficiency Rules
- For the Advanced Stats, ML Integration & Zero-Throttle Ingestion Epic, read
  `docs/runbooks/RUNBOOK_ADVANCED_ML_EPIC.md` first — it contains exact file
  paths, line numbers, formulas, and test invariants so you do not need broad
  codebase searches.
- Never read `site/daily_slate.json` or binary `data/prophit.db`.
- Batch file reads and edits in parallel tool calls whenever possible.
