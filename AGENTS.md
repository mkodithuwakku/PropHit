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
