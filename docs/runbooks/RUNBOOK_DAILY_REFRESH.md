# Runbook: Daily Refresh

Automated morning pipeline (7:30–8:30 AM local):

1. Pull schedule and official morning injury report.
2. Ingest consensus odds for tonight's games.
3. Compute feature vectors for all active players.
4. Execute ML inference (Minutes → Rates → Probabilities).
5. Generate `docs/daily_slate.json` and static HTML.
6. Commit and push to `gh-pages` branch.
