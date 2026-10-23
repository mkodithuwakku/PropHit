# Runbook: Daily Refresh

The Daily Morning Refresh workflow is scheduled for 12:30 UTC and also
supports manual runs from Actions → Daily Morning Refresh → Run workflow.

For a local live refresh:

```bash
.venv/bin/python -m src.publication.daily_refresh --db data/prophit.db --out site --live
```

The runner pulls public consensus lines, player game logs, and defensive
snapshots; checks whether calibration is due; generates projections and
model-record statistics; then writes `site/daily_slate.json` and the web assets.
The workflow commits `site/` and `data/prophit.db` to `main`.
Pages is deployed by the separate Deploy GitHub Pages workflow. Confirm its
success rather than assuming that committing refreshed data deployed it.

Matchup verification:

- Compare the published slate date and home/away pairs against today's NBA
  schedule. Games are grouped using the line's home/away fields, never a
  player's mutable roster record.
- `player_team_abbr` is persisted with each line. Existing databases gain this
  column automatically; a live pull populates it on new snapshots.
  Rows whose player team belongs to neither team in the matchup are excluded
  from publication rather than assigned a guessed opponent.
- Check the rendered page, not just the JSON: `[hidden]` must remain hidden
  even on components with explicit CSS `display` rules. A false demo badge
  previously appeared because `.demo-badge` overrode the hidden attribute.
- Verify Games, selection/clear in Browse, and Best Picks in a real browser.
  The JSON alone cannot establish that the page displays correctly.
