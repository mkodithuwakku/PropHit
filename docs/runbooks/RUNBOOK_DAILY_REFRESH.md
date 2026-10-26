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
- Best Picks excludes every 0.5 line; Browse retains them. Check both EV and
  probability sorting, including a slate with no eligible Best Picks.
- Confirm each recommendation's side, EV, displayed win probability, and bar
  agree with that side in the JSON. A positive-EV over may have a lower win
  probability than the under; never substitute the larger probability.
  Under probabilities come from `p_under`, so integer-line pushes are not
  counted as under wins.
- At 390px, Browse should have one card per row, fully visible player names
  and matchups, and labeled line/projection values. At 700px and wider, it
  uses two columns. Check Factors and milestone ladders still expand.

For web-only changes, synchronize `web/index.html`, `web/styles.css`, and
`web/app.js` into `site/` without regenerating the slate or its timestamp.
Pages deploys the `site/` bundle; the daily builder copies these same assets
from `web/` on subsequent refreshes. Bump the asset query versions in
`web/index.html` when changing the UI so cached clients load the new bundle.
