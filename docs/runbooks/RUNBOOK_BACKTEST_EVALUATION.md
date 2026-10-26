# Runbook: Backtest & Model Record Evaluation

Prophit provides two complementary offline evaluation mechanisms in `src/evaluation/`:
1. **Trailing-Mean Proxy Backtest (`src/evaluation/backtest.py`):** Evaluates the model across all historical player-games in `fact_player_games` against synthesized book lines (pre-game L20 mean rounded to `0.5` at flat `-110/-110`).
2. **Settled Consensus Record (`src/evaluation/record.py`):** Reconstructs morning projections for every past date that has both stored `fact_consensus_lines` and completed box scores in `fact_player_games`, grading actual unit profit and ROI at the offered bookmaker prices.

---

## 1. Run the Trailing-Mean Proxy Backtest

Because sportsbooks do not publish free historical prop archives, `backtest()` grades the model against each player's pre-game L20 average rounded to the nearest `0.5` at `-110` on both sides (filtering to appearances with `minutes >= 10.0` and `min_history >= 10`).

```bash
.venv/bin/python -c "
import json
from src.ingestion.db import ProphitDB
from src.evaluation.backtest import backtest

db = ProphitDB('data/prophit.db')
report = backtest(db, stats=('pts', 'reb', 'ast', 'fg3m'), min_ev_pct=5.0)
print(json.dumps(report, indent=2))
db.close()
"
```

### Interpreting Backtest Output
- `breakeven_pct`: `52.4%` (required win rate at `-110` vig).
- `overall` and `per_stat` (`pts`, `reb`, `ast`, `fg3m`):
  - `bets`, `wins`, `losses`, `pushes`: Graded counts for plays meeting `EV >= min_ev_pct`. Integer lines that land on the exact number are graded as `pushes` (`0.0u`), never wins.
  - `win_pct`: Win percentage across decided bets (`wins / (wins + losses)`).
  - `units`: Net units won/lost at flat 1u staking (`+0.9091u` per win, `-1.0u` per loss, `0.0u` per push).
  - `roi_pct`: Return on investment (`100 * units / bets`).
  - `favored_n`, `favored_hit_pct`: Directional accuracy across *all* evaluated lines (taking whichever side has $P \ge 0.50$).

---

## 2. Run Settled Consensus Line Evaluation

To inspect the real-world record on dates where live consensus lines were ingested before tip-off and box scores were subsequently ingested:

```bash
.venv/bin/python -c "
import json
from src.ingestion.db import ProphitDB
from src.evaluation.record import summarize_model_stats

db = ProphitDB('data/prophit.db')
stats = summarize_model_stats(db, min_ev_pct=5.0)
print(json.dumps(stats, indent=2))
db.close()
"
```

This function also writes audit rows into `fact_model_evaluations` (`model_version = "nb-mc-v1"`) and powers the `MODEL RECORD` banner in the dashboard header.

---

## 3. Run Offline Evaluation Tests

```bash
.venv/bin/python -m pytest tests/test_backtest.py tests/test_evaluation.py tests/test_calibration.py -q
```
