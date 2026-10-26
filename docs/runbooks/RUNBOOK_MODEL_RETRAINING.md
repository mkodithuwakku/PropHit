# Runbook: Model Retraining & Self-Calibration

Prophit maintains model quality through two mechanisms:
1. **Automated Bi-Weekly Self-Calibration (`src/evaluation/calibrate.py`):** Runs automatically inside `src/publication/daily_refresh.py` every 14 days to tune the EV threshold (`min_ev_pct`) and suppress underperforming stat markets (`suppressed_markets`) in SQLite (`model_settings` table).
2. **Walk-Forward Model Validation (`src/models/`):** Evaluates Stage 1 (`MinutesModel`) and Stage 2 (`stat_models.py`) regressors on expanding leak-free time-series splits against naive baselines.

---

## 1. Automated Bi-Weekly Self-Calibration

Every morning refresh calls `run_calibration(db)`. If `model_settings.last_calibration` is missing or at least `14` days old (`INTERVAL_DAYS = 14`), the calibrator:
1. Runs `backtest(db, min_ev_pct=0.0, collect=True)` to collect graded historical plays.
2. **EV Gate Sweep:** Evaluates `EV_LADDER = (5.0, 8.0, 10.0, 12.0, 15.0)` across plays with at least `MIN_BETS_FOR_TUNING = 100` bets. If the best gate improves ROI by more than `+1.0%`, `min_ev_pct` moves toward it, clamped to at most `±2.0%` (`MAX_GATE_STEP`) per run.
3. **Market Suppression:** For each stat market with at least `MIN_BETS_FOR_SUPPRESSION = 50` bets at the active gate:
   - Suppresses the market if graded ROI is `<= -5.0%` (`SUPPRESS_ROI_PCT`).
   - Unsuppresses a previously suppressed market once graded ROI recovers to `>= +3.0%` (`UNSUPPRESS_ROI_PCT`).
4. Persists `min_ev_pct`, `suppressed_markets`, `last_calibration`, and `calibration_notes` in `model_settings`.

### Force a Manual Calibration Run

```bash
.venv/bin/python -c "
import json
from src.ingestion.db import ProphitDB
from src.evaluation.calibrate import run_calibration

db = ProphitDB('data/prophit.db')
report = run_calibration(db, interval_days=0)
print(json.dumps(report, indent=2))
db.close()
"
```

### Inspect or Reset Calibration Settings

```bash
.venv/bin/python -c "
from src.ingestion.db import ProphitDB

db = ProphitDB('data/prophit.db')
for row in db.query('SELECT key, value, updated_at, reason FROM model_settings'):
    print(dict(row))
db.close()
"
```

---

## 2. Walk-Forward Validation of Stage 1 & Stage 2 Models

Before changing feature definitions or hyperparameters in `src/models/minutes_model.py` or `src/models/stat_models.py`, verify that expanding walk-forward MAE beats the season-average / naive per-minute baseline with zero future leakage:

```bash
.venv/bin/python -c "
from src.ingestion.db import ProphitDB
from src.models.minutes_model import MinutesModel, build_minutes_samples
from src.models.stat_models import build_rate_samples, build_stat_model, walk_forward_rate_eval

db = ProphitDB('data/prophit.db')

min_samples = build_minutes_samples(db, min_prior_games=5)
if min_samples:
    print('Minutes WFV:', MinutesModel().walk_forward(min_samples, min_train=30))

for stat in ('pts', 'reb', 'ast', 'fg3m'):
    samples = build_rate_samples(db, stat, min_prior_games=5)
    if samples:
        res = walk_forward_rate_eval(lambda s=stat: build_stat_model(s), samples, min_train=40)
        print(f'{stat.upper()} WFV:', res)

db.close()
"
```

### Verification Gate
Run the offline model and calibration test suites before committing any model updates:

```bash
.venv/bin/python -m pytest tests/test_models.py tests/test_calibration.py -q
```
