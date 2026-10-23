# Runbook: Model Retraining

Weekly/monthly model update procedure:

1. Extract past 60 days of actual game logs from SQLite.
2. Re-fit LightGBM hyperparameters on walk-forward validation splits
   (no future data leakage).
3. Evaluate calibration curves (Brier score) and MAE.
4. Save model weights to `models/` directory if Brier score improves.
