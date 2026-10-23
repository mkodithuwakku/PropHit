# Model Specification

Multi-stage pipeline (SRD Section 5):

- **Stage 1 — Minutes:** LightGBM regressor, Huber/MAE loss; features from
  rotation trends, starter/bench status, vacated minutes, blowout index,
  foul rates, rest days.
- **Stage 2 — Rates:** per-minute stat models. Points: GBM conditioned on
  projected minutes/TS%/usage/opp defense. Rebounds & assists: NegBinomial
  GLM / Poisson. 3PM: Zero-Inflated Poisson.
- **Stage 3 — Distribution:** discrete PMF P(X = k), k in [0, 60], via
  Negative Binomial (mean + fitted dispersion) and 10,000-trial Monte Carlo
  combining minutes variance with per-minute Poisson rates.
- **Stage 4 — Evaluation:** cumulative P(X >= k) vs consensus prices → +EV;
  milestone ladder fair odds when projection >= line + 2.0 (SRD 6.2).
