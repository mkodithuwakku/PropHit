"""Stage 2: stat-specific rate models (Milestone 3).

Per SRD 5.2 each stat gets a model matching its distribution:
  - pts  -> LightGBM regressor on counts conditioned on minutes/features
  - reb  -> Negative Binomial GLM (discrete count, overdispersed)
  - ast  -> Poisson GLM (discrete count)
  - fg3m -> Zero-Inflated Poisson (many zero games)

GLM models use a log(minutes) offset so they learn per-minute rates; the
LightGBM model predicts counts directly with minutes as a feature.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.features.feature_store import FeatureStore
from src.ingestion.db import ProphitDB
from src.models.minutes_model import make_regressor, to_matrix

# Normalized feature keys shared across all stat models.
RATE_FEATURES = [
    "recent_rate",
    "recent_ewma",
    "minutes_l7",
    "usage_volume_l7",
    "is_b2b",
    "days_rest",
]

COUNT_STATS = ("pts", "reb", "ast", "fg3m")

DEFAULT_LGBM_COUNT_PARAMS = {
    "objective": "poisson",
    "n_estimators": 150,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "deterministic": True,
    "seed": 42,
}


def _safe_log_minutes(minutes: np.ndarray) -> np.ndarray:
    return np.log(np.clip(np.asarray(minutes, dtype=float), 1.0, None))


def _design_matrix(
    feature_dicts: list[dict], extra_minutes: Optional[np.ndarray] = None
) -> np.ndarray:
    X = to_matrix(feature_dicts, RATE_FEATURES)
    X = np.where(np.isnan(X), 0.0, X)  # GLMs need finite inputs
    if extra_minutes is not None:
        X = np.column_stack([X, _safe_log_minutes(extra_minutes)])
    import statsmodels.api as sm

    return sm.add_constant(X, has_constant="add")


# --- GLM rate models ---------------------------------------------------------


class _GLMRateModel:
    """Base for offset-adjusted count GLMs (Poisson / NegBinomial)."""

    def __init__(self):
        self._res = None

    def _family(self):  # pragma: no cover - overridden
        raise NotImplementedError

    def fit(self, samples: list[dict]) -> "_GLMRateModel":
        import statsmodels.api as sm

        X = _design_matrix([s["features"] for s in samples])
        y = np.asarray([s["target"] for s in samples], dtype=float)
        offset = _safe_log_minutes(np.asarray([s["minutes"] for s in samples]))
        self._res = sm.GLM(y, X, family=self._family(), offset=offset).fit()
        return self

    def predict_count(self, features: dict, minutes: float) -> float:
        if self._res is None:
            raise RuntimeError("predict_count called before fit")
        X = _design_matrix([features])
        offset = _safe_log_minutes(np.asarray([minutes]))
        return float(self._res.predict(exog=X, offset=offset)[0])

    def predict_rate(self, features: dict, minutes: float) -> float:
        return self.predict_count(features, minutes) / max(minutes, 1.0)


class PoissonRateModel(_GLMRateModel):
    """Poisson GLM for count stats (e.g., assists)."""

    def _family(self):
        import statsmodels.api as sm

        return sm.families.Poisson()


class NegativeBinomialRateModel(_GLMRateModel):
    """Negative Binomial GLM for overdispersed counts (e.g., rebounds)."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def _family(self):
        import statsmodels.api as sm

        return sm.families.NegativeBinomial(alpha=self.alpha)


class ZIPRateModel:
    """Zero-Inflated Poisson for sparse count stats (e.g., 3PM)."""

    def __init__(self):
        self._res = None

    def fit(self, samples: list[dict]) -> "ZIPRateModel":
        import warnings

        from statsmodels.discrete.count_model import ZeroInflatedPoisson

        X = _design_matrix([s["features"] for s in samples])
        y = np.asarray([s["target"] for s in samples], dtype=float)
        offset = _safe_log_minutes(np.asarray([s["minutes"] for s in samples]))
        model = ZeroInflatedPoisson(y, X, offset=offset)
        # hessian inversion only affects SEs, which predictions don't need
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                self._res = model.fit(method="bfgs", maxiter=300, disp=0)
            except Exception:
                self._res = model.fit(maxiter=300, disp=0)
        return self

    def predict_count(self, features: dict, minutes: float) -> float:
        if self._res is None:
            raise RuntimeError("predict_count called before fit")
        X = _design_matrix([features])
        offset = _safe_log_minutes(np.asarray([minutes]))
        pred = self._res.predict(exog=X, offset=offset)
        return float(np.asarray(pred).ravel()[0])

    def predict_rate(self, features: dict, minutes: float) -> float:
        return self.predict_count(features, minutes) / max(minutes, 1.0)


class LightGBMRateModel:
    """Count regressor conditioned on minutes (used for points)."""

    def __init__(self, params: Optional[dict] = None):
        self.params = {**DEFAULT_LGBM_COUNT_PARAMS, **(params or {})}
        self._model = None

    def fit(self, samples: list[dict]) -> "LightGBMRateModel":
        X = to_matrix([s["features"] for s in samples], RATE_FEATURES)
        X = np.column_stack(
            [X, np.asarray([s["minutes"] for s in samples], dtype=float)]
        )
        y = np.asarray([s["target"] for s in samples], dtype=float)
        self._model = make_regressor(self.params)
        self._model.fit(X, y)
        return self

    def predict_count(self, features: dict, minutes: float) -> float:
        if self._model is None:
            raise RuntimeError("predict_count called before fit")
        X = to_matrix([features], RATE_FEATURES)
        X = np.column_stack([X, np.asarray([minutes], dtype=float)])
        return float(max(self._model.predict(X)[0], 0.0))

    def predict_rate(self, features: dict, minutes: float) -> float:
        return self.predict_count(features, minutes) / max(minutes, 1.0)


STAT_MODEL_REGISTRY = {
    "pts": LightGBMRateModel,
    "reb": NegativeBinomialRateModel,
    "ast": PoissonRateModel,
    "fg3m": ZIPRateModel,
}


def build_stat_model(stat: str, **kwargs):
    """Factory: stat_type -> model per SRD 5.2."""
    cls = STAT_MODEL_REGISTRY.get(stat, PoissonRateModel)
    return cls(**kwargs) if cls is NegativeBinomialRateModel else cls()


# --- dataset assembly --------------------------------------------------------


def _rate_features(feats: dict, stat: str) -> dict:
    return {
        "recent_rate": feats.get(f"{stat}_per_min_l20"),
        "recent_ewma": feats.get(f"{stat}_ewma"),
        "minutes_l7": feats.get("minutes_l7"),
        "usage_volume_l7": feats.get("usage_volume_l7"),
        "is_b2b": feats.get("is_b2b"),
        "days_rest": feats.get("days_rest"),
    }


def build_rate_samples(
    db: ProphitDB,
    stat: str,
    player_ids: Optional[list[int]] = None,
    min_prior_games: int = 5,
) -> list[dict]:
    """Assemble (features -> stat count) rows from fact_player_games.

    Each sample's features come only from games strictly before its date.
    """
    store = FeatureStore(db)
    if player_ids is None:
        player_ids = [
            r["player_id"]
            for r in db.query("SELECT DISTINCT player_id FROM fact_player_games")
        ]

    samples = []
    for pid in player_ids:
        games = db.query(
            "SELECT * FROM fact_player_games "
            f"WHERE player_id = ? AND minutes > 0 AND {stat} IS NOT NULL "
            "ORDER BY game_date",
            (pid,),
        )
        for game in games:
            feats = store.rolling_features(pid, game["game_date"])
            if feats["games_available"] < min_prior_games:
                continue
            samples.append(
                {
                    "player_id": pid,
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "minutes": game["minutes"],
                    "features": _rate_features(feats, stat),
                    "target": game[stat],
                }
            )
    return samples


def walk_forward_rate_eval(
    model_factory,
    samples: list[dict],
    min_train: int,
) -> dict:
    """Walk-forward count-MAE vs a naive constant-rate baseline.

    Baseline predicts train-mean rate x test minutes each step.
    """
    ordered = sorted(samples, key=lambda s: (s["game_date"], s.get("game_id", "")))
    model_errs, base_errs = [], []
    for test in ordered:
        train = [s for s in ordered if s["game_date"] < test["game_date"]]
        if len(train) < min_train:
            continue
        rates = [
            s["target"] / s["minutes"]
            for s in train
            if s["minutes"] and s["minutes"] > 0
        ]
        baseline = (sum(rates) / len(rates)) * test["minutes"] if rates else 0.0
        model = model_factory()
        model.fit(train)
        pred = model.predict_count(test["features"], test["minutes"])
        model_errs.append(abs(pred - test["target"]))
        base_errs.append(abs(baseline - test["target"]))

    model_mae = sum(model_errs) / len(model_errs) if model_errs else math.nan
    base_mae = sum(base_errs) / len(base_errs) if base_errs else math.nan
    return {
        "n_test": len(model_errs),
        "model_mae": model_mae,
        "baseline_mae": base_mae,
        "beats_baseline": model_mae < base_mae,
    }
