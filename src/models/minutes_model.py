"""Stage 1: LightGBM minutes predictor (Milestone 3).

Predicts actual minutes played from strictly pre-tip-off features: rotation
trends (rolling means/EWMA), usage, foul tendencies, rest schedule, and
altitude. Loss is Huber (robust to garbage-time outliers) per SRD 5.1.

Walk-forward evaluation compares model MAE against the expanding
season-average baseline — the M3 verification gate.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np

from src.features.feature_store import FeatureStore
from src.ingestion.db import ProphitDB

MINUTES_FEATURES = [
    "minutes_l3",
    "minutes_l7",
    "minutes_l20",
    "minutes_ewma",
    "pts_l7",
    "usage_volume_l7",
    "fga_l7",
    "pf_l7",
    "days_rest",
    "is_b2b",
    "is_3_in_4",
    "games_last_4d",
    "is_home",
    "altitude_game",
]

DEFAULT_LGBM_PARAMS = {
    "objective": "huber",
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 5,
    "deterministic": True,
    "seed": 42,
}


def to_matrix(feature_dicts: list[dict], names: list[str]) -> np.ndarray:
    """(n_samples, n_features) float matrix; missing -> NaN (lgbm-safe)."""
    return np.asarray(
        [[fd.get(name, np.nan) for name in names] for fd in feature_dicts],
        dtype=float,
    )


class _RidgeRegressor:
    """Numpy ridge fallback used only when lightgbm is unavailable."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self._w: Optional[np.ndarray] = None
        self._fill: Optional[np.ndarray] = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self._fill = np.where(np.isnan(X).all(axis=0), 0.0, np.nanmean(X, axis=0))
        Xf = np.where(np.isnan(X), self._fill, X)
        Xb = np.column_stack([np.ones(len(Xf)), Xf])
        reg = self.alpha * np.eye(Xb.shape[1])
        reg[0, 0] = 0.0  # never penalize intercept
        self._w = np.linalg.pinv(Xb.T @ Xb + reg) @ Xb.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        Xf = np.where(np.isnan(X), self._fill, X)
        Xb = np.column_stack([np.ones(len(Xf)), Xf])
        return Xb @ self._w


class _LGBMBoosterRegressor:
    """fit/predict wrapper around lightgbm's native Booster API.

    Uses ``lgbm.train`` directly so no scikit-learn dependency is needed.
    Accepts sklearn-style ``n_estimators`` and converts it to rounds.
    """

    def __init__(self, params: dict):
        params = dict(params)
        self._num_rounds = int(params.pop("n_estimators", 200))
        params["verbosity"] = -1
        self._params = params
        self._booster = None

    def fit(self, X, y):
        import lightgbm as lgbm

        ds = lgbm.Dataset(np.asarray(X), label=np.asarray(y))
        self._booster = lgbm.train(self._params, ds, num_boost_round=self._num_rounds)
        return self

    def predict(self, X):
        return self._booster.predict(np.asarray(X))


def make_regressor(params: Optional[dict] = None):
    """LightGBM regressor; falls back to ridge if lightgbm is missing."""
    merged = {**DEFAULT_LGBM_PARAMS, **(params or {})}
    try:
        import lightgbm  # noqa: F401

        return _LGBMBoosterRegressor(merged)
    except ImportError:
        return _RidgeRegressor()


def mean_baseline(train_samples: list[dict], target_key: str) -> float:
    """Expanding season-average baseline for the WFV gate."""
    vals = [s[target_key] for s in train_samples if s[target_key] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def walk_forward_mae(
    samples: list[dict],
    model_factory: Callable[[], object],
    target_key: str,
    min_train: int,
    baseline_fn: Optional[Callable[[list[dict]], float]] = None,
) -> dict:
    """Expanding-window walk-forward MAE vs a baseline predictor.

    Each step fits ``model_factory()`` on all samples strictly before the
    test sample, predicts once, and compares |error| to the baseline's.
    """
    baseline_fn = baseline_fn or (lambda tr: mean_baseline(tr, target_key))
    ordered = sorted(samples, key=lambda s: (s["game_date"], s.get("game_id", "")))
    model_errs, base_errs = [], []
    for test in ordered:
        # strictly leak-free: train only on games before the test date
        train = [s for s in ordered if s["game_date"] < test["game_date"]]
        if len(train) < min_train or test[target_key] is None:
            continue
        model = model_factory()
        model.fit(train)
        pred = model.predict(test["features"])
        model_errs.append(abs(pred - test[target_key]))
        base_errs.append(abs(baseline_fn(train) - test[target_key]))

    model_mae = sum(model_errs) / len(model_errs) if model_errs else math.nan
    base_mae = sum(base_errs) / len(base_errs) if base_errs else math.nan
    return {
        "n_test": len(model_errs),
        "model_mae": model_mae,
        "baseline_mae": base_mae,
        "mae_improvement_pct": (
            100.0 * (base_mae - model_mae) / base_mae
            if base_mae and not math.isnan(base_mae)
            else math.nan
        ),
        "beats_baseline": model_mae < base_mae,
    }


class MinutesModel:
    """Stage-1 regressor predicting minutes for one upcoming game."""

    FEATURES = MINUTES_FEATURES

    def __init__(self, params: Optional[dict] = None):
        self.params = params or {}
        self._model = None

    def fit(self, samples: list[dict]) -> "MinutesModel":
        X = to_matrix([s["features"] for s in samples], self.FEATURES)
        y = np.asarray([s["minutes"] for s in samples], dtype=float)
        self._model = make_regressor(self.params)
        self._model.fit(X, y)
        return self

    def predict(self, features: dict) -> float:
        if self._model is None:
            raise RuntimeError("MinutesModel.predict called before fit")
        X = to_matrix([features], self.FEATURES)
        return float(self._model.predict(X)[0])

    def predict_many(self, feature_dicts: list[dict]) -> list[float]:
        return [self.predict(fd) for fd in feature_dicts]

    def walk_forward(self, samples: list[dict], min_train: int = 30) -> dict:
        return walk_forward_mae(
            samples,
            model_factory=lambda: MinutesModel(self.params),
            target_key="minutes",
            min_train=min_train,
        )


def build_minutes_samples(
    db: ProphitDB,
    player_ids: Optional[list[int]] = None,
    min_prior_games: int = 5,
) -> list[dict]:
    """Assemble (features -> minutes) training rows from fact_player_games.

    Features for each game use only games strictly before it; game context
    (home/away, altitude) is known pre-tip-off and included.
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
            "WHERE player_id = ? AND minutes IS NOT NULL "
            "ORDER BY game_date",
            (pid,),
        )
        for game in games:
            feats = store.rolling_features(pid, game["game_date"])
            if feats["games_available"] < min_prior_games:
                continue
            ctx = store.game_context(
                game["game_date"],
                opponent_team_id=game["opponent_team_id"],
                is_home=bool(game["is_home"]) if game["is_home"] is not None else None,
            )
            feats["is_home"] = game["is_home"]
            feats["altitude_game"] = ctx["altitude_game"]
            samples.append(
                {
                    "player_id": pid,
                    "game_id": game["game_id"],
                    "game_date": game["game_date"],
                    "features": feats,
                    "minutes": game["minutes"],
                }
            )
    return samples
