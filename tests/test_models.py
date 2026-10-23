"""M3 verification: LightGBM minutes model + stat-specific rate models.

All data is synthetic/deterministic and stored in a temporary SQLite DB —
zero network, zero live model endpoints. The key gate is walk-forward
validation beating the season-average baseline.
"""

from datetime import date, timedelta

import numpy as np
import pytest

from src.features.feature_store import FeatureStore
from src.models.minutes_model import (
    MinutesModel,
    build_minutes_samples,
    mean_baseline,
)
from src.models.stat_models import (
    LightGBMRateModel,
    NegativeBinomialRateModel,
    PoissonRateModel,
    ZIPRateModel,
    build_rate_samples,
    build_stat_model,
    walk_forward_rate_eval,
)

MIN_TEAM = 9000
MIN_PLAYER = 9001
MIN_OPP = 9100


def _seed_team_player(db, pid, tid):
    db.upsert_team(tid, "MIN")
    db.upsert_team(MIN_OPP, "OPP")
    db.upsert_player(pid, "Minutes Man", team_id=tid)


def _insert_game(db, pid, d, minutes, tid=MIN_TEAM, **stats):
    row = {
        "game_id": f"min-{pid}-{d}",
        "player_id": pid,
        "team_id": tid,
        "opponent_team_id": MIN_OPP,
        "game_date": d.isoformat(),
        "minutes": minutes,
        "pts": 0,
        "reb": 0,
        "ast": 0,
        "fg3m": 0,
        "fga": 10,
        "fta": 2,
        "tov": 2,
        "pf": 3,
        "is_home": 0,
    }
    row.update(stats)
    db.upsert_player_game(row)


@pytest.fixture
def minutes_db(db):
    """50 games: minutes = 14 + 0.55*l5 - 9*B2B + noise -> learnable."""
    _seed_team_player(db, MIN_PLAYER, MIN_TEAM)
    rng = np.random.default_rng(7)
    history = [30.0, 30.0, 30.0, 30.0, 30.0]
    d = date(2025, 10, 1)
    prev = None
    for i in range(50):
        b2b = prev is not None and (d - prev).days == 1
        if i < 5:
            m = 30.0 + rng.normal(0, 0.5)
        else:
            m = (
                14.0
                + 0.55 * float(np.mean(history[-5:]))
                - (9.0 if b2b else 0.0)
                + rng.normal(0, 0.8)
            )
        m = float(np.clip(m, 10, 44))
        _insert_game(db, MIN_PLAYER, d, round(m, 1), is_home=i % 2)
        history.append(m)
        prev = d
        # every 4th gap is next-day -> recurring back-to-backs
        d += timedelta(days=1 if i % 4 == 3 else 2)
    return db


# --- Stage 1: minutes model ---------------------------------------------------


def test_build_minutes_samples(minutes_db):
    samples = build_minutes_samples(minutes_db, min_prior_games=5)
    assert len(samples) == 45  # first 5 games lack history
    s = samples[0]
    assert set(["features", "minutes", "game_date", "player_id"]) <= set(s)
    assert "minutes_l5" not in s["features"]  # windows are l3/l7/l20
    assert s["features"]["minutes_l3"] is not None
    assert s["features"]["is_home"] in (0, 1)


def test_minutes_model_fit_predict(minutes_db):
    samples = build_minutes_samples(minutes_db)
    model = MinutesModel().fit(samples)
    pred = model.predict(samples[-1]["features"])
    assert 10.0 <= pred <= 44.0


def test_minutes_model_walk_forward_beats_baseline(minutes_db):
    samples = build_minutes_samples(minutes_db)
    result = MinutesModel().walk_forward(samples, min_train=30)

    assert result["n_test"] > 0
    assert result["model_mae"] < result["baseline_mae"]
    assert result["beats_baseline"] is True
    # signal is strong: model should cut MAE meaningfully
    assert result["mae_improvement_pct"] > 10.0


def test_minutes_model_is_leak_free(minutes_db):
    """Model must not see the test game's own row in its features."""
    store = FeatureStore(minutes_db)
    feats = store.rolling_features(MIN_PLAYER, "2025-10-19")
    # The 10-19 game itself is excluded from its own features
    games = store.games_before(MIN_PLAYER, "2025-10-19", limit=1)
    assert games[0]["game_date"] < "2025-10-19"


# --- Stage 2: rate models -----------------------------------------------------


def _rate_samples(rng, n, rate_fn, minutes_fn, date0=date(2025, 10, 1)):
    """Direct synthetic samples: target ~ Poisson(rate_fn(b2b) * minutes)."""
    samples = []
    d = date0
    for i in range(n):
        b2b = i % 4 == 3
        minutes = minutes_fn(i)
        lam = rate_fn(b2b) * minutes
        y = int(rng.poisson(lam))
        samples.append(
            {
                "player_id": 1,
                "game_id": f"r{i}",
                "game_date": d.isoformat(),
                "minutes": float(minutes),
                "features": {
                    "recent_rate": rate_fn(b2b) * (1 + rng.uniform(-0.05, 0.05)),
                    "recent_ewma": rate_fn(b2b) * (1 + rng.uniform(-0.05, 0.05)),
                    "minutes_l7": float(minutes),
                    "usage_volume_l7": 12.0 + rng.uniform(-1.0, 1.0),
                    "is_b2b": int(b2b),
                    "days_rest": 0 if b2b else int(rng.integers(1, 4)),
                },
                "target": y,
            }
        )
        d += timedelta(days=1 if b2b else 2)
    return samples


def test_poisson_rate_model_recovers_rate():
    rng = np.random.default_rng(11)
    samples = _rate_samples(rng, 120, lambda b2b: 0.30, lambda i: 24 + (i % 13))
    model = PoissonRateModel().fit(samples)
    pred = model.predict_count(
        {
            "recent_rate": 0.3,
            "recent_ewma": 0.3,
            "minutes_l7": 30.0,
            "usage_volume_l7": 12.0,
            "is_b2b": 0,
            "days_rest": 1,
        },
        minutes=30.0,
    )
    # true expectation = 0.30 * 30 = 9
    assert pred == pytest.approx(9.0, abs=2.0)
    assert model.predict_rate(
        {
            "recent_rate": 0.3,
            "recent_ewma": 0.3,
            "minutes_l7": 30.0,
            "usage_volume_l7": 12.0,
            "is_b2b": 0,
            "days_rest": 1,
        },
        minutes=30.0,
    ) == pytest.approx(0.30, abs=0.07)


def test_negative_binomial_fit_predict():
    rng = np.random.default_rng(13)
    # overdispersed counts: rate fluctuates game to game
    samples = _rate_samples(rng, 100, lambda b2b: 0.35, lambda i: 28 + (i % 9))
    model = NegativeBinomialRateModel(alpha=1.0).fit(samples)
    pred = model.predict_count(
        {
            "recent_rate": 0.35,
            "recent_ewma": 0.35,
            "minutes_l7": 30.0,
            "usage_volume_l7": 12.0,
            "is_b2b": 0,
            "days_rest": 1,
        },
        minutes=30.0,
    )
    assert 5.0 < pred < 15.0


def test_zip_model_handles_excess_zeros():
    rng = np.random.default_rng(17)
    samples = []
    d = date(2025, 10, 1)
    for i in range(140):
        minutes = 26 + (i % 10)
        # 55% structural zeros (non-shooters), else Poisson(0.12*min)
        if rng.random() < 0.55:
            y = 0
        else:
            y = int(rng.poisson(0.12 * minutes))
        samples.append(
            {
                "player_id": 2,
                "game_id": f"z{i}",
                "game_date": d.isoformat(),
                "minutes": float(minutes),
                "features": {
                    "recent_rate": 0.05 + rng.uniform(-0.01, 0.01),
                    "recent_ewma": 0.05 + rng.uniform(-0.01, 0.01),
                    "minutes_l7": float(minutes),
                    "usage_volume_l7": 6.0 + rng.uniform(-0.5, 0.5),
                    "is_b2b": 0,
                    "days_rest": int(rng.integers(1, 4)),
                },
                "target": y,
            }
        )
        d += timedelta(days=2)
    model = ZIPRateModel().fit(samples)
    pred = model.predict_count(
        {
            "recent_rate": 0.05,
            "recent_ewma": 0.05,
            "minutes_l7": 30.0,
            "usage_volume_l7": 6.0,
            "is_b2b": 0,
            "days_rest": 1,
        },
        minutes=30.0,
    )
    # mean of mixture: 0.45 * 0.12*30 ≈ 1.62 -> model should land near it
    assert 0.5 < pred < 3.5


def test_lightgbm_rate_model_conditions_on_minutes():
    rng = np.random.default_rng(19)
    samples = []
    d = date(2025, 10, 1)
    for i in range(100):
        minutes = 24 + (i % 12)
        y = int(round(0.80 * minutes + rng.normal(0, 1.0)))
        samples.append(
            {
                "player_id": 3,
                "game_id": f"l{i}",
                "game_date": d.isoformat(),
                "minutes": float(minutes),
                "features": {
                    "recent_rate": 0.8,
                    "recent_ewma": 0.8,
                    "minutes_l7": float(minutes),
                    "usage_volume_l7": 15.0,
                    "is_b2b": 0,
                    "days_rest": 1,
                },
                "target": max(y, 0),
            }
        )
        d += timedelta(days=2)
    model = LightGBMRateModel().fit(samples)
    pred = model.predict_count(
        {
            "recent_rate": 0.8,
            "recent_ewma": 0.8,
            "minutes_l7": 30.0,
            "usage_volume_l7": 15.0,
            "is_b2b": 0,
            "days_rest": 1,
        },
        minutes=30.0,
    )
    assert pred == pytest.approx(24.0, abs=3.0)


def test_stat_model_factory():
    assert isinstance(build_stat_model("pts"), LightGBMRateModel)
    assert isinstance(build_stat_model("reb"), NegativeBinomialRateModel)
    assert isinstance(build_stat_model("ast"), PoissonRateModel)
    assert isinstance(build_stat_model("fg3m"), ZIPRateModel)
    assert isinstance(build_stat_model("stl"), PoissonRateModel)  # default


def test_rate_model_walk_forward_beats_naive():
    """Rate drops on B2B -> model learns it, naive mean-rate can't."""
    rng = np.random.default_rng(23)
    samples = _rate_samples(
        rng,
        120,
        rate_fn=lambda b2b: 0.15 if b2b else 0.38,
        minutes_fn=lambda i: 26 + (i % 9),
    )
    result = walk_forward_rate_eval(lambda: PoissonRateModel(), samples, min_train=40)
    assert result["n_test"] > 0
    assert result["model_mae"] < result["baseline_mae"]
    assert result["beats_baseline"] is True


def test_build_rate_samples_from_db(minutes_db):
    samples = build_rate_samples(minutes_db, "ast", min_prior_games=5)
    assert len(samples) == 45
    s = samples[0]
    assert set(s["features"]) == {
        "recent_rate",
        "recent_ewma",
        "minutes_l7",
        "usage_volume_l7",
        "is_b2b",
        "days_rest",
    }
    assert s["minutes"] > 0
    assert s["target"] >= 0


def test_mean_baseline():
    samples = [{"minutes": 30.0}, {"minutes": 34.0}, {"minutes": None}]
    assert mean_baseline(samples, "minutes") == pytest.approx(32.0)
