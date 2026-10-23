"""M4 verification: PMF engine, +EV comparator, milestone ladders.

Fair-odds pricing is checked against known statistical distributions
(scipy references) — the M4 verification gate.
"""

import numpy as np
import pytest
from scipy import stats

from src.decision.ev_calculator import (
    EVCalculator,
    american_to_decimal,
    american_to_implied_prob,
    decimal_to_american,
    expected_value_pct,
    prob_to_fair_american,
)
from src.decision.milestone_engine import MilestoneEngine, _base_threshold
from src.models.distribution import (
    POISSON_LIMIT_R,
    DistributionEngine,
    fit_dispersion,
    pmf_mean,
    prob_at_least,
    prob_over_line,
)


@pytest.fixture
def engine():
    return DistributionEngine(max_k=60, mc_trials=10_000, seed=7)


# --- Negative Binomial PMF ----------------------------------------------------


def test_nb_pmf_matches_scipy(engine):
    mu, r = 12.0, 5.0
    pmf = engine.neg_binomial_pmf(mu, r=r)
    expected = stats.nbinom.pmf(np.arange(61), r, r / (r + mu))
    expected /= expected.sum()
    assert pmf == pytest.approx(expected, abs=1e-9)
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf_mean(pmf) == pytest.approx(mu, abs=0.2)


def test_pmf_collapses_to_poisson_without_overdispersion(engine):
    mu = 10.0
    pmf = engine.neg_binomial_pmf(mu)  # default r -> Poisson limit
    expected = stats.poisson.pmf(np.arange(61), mu)
    expected /= expected.sum()
    assert pmf == pytest.approx(expected, abs=1e-9)


def test_fit_dispersion_values():
    # strongly overdispersed -> small r; underdispersed -> Poisson limit
    spread = [2, 18, 5, 22, 9, 15, 1, 20, 11, 7]
    r = fit_dispersion(spread)
    assert 0.05 <= r < 10.0
    steady = [10, 11, 9, 10, 10, 11, 9, 10]
    assert fit_dispersion(steady) == POISSON_LIMIT_R


def test_tail_probabilities(engine):
    pmf = engine.neg_binomial_pmf(10.0)  # ~Poisson(10)
    # P(X >= 11) for Poisson(10) per scipy
    assert prob_at_least(pmf, 11) == pytest.approx(
        1 - stats.poisson.cdf(10, 10), abs=1e-6
    )
    # over 9.5 -> needs 10+; over 10.0 -> needs 11+
    assert prob_over_line(pmf, 9.5) == pytest.approx(
        1 - stats.poisson.cdf(9, 10), abs=1e-6
    )
    assert prob_over_line(pmf, 10.0) == pytest.approx(
        1 - stats.poisson.cdf(10, 10), abs=1e-6
    )
    # tail probs are non-increasing in k
    ks = [prob_at_least(pmf, k) for k in (8, 10, 12, 15)]
    assert ks == sorted(ks, reverse=True)


# --- Monte Carlo engine --------------------------------------------------------


def test_monte_carlo_is_deterministic_and_calibrated(engine):
    pmf1 = engine.monte_carlo_pmf(30.0, 3.0, 0.40)
    pmf2 = engine.monte_carlo_pmf(30.0, 3.0, 0.40)
    assert np.array_equal(pmf1, pmf2)  # seeded -> reproducible
    assert pmf1.sum() == pytest.approx(1.0)
    # E[X] = E[minutes] * rate = 30 * 0.4 = 12
    assert pmf_mean(pmf1) == pytest.approx(12.0, abs=0.5)


def test_monte_carlo_minutes_variance_widens_distribution(engine):
    tight = engine.monte_carlo_pmf(30.0, 0.5, 0.40)
    loose = engine.monte_carlo_pmf(30.0, 8.0, 0.40)
    # wider minutes variance -> more mass in the tails
    tail_tight = prob_at_least(tight, 20) + tight[4]
    tail_loose = prob_at_least(loose, 20) + loose[4]
    assert tail_loose > tail_tight


def test_blended_pmf(engine):
    pmf = engine.blended_pmf(
        mu=12.0,
        history=[10, 14, 12, 9, 15, 11],
        minutes_mean=30.0,
        minutes_std=3.0,
        rate_mean=0.40,
    )
    assert pmf.sum() == pytest.approx(1.0)
    assert pmf_mean(pmf) == pytest.approx(12.0, abs=1.5)


# --- odds math ------------------------------------------------------------------


def test_american_decimal_implied_conversions():
    assert american_to_decimal(-110) == pytest.approx(1.90909, abs=1e-4)
    assert american_to_decimal(150) == pytest.approx(2.50)
    assert american_to_implied_prob(-110) == pytest.approx(110 / 210)
    assert american_to_implied_prob(150) == pytest.approx(0.40)
    assert decimal_to_american(2.50) == 150
    assert decimal_to_american(1.90909) == pytest.approx(-110, abs=1)
    with pytest.raises(ValueError):
        american_to_decimal(0)


def test_fair_american_matches_spec_examples():
    # SRD 6.2 example values
    assert prob_to_fair_american(0.655) == pytest.approx(-190, abs=2)
    assert prob_to_fair_american(0.444) == pytest.approx(125, abs=2)
    assert prob_to_fair_american(0.278) == pytest.approx(260, abs=3)


def test_expected_value_pct():
    # p=0.55 at -110: payout .9091 -> EV = .55*.9091 - .45 = +5.0%
    assert expected_value_pct(0.55, -110) == pytest.approx(5.0, abs=0.05)
    # same p at -120 is below +5%
    assert expected_value_pct(0.55, -120) == pytest.approx(0.83, abs=0.05)


# --- EV comparator ----------------------------------------------------------------


def test_ev_calculator_recommends_plus_ev_side():
    pmf = DistributionEngine().neg_binomial_pmf(12.0)
    calc = EVCalculator(min_ev_pct=5.0)
    # model mean 12 vs line 9.5 -> strong over lean
    res = calc.evaluate_line(pmf, line=9.5, over_price=-110, under_price=-110)
    assert res["p_over"] > 0.5
    assert res["recommended_side"] == "over"
    assert res["recommended_ev_pct"] == res["ev_over_pct"]


def test_ev_calculator_rejects_thin_edges():
    pmf = DistributionEngine().neg_binomial_pmf(10.0)
    calc = EVCalculator(min_ev_pct=5.0)
    res = calc.evaluate_line(pmf, line=9.5, over_price=-150, under_price=-150)
    assert res["recommended_side"] is None


def test_ev_calculator_odds_limits():
    pmf = DistributionEngine().neg_binomial_pmf(15.0)
    calc = EVCalculator(min_ev_pct=1.0, max_american_odds=200)
    res = calc.evaluate_line(pmf, line=9.5, over_price=None, under_price=400)
    assert res["ev_under_pct"] is None  # +400 filtered out by max odds


# --- milestone ladder -------------------------------------------------------------


def test_base_threshold_matches_spec_examples():
    assert _base_threshold(9.5) == 10
    assert _base_threshold(12.5) == 14
    assert _base_threshold(1.5) == 2


def test_milestone_ladder_spec_shape():
    engine = DistributionEngine()
    pmf = engine.neg_binomial_pmf(15.4, r=6.0)
    ladder = MilestoneEngine().build_ladder(
        pmf,
        consensus_line=12.5,
        model_mean=15.4,
        player_name="Domantas Sabonis",
        market="player_rebounds",
    )
    assert ladder["triggered"] is True
    assert ladder["model_edge"] == pytest.approx(2.9)
    assert [m["k"] for m in ladder["milestones"]] == [14, 16, 18]

    p14 = ladder["milestones"][0]["p_over"]
    p16 = ladder["milestones"][1]["p_over"]
    assert p14 > p16 > 0  # tail decreases with k
    # each rung's fair price is the vig-free conversion of its probability
    for m in ladder["milestones"]:
        assert m["fair_american"] == prob_to_fair_american(m["p_over"])
        assert m["alt_target_american"] == m["fair_american"] + 25


def test_milestone_not_triggered_without_edge():
    pmf = DistributionEngine().neg_binomial_pmf(10.0)
    ladder = MilestoneEngine().build_ladder(pmf, consensus_line=9.5, model_mean=10.4)
    assert ladder["triggered"] is False
    assert ladder["model_edge"] == pytest.approx(0.9)


def test_milestone_trigger_boundary():
    pmf = DistributionEngine().neg_binomial_pmf(11.5)
    engine = MilestoneEngine(trigger_diff=2.0)
    assert engine.build_ladder(pmf, 9.5, 11.5)["triggered"] is True
    assert engine.build_ladder(pmf, 9.5, 11.49)["triggered"] is False


def test_end_to_end_pmfs_to_ladder():
    """Full path: MC distribution -> EV line eval -> milestone ladder."""
    dist = DistributionEngine(seed=3)
    pmf = dist.monte_carlo_pmf(32.0, 4.0, 0.42)  # mean ~13.4
    ev = EVCalculator(min_ev_pct=5.0).evaluate_line(
        pmf, line=10.5, over_price=-110, under_price=-110
    )
    assert ev["p_over"] == pytest.approx(prob_over_line(pmf, 10.5))
    assert isinstance(ev["recommended_side"], (str, type(None)))

    ladder = MilestoneEngine().build_ladder(
        pmf, consensus_line=10.5, model_mean=pmf_mean(pmf)
    )
    assert ladder["triggered"] is True
    assert [m["k"] for m in ladder["milestones"]] == [12, 14, 16]
