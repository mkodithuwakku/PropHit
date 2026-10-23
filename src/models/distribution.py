"""Stage 3: discrete probability distribution engine (Milestone 4).

Produces the full probability mass function P(X = k) for integer stat
outcomes k in [0, 60] per SRD 5.3, via two estimators:

  - Parametric: Negative Binomial (mean mu, dispersion r fit by
    method-of-moments on the player's historical consistency). Reduces to
    Poisson when the player is not overdispersed.
  - Monte Carlo: minutes ~ N(mu_min, sigma_min) x per-minute Poisson rate,
    optionally with gamma-distributed rates for tail volatility.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from scipy import stats

MAX_K = 60
DEFAULT_MC_TRIALS = 10_000
POISSON_LIMIT_R = 1.0e6  # r this large is indistinguishable from Poisson


def fit_dispersion(values: list[float]) -> float:
    """Method-of-moments NB dispersion r from historical outcomes.

    Var = mu + mu^2/r  =>  r = mu^2 / (var - mu).  When the data is not
    overdispersed, returns POISSON_LIMIT_R (the Poisson limit r -> inf).
    """
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    if len(arr) < 2:
        return POISSON_LIMIT_R
    mu = arr.mean()
    var = arr.var(ddof=1)
    if mu <= 0 or var <= mu:
        return POISSON_LIMIT_R
    return float(np.clip(mu * mu / (var - mu), 0.05, POISSON_LIMIT_R))


def prob_at_least(pmf: np.ndarray, k: int) -> float:
    """P(X >= k) from a PMF array (index = outcome)."""
    k = max(int(k), 0)
    if k >= len(pmf):
        return 0.0
    return float(pmf[k:].sum())


def prob_over_line(pmf: np.ndarray, line: float) -> float:
    """P(X > line): over 9.5 wins at 10+, over 10.0 wins at 11+."""
    return prob_at_least(pmf, math.floor(line) + 1)


def pmf_mean(pmf: np.ndarray) -> float:
    return float(np.dot(np.arange(len(pmf)), pmf))


class DistributionEngine:
    """Generates discrete PMFs for player stat outcomes."""

    def __init__(
        self,
        max_k: int = MAX_K,
        mc_trials: int = DEFAULT_MC_TRIALS,
        seed: int = 42,
    ):
        self.max_k = max_k
        self.mc_trials = mc_trials
        self.seed = seed
        self._support = np.arange(max_k + 1)

    def _normalize(self, pmf: np.ndarray) -> np.ndarray:
        total = pmf.sum()
        if total <= 0:
            out = np.zeros_like(pmf)
            out[0] = 1.0
            return out
        return pmf / total

    def neg_binomial_pmf(
        self,
        mu: float,
        r: Optional[float] = None,
        history: Optional[list[float]] = None,
    ) -> np.ndarray:
        """PMF of NegBin(mu, r); r None -> fit from history (or Poisson)."""
        if mu <= 0:
            pmf = np.zeros(self.max_k + 1)
            pmf[0] = 1.0
            return pmf
        if r is None:
            r = fit_dispersion(history) if history else POISSON_LIMIT_R
        if r >= POISSON_LIMIT_R:
            pmf = stats.poisson.pmf(self._support, mu)
        else:
            p = r / (r + mu)
            pmf = stats.nbinom.pmf(self._support, r, p)
        return self._normalize(pmf)

    def monte_carlo_pmf(
        self,
        minutes_mean: float,
        minutes_std: float,
        rate_mean: float,
        rate_std: Optional[float] = None,
        trials: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Simulate minutes variance x per-minute Poisson rates.

        ``rate_std`` adds a Gamma(rate_mean, rate_std) multiplier on the
        per-minute rate — capturing tail volatility beyond pure Poisson
        (the mixture is approximately Negative Binomial).
        """
        n = trials or self.mc_trials
        rng = np.random.default_rng(self.seed if seed is None else seed)
        minutes = np.clip(rng.normal(minutes_mean, minutes_std, n), 0.0, None)
        rate = np.full(n, max(rate_mean, 0.0))
        if rate_std and rate_mean > 0:
            shape = (rate_mean / rate_std) ** 2
            rate = rng.gamma(shape=shape, scale=rate_std**2 / rate_mean, size=n)
        counts = rng.poisson(minutes * rate)
        pmf = np.bincount(counts, minlength=self.max_k + 1)
        return self._normalize(pmf[: self.max_k + 1].astype(float))

    def blended_pmf(
        self,
        mu: float,
        history: Optional[list[float]],
        minutes_mean: float,
        minutes_std: float,
        rate_mean: float,
        weight_mc: float = 0.5,
    ) -> np.ndarray:
        """Average of parametric NB and Monte Carlo PMFs (default 50/50)."""
        nb = self.neg_binomial_pmf(mu, history=history)
        mc = self.monte_carlo_pmf(minutes_mean, minutes_std, rate_mean)
        return self._normalize(weight_mc * mc + (1.0 - weight_mc) * nb)
