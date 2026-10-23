"""Expected value & implied-probability margins (Milestone 4).

Implements SRD 6.1:
  P(Over L) = sum_{k > L} P(X = k)
  Implied Prob = |odds| / (|odds| + 100)   [negative odds]
               = 100 / (odds + 100)       [positive odds]
  EV = P_model * NetPayout - (1 - P_model) * Stake
Flag a Recommended Play when EV >= +5.0% (configurable).
"""

from __future__ import annotations

import numpy as np

from src.models.distribution import prob_at_least, prob_over_line

DEFAULT_MIN_EV_PCT = 5.0


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds of 0 are invalid")
    if odds > 0:
        return 1.0 + odds / 100.0
    return 1.0 + 100.0 / abs(odds)


def decimal_to_american(decimal: float) -> int:
    if decimal <= 1.0:
        raise ValueError("decimal odds must exceed 1.0")
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100.0))
    return int(round(-100.0 / (decimal - 1.0)))


def american_to_implied_prob(odds: float) -> float:
    return 1.0 / american_to_decimal(odds)


def prob_to_fair_american(p: float) -> int:
    """Fair (vig-free) American odds for a win probability."""
    p = float(np.clip(p, 1e-9, 1.0 - 1e-9))
    return decimal_to_american(1.0 / p)


def expected_value_pct(
    p_model: float, american_odds: float, stake: float = 1.0
) -> float:
    """EV as a percentage of stake. +5.0 means 5% expected profit."""
    net_payout = (american_to_decimal(american_odds) - 1.0) * stake
    ev = p_model * net_payout - (1.0 - p_model) * stake
    return 100.0 * ev / stake


class EVCalculator:
    """Compares model probabilities to consensus American odds."""

    def __init__(
        self,
        min_ev_pct: float = DEFAULT_MIN_EV_PCT,
        min_american_odds: Optional[float] = None,
        max_american_odds: Optional[float] = None,
    ):
        self.min_ev_pct = min_ev_pct
        self.min_american_odds = min_american_odds
        self.max_american_odds = max_american_odds

    def _odds_allowed(self, odds: float) -> bool:
        if self.min_american_odds is not None and odds < self.min_american_odds:
            return False
        if self.max_american_odds is not None and odds > self.max_american_odds:
            return False
        return True

    def evaluate_line(
        self,
        pmf: np.ndarray,
        line: float,
        over_price: Optional[float] = None,
        under_price: Optional[float] = None,
        milestone_k: Optional[int] = None,
    ) -> dict:
        """Evaluate a prop line against the model PMF.

        Returns probabilities, implied probs, EV%, and the recommended
        side (if any meets the +EV threshold and odds limits).
        """
        p_over = (
            prob_at_least(pmf, milestone_k)
            if milestone_k is not None
            else prob_over_line(pmf, line)
        )
        # Integer lines can push: the mass at exactly `line` wins for neither
        # side, so under-win prob is 1 - p_over - p_push (not 1 - p_over).
        p_push = (
            float(pmf[int(line)])
            if milestone_k is None and float(line).is_integer() and int(line) < len(pmf)
            else 0.0
        )
        p_under = 1.0 - p_over - p_push

        result = {
            "line": line,
            "p_over": p_over,
            "p_under": p_under,
            "over_price": over_price,
            "under_price": under_price,
            "fair_over_american": prob_to_fair_american(p_over),
            "fair_under_american": prob_to_fair_american(p_under),
            "ev_over_pct": None,
            "ev_under_pct": None,
            "recommended_side": None,
            "recommended_ev_pct": None,
        }

        result["p_push"] = p_push

        def _ev_pct(p_win: float, price: float) -> float:
            # Win pays decimal `price`, push returns stake, loss pays 0.
            return 100.0 * (p_win * american_to_decimal(price) + p_push - 1.0)

        candidates = []
        if over_price is not None and self._odds_allowed(over_price):
            ev = _ev_pct(p_over, over_price)
            result["ev_over_pct"] = ev
            candidates.append(("over", ev))
        if under_price is not None and self._odds_allowed(under_price):
            ev = _ev_pct(p_under, under_price)
            result["ev_under_pct"] = ev
            candidates.append(("under", ev))

        best = max(candidates, key=lambda c: c[1], default=None)
        if best and best[1] >= self.min_ev_pct:
            result["recommended_side"] = best[0]
            result["recommended_ev_pct"] = best[1]
        return result
