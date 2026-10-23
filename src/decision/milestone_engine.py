"""Milestone / alt-line ladder fair-odds pricing (Milestone 4).

Implements SRD 6.2. When the model projection exceeds the consensus line
by >= trigger_diff, the engine prices alternate "N or more" milestones so
the user can spot mispriced Bet365 ladders that consensus feeds don't list.

Milestone thresholds are the round (even) numbers strictly above the line,
matching the spec examples: line 9.5 -> 10+, 12+, 14+; line 12.5 -> 14+,
16+, 18+.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from src.decision.ev_calculator import prob_to_fair_american
from src.models.distribution import prob_at_least, prob_over_line

DEFAULT_TRIGGER_DIFF = 2.0
DEFAULT_OFFSETS = (0, 2, 4)
DEFAULT_ALT_MARGIN = 25  # target = fair odds + 25 cents of safety margin


def _base_threshold(line: float) -> int:
    """Smallest even integer strictly above the line."""
    k = math.floor(line) + 1
    return k if k % 2 == 0 else k + 1


class MilestoneEngine:
    """Prices milestone ladders and flags alt-line opportunities."""

    def __init__(
        self,
        trigger_diff: float = DEFAULT_TRIGGER_DIFF,
        offsets: tuple[int, ...] = DEFAULT_OFFSETS,
        alt_margin: int = DEFAULT_ALT_MARGIN,
    ):
        self.trigger_diff = trigger_diff
        self.offsets = offsets
        self.alt_margin = alt_margin

    def build_ladder(
        self,
        pmf: np.ndarray,
        consensus_line: float,
        model_mean: float,
        player_name: Optional[str] = None,
        market: Optional[str] = None,
    ) -> dict:
        """Full milestone evaluation for one player prop.

        Returns the base-line fair price plus priced milestone rungs:
        ``{"k", "p_over", "fair_american", "alt_target_american"}``.
        ``triggered`` indicates the model-vs-consensus gap clears the
        configured trigger_diff.
        """
        p_base = prob_over_line(pmf, consensus_line)
        base = _base_threshold(consensus_line)

        rungs = []
        for offset in self.offsets:
            k = base + offset
            p = prob_at_least(pmf, k)
            if p <= 0.0:
                continue
            fair = prob_to_fair_american(p)
            rungs.append(
                {
                    "k": k,
                    "p_over": p,
                    "fair_american": fair,
                    "alt_target_american": fair + self.alt_margin,
                }
            )

        return {
            "player_name": player_name,
            "market": market,
            "consensus_line": consensus_line,
            "model_mean": model_mean,
            "model_edge": model_mean - consensus_line,
            "triggered": model_mean - consensus_line >= self.trigger_diff,
            "base_over": {
                "p_over": p_base,
                "fair_american": prob_to_fair_american(p_base),
            },
            "milestones": rungs,
        }
