"""Reward computation for DPO training (like SREG's rubric.py).

Single authority for converting verifier scores to training rewards.
Called by training/env.py and format_dpo.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from verifier import VerifierTool


def score_submission(chosen, rejected, eval_type="decision"):
    """Score dispatcher for DPO reward. Returns float [0.0, 1.0].

    Args:
        chosen: dict with reasoning, action, code (privi's decision)
        rejected: dict with reasoning, action, code (base's decision)
        eval_type: "decision" (per-step) or "trajectory" (full)

    Returns:
        Preference margin: how much better chosen is than rejected.
        1.0 = maximally different, 0.0 = identical.
    """
    if eval_type == "decision":
        return VerifierTool.score_pair_divergence(chosen, rejected)

    return 0.0


def filter_pairs(pairs, min_divergence=0.2):
    """Filter pairs by minimum divergence score."""
    return [p for p in pairs if p.get("divergence_score", 0) >= min_divergence]
