from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class TrialResult:
    identity_score: float
    pad_score: float
    is_attack: bool
    accepted: bool


def evaluate_trial(
    identity_score: float,
    pad_score: float,
    tau_id: float,
    tau_pad: float,
) -> bool:
    """Accept only if both the identity and PAD checks pass."""
    return bool(identity_score >= tau_id and pad_score >= tau_pad)


def summarize_trial_results(results: Sequence[dict]) -> dict:
    """Summarize trial outcomes and counts for a batch of evaluations."""
    total = len(results)
    accepted_trials = sum(1 for r in results if r.get("accepted", False))
    attack_successes = sum(1 for r in results if r.get("is_attack", False) and r.get("accepted", False))
    genuine_accepts = sum(1 for r in results if not r.get("is_attack", False) and r.get("accepted", False))
    rejected_genuine = sum(1 for r in results if not r.get("is_attack", False) and not r.get("accepted", False))
    attack_trials = sum(1 for r in results if r.get("is_attack", False))

    return {
        "total_trials": total,
        "accepted_trials": accepted_trials,
        "attack_successes": attack_successes,
        "attack_trials": attack_trials,
        "genuine_accepts": genuine_accepts,
        "rejected_genuine": rejected_genuine,
    }


def threshold_search(
    genuine_scores: Sequence[float],
    impostor_scores: Sequence[float],
    min_score: float = 0.0,
    max_score: float = 1.0,
    step: float = 0.01,
) -> float:
    """Pick an identity threshold that best separates genuine and impostor scores."""
    best_tau = min_score
    best_objective = float("-inf")

    candidate_values = [min_score + i * step for i in range(int((max_score - min_score) / step) + 1)]

    for tau in candidate_values:
        genuine_pass = sum(1 for s in genuine_scores if s >= tau)
        impostor_pass = sum(1 for s in impostor_scores if s >= tau)

        # Prefer thresholds that accept genuine users while rejecting impostors,
        # and break ties by choosing a stricter threshold.
        objective = (genuine_pass - impostor_pass) + (tau / max(1.0, max_score - min_score + 1e-9))
        if objective > best_objective:
            best_objective = objective
            best_tau = tau

    return best_tau
