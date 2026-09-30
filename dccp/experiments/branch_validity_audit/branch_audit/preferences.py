"""Pure reconstruction of DCCP nominal-vs-alternative high-margin labels."""

from __future__ import annotations

from typing import Any


def build_dccp_labels(
    candidate_scores: dict[int, float],
    *,
    margin_pos: float,
    margin_neg: float,
    max_pairs_per_state: int | None,
) -> list[dict[str, Any]]:
    if 0 not in candidate_scores:
        raise ValueError("Candidate 0 (nominal) is required")
    nominal = float(candidate_scores[0])
    pairs = []
    for candidate_index in sorted(index for index in candidate_scores if index != 0):
        alternative = float(candidate_scores[candidate_index])
        margin = alternative - nominal
        if margin > float(margin_pos):
            winner, loser = candidate_index, 0
        elif margin < -float(margin_neg):
            winner, loser = 0, candidate_index
        else:
            continue
        pairs.append({
            "alternative_candidate_index": candidate_index,
            "winner_candidate_index": winner,
            "loser_candidate_index": loser,
            "nominal_score": nominal,
            "alternative_score": alternative,
            "margin_alt_minus_nominal": margin,
            "weight_abs_margin": abs(margin),
        })
    pairs.sort(key=lambda row: (-float(row["weight_abs_margin"]), int(row["alternative_candidate_index"])))
    if max_pairs_per_state is not None:
        pairs = pairs[: max(0, int(max_pairs_per_state))]
    return pairs


def extreme_candidates(candidate_scores: dict[int, float]) -> tuple[list[int], list[int]]:
    if not candidate_scores:
        return [], []
    maximum, minimum = max(candidate_scores.values()), min(candidate_scores.values())
    winners = sorted(index for index, score in candidate_scores.items() if score == maximum)
    losers = sorted(index for index, score in candidate_scores.items() if score == minimum)
    return winners, losers
