from branch_audit.preferences import build_dccp_labels, extreme_candidates


def test_exact_dccp_nominal_threshold_semantics():
    pairs = build_dccp_labels(
        {0: 0.50, 1: 0.61, 2: 0.40, 3: 0.60, 4: 0.39},
        margin_pos=0.10,
        margin_neg=0.10,
        max_pairs_per_state=None,
    )
    # Strict inequalities: +/-0.10 are discarded. Equal weights sort by candidate index.
    assert [(row["winner_candidate_index"], row["loser_candidate_index"]) for row in pairs] == [
        (1, 0), (0, 4)
    ]


def test_extreme_candidates_preserves_ties():
    assert extreme_candidates({0: 0.1, 1: 0.8, 2: 0.8, 3: 0.1}) == ([1, 2], [0, 3])
