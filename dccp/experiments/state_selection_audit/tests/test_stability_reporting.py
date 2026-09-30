"""固定候选确认性统计的纯 CPU 回归测试。"""

from __future__ import annotations

import pytest

from state_audit.stability_reporting import compute_confirmatory_state


def test_confirmatory_stability_uses_fixed_candidate_and_paired_seeds():
    seeds = list(range(25))
    nominal = {seed: seed < 2 for seed in seeds}
    candidate = {seed: seed < 22 for seed in seeds}
    row = compute_confirmatory_state(
        trajectory_id="traj_0000",
        decision_index=3,
        confirmatory_candidate_index=5,
        nominal_outcomes=nominal,
        candidate_outcomes=candidate,
        bootstrap_samples=2000,
        bootstrap_seed=17,
    )
    assert row["paired_seed_count"] == 25
    assert row["nominal_success_rate"] == 2 / 25
    assert row["candidate_success_rate"] == 22 / 25
    assert row["paired_improvement"] == 20 / 25
    assert row["rescued_nominal_failures"] == 20
    assert row["harmed_nominal_successes"] == 0
    assert row["paired_improvement_ci95_low"] > 0
    assert row["exact_mcnemar_p"] < 0.05


def test_confirmatory_stability_rejects_unpaired_seed_sets():
    with pytest.raises(ValueError, match="paired seeds"):
        compute_confirmatory_state(
            trajectory_id="traj_0000",
            decision_index=3,
            confirmatory_candidate_index=5,
            nominal_outcomes={1: False},
            candidate_outcomes={2: True},
            bootstrap_samples=10,
            bootstrap_seed=1,
        )
