"""不依赖 GPU、模型和模拟器的核心回归测试。"""

from __future__ import annotations

import json
import multiprocessing as mp
import stat

import numpy as np

from state_audit.artifact_store import ArtifactStore
from state_audit.metrics import compute_state_metrics, summarize_methods
from state_audit.seeding import derive_seed, temporary_seed
from state_audit.selectors import (
    draw_random_selections,
    legal_index_sets,
    select_oracle_indices,
)
def _append_rows_worker(path: str, worker_id: int, count: int) -> None:
    for row_id in range(count):
        ArtifactStore.append_jsonl(
            path, [{"worker_id": worker_id, "row_id": row_id}]
        )



def test_artifact_store_atomic_roundtrip(tmp_path):
    store = ArtifactStore(tmp_path / "run")
    store.ensure_layout()
    ArtifactStore.write_json_atomic(store.run_dir / "x.json", {"中文": np.int64(3)})
    assert json.loads((store.run_dir / "x.json").read_text())["中文"] == 3
    ArtifactStore.write_npz_atomic(store.run_dir / "x.npz", values=np.arange(4))
    with np.load(store.run_dir / "x.npz", allow_pickle=False) as data:
        assert data["values"].tolist() == [0, 1, 2, 3]
    rows = [{"a": 1}, {"a": 2}]
    ArtifactStore.write_jsonl_atomic(store.run_dir / "x.jsonl", rows)
    assert ArtifactStore.read_jsonl(store.run_dir / "x.jsonl") == rows
    assert stat.S_IMODE((store.run_dir / "x.json").stat().st_mode) == 0o660
    assert stat.S_IMODE((store.run_dir / "x.npz").stat().st_mode) == 0o660
    assert stat.S_IMODE(store.run_dir.stat().st_mode) == 0o2775


def test_seed_is_stable_and_temporary():
    assert derive_seed("traj", 2) == derive_seed("traj", 2)
    np.random.seed(5)
    before = np.random.get_state()
    with temporary_seed(99):
        first = np.random.rand()
    after = np.random.get_state()
    assert first == np.random.RandomState(99).rand()
    assert np.array_equal(before[1], after[1])


def test_legal_random_and_oracle_share_constraints():
    valid = [True, True, True, False, True]
    legal = legal_index_sets(valid, budget=2, nms_gap=1)
    assert (0, 2) in legal
    assert (0, 1) not in legal
    draws = draw_random_selections(valid, budget=2, nms_gap=1, draws=50, seed=7)
    assert len(draws) == 50
    assert all(item in legal for item in draws)
    oracle = select_oracle_indices([0.1, 0.9, 0.4, 8.0, 0.8], valid, budget=2, nms_gap=1)
    assert oracle == (1, 4)


def _branch_rows():
    rows = []
    outcomes = {
        0: {0: [False, False], 1: [True, True]},
        1: {0: [True, True], 1: [True, True]},
        2: {0: [False, False], 1: [False, True]},
    }
    for state, candidates in outcomes.items():
        for candidate, values in candidates.items():
            for seed, success in zip([101, 202], values):
                rows.append(
                    {
                        "trajectory_id": "traj_0000",
                        "decision_index": state,
                        "candidate_index": candidate,
                        "is_nominal": candidate == 0,
                        "evaluation_seed": seed,
                        "success": success,
                        "status": "OK",
                    }
                )
    return rows


def test_metrics_and_capture_ratio():
    metrics = compute_state_metrics(_branch_rows())
    assert [item.improvement for item in metrics] == [1.0, 0.0, 0.5]
    selections = [
        {
            "trajectory_id": "traj_0000",
            "method": "joint",
            "selected_indices": [0, 2],
            "eligible_indices": [0, 1, 2],
        },
        {
            "trajectory_id": "traj_0000",
            "method": "curvature",
            "selected_indices": [1],
            "eligible_indices": [0, 1, 2],
        },
    ]
    per_trajectory, summary = summarize_methods(
        metrics,
        selections,
        state_budget=2,
        nms_gap=0,
        random_draws=20,
        random_seed=3,
    )
    joint = next(row for row in per_trajectory if row["method"] == "joint")
    assert joint["capture_ratio"] == 1.0
    assert {row["method"] for row in summary} == {"curvature", "joint", "oracle", "random"}
