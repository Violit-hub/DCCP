"""使用合成轨迹测试闸门、过滤器、MP4 和离线检查页面。"""

from __future__ import annotations

import json

import imageio.v2 as imageio
import numpy as np
import pytest
from PIL import Image

from state_audit.artifact_store import ArtifactStore
from state_audit.branch_evaluator import EvaluationScope
from state_audit.config import AuditConfig, PathConfig
from state_audit.stage_gates import require_restore_gate
from state_audit.video import BranchVideoRecorder
from state_audit.visualization import generate_inspection_site


def _config(tmp_path) -> AuditConfig:
    return AuditConfig(
        task="coffee",
        run_name="synthetic_visual_check",
        paths=PathConfig(
            dccp_root=str(tmp_path),
            policy_checkpoint=str(tmp_path / "policy"),
            dataset_config=str(tmp_path / "dataset.json"),
            initial_states=str(tmp_path / "states.pkl"),
            progress_lrm_config=str(tmp_path / "lrm.yaml"),
            output_root=str(tmp_path / "outputs"),
        ),
    )


def test_restore_gate_is_enforced(tmp_path):
    cfg = _config(tmp_path)
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    with pytest.raises(RuntimeError, match="尚未执行"):
        require_restore_gate(cfg)
    ArtifactStore.write_json_atomic(store.report_dir / "state_restore_gate.json", {"passed": False})
    with pytest.raises(RuntimeError, match="未通过"):
        require_restore_gate(cfg)
    ArtifactStore.write_json_atomic(store.report_dir / "state_restore_gate.json", {"passed": True})
    assert require_restore_gate(cfg)["passed"] is True


def test_evaluation_scope_matches_exact_branch():
    scope = EvaluationScope(
        trajectory_id="traj_0001",
        decision_index=3,
        candidate_index=2,
        evaluation_seed=101,
        max_branches=1,
        save_video=True,
    )
    assert scope.matches("traj_0001", 3, 2, 101)
    assert not scope.matches("traj_0001", 4, 2, 101)
    assert not scope.matches("traj_0000", 3, 2, 101)


def test_mp4_recorder_writes_readable_video(tmp_path):
    path = tmp_path / "branch.mp4"
    with BranchVideoRecorder(path, fps=5) as recorder:
        for index in range(4):
            frame = np.zeros((32, 48, 3), dtype=np.uint8)
            frame[:, :, index % 3] = 50 + index * 40
            recorder.append(frame)
    assert path.exists() and path.stat().st_size > 0
    reader = imageio.get_reader(path)
    first = reader.get_data(0)
    reader.close()
    assert first.shape == (32, 48, 3)


def test_generate_complete_inspection_site_from_synthetic_artifacts(tmp_path):
    cfg = _config(tmp_path)
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    ArtifactStore.write_json_atomic(store.report_dir / "state_restore_gate.json", {"passed": True})

    steps = []
    for decision_index in range(5):
        frame_path = store.nominal_dir / "traj_0000" / "frames" / f"step_{decision_index:04d}.png"
        frame_path.parent.mkdir(parents=True, exist_ok=True)
        frame = np.zeros((72, 96, 3), dtype=np.uint8)
        frame[:, :, 0] = decision_index * 40
        frame[:, :, 1] = 160 - decision_index * 20
        Image.fromarray(frame).save(frame_path)
        steps.append(
            {
                "trajectory_id": "traj_0000",
                "decision_index": decision_index,
                "low_level_step": decision_index * 8,
                "frame_path": str(frame_path.relative_to(store.run_dir)),
                "entropy": 0.1 + decision_index * 0.2,
            }
        )
    trajectory = {
        "trajectory_id": "traj_0000",
        "success": False,
        "finish_low_level_step": 40,
        "num_decisions": 5,
        "steps": steps,
    }
    ArtifactStore.write_json_atomic(store.nominal_dir / "trajectories.json", [trajectory])
    ArtifactStore.write_jsonl_atomic(
        store.selection_dir / "selected_states.jsonl",
        [
            {
                "trajectory_id": "traj_0000",
                "method": "curvature",
                "selected_indices": [1],
                "eligible_indices": [1, 2, 3],
            },
            {
                "trajectory_id": "traj_0000",
                "method": "joint",
                "selected_indices": [2],
                "eligible_indices": [1, 2, 3],
            },
        ],
    )
    ArtifactStore.write_json_atomic(store.selection_dir / "SELECTIONS_FROZEN.json", {"num_rows": 2})
    ArtifactStore.write_npz_atomic(
        store.selection_dir / "traj_0000_scores.npz",
        valid_start_indices=np.arange(5),
        progress_scores=np.asarray([0.0, 0.2, 0.7, 0.8, 1.0]),
        curvature_scores=np.asarray([0.0, 0.3, 0.8, 0.2, 0.0]),
        entropy_scores=np.asarray([0.1, 0.3, 0.5, 0.7, 0.9]),
        decision_scores=np.asarray([0.0, 0.4, 1.0, 0.5, 0.0]),
        valid_mask=np.asarray([False, True, True, True, False]),
    )
    ArtifactStore.write_jsonl_atomic(
        store.branch_dir / "branch_results.jsonl",
        [
            {
                "trajectory_id": "traj_0000",
                "decision_index": 1,
                "candidate_index": 0,
                "evaluation_seed": 101,
                "status": "OK",
                "success": False,
                "video_path": None,
            }
        ],
    )
    ArtifactStore.write_json_atomic(
        store.report_dir / "summary.json",
        [
            {
                "method": "joint",
                "num_trajectories": 1,
                "mean_improvement": 0.5,
                "opportunity_rate": 1.0,
                "macro_capture_ratio": 0.8,
                "micro_capture_ratio": 0.8,
            }
        ],
    )

    index = generate_inspection_site(cfg)
    page = index.read_text(encoding="utf-8")
    assert "Coffee 状态选择审计" in page
    assert "traj_0000" in page and "curvature" in page and "joint" in page
    assert (index.parent / "assets" / "traj_0000_contact.jpg").exists()
    assert (index.parent / "assets" / "traj_0000_selection.svg").exists()
    manifest = json.loads((index.parent / "inspection_manifest.json").read_text())
    assert manifest["num_trajectories"] == 1
    assert manifest["num_selection_rows"] == 2
