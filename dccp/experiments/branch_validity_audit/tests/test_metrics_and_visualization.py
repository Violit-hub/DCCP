import json
from pathlib import Path

from branch_audit.artifacts import AuditStore, sha256_file
from branch_audit.config import (
    AuditConfig, CandidateConfig, MetricsConfig, PathConfig, SimulatorConfig,
    StateConfig, WorldModelConfig,
)
from branch_audit.metrics import compute_metrics, kendall_tau_b, spearman
from branch_audit.visualization import generate_inspection


def _fixture(tmp_path: Path):
    paths = PathConfig(
        dccp_root=str(tmp_path), source_state_run=str(tmp_path), source_state_config=str(tmp_path),
        policy_checkpoint=str(tmp_path), dataset_config=str(tmp_path), progress_lrm_config=str(tmp_path),
        wm_inference_config=str(tmp_path), output_root=str(tmp_path / "outputs"), mujoco_path=str(tmp_path),
    )
    cfg = AuditConfig(
        task="coffee", run_name="fixture", paths=paths,
        states=StateConfig(max_states=1),
        candidates=CandidateConfig(total_candidates=3),
        world_model=WorldModelConfig(horizon_H=1, seeds=[7]),
        simulator=SimulatorConfig(evaluation_seeds=[1, 2], save_video=False),
        metrics=MetricsConfig(bootstrap_samples=50, bootstrap_seed=3),
    )
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    frame = store.states / "s0" / "frame.png"
    frame.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image
    Image.new("RGB", (16, 16), "navy").save(frame)
    states = [{
        "state_id": "s0", "trajectory_id": "t0", "decision_index": 2, "low_level_step": 16,
        "frame_path": "states/s0/frame.png", "simulator_state_path": "states/s0/state.npz",
        "model_xml_path": "states/s0/model.xml", "nominal_action_path": "states/s0/nominal.npz",
        "nominal_tokens_path": "states/s0/tokens.npz",
    }]
    AuditStore.write_json(store.states / "manifest.json", {"states": states})
    candidates = []
    for index in range(3):
        candidates.append({
            "state_id": "s0", "candidate_index": index, "is_nominal": index == 0,
            "action_path": f"candidates/s0/candidate_{index:02d}.npz", "seed": index, "token_hash": str(index),
        })
    AuditStore.write_json(store.candidates / "manifest.json", {"candidates": candidates})
    labels = [{
        "state_id": "s0",
        "candidate_mean_scores": {"0": 0.2, "1": 0.8, "2": 0.05},
        "candidate_scores_by_wm_seed": {"0": {"7": 0.2}, "1": {"7": 0.8}, "2": {"7": 0.05}},
        "predicted_top_candidates": [1], "predicted_bottom_candidates": [2],
        "dccp_emitted_pairs": [
            {"alternative_candidate_index": 1, "winner_candidate_index": 1, "loser_candidate_index": 0, "margin_alt_minus_nominal": 0.6, "weight_abs_margin": 0.6},
            {"alternative_candidate_index": 2, "winner_candidate_index": 0, "loser_candidate_index": 2, "margin_alt_minus_nominal": -0.15, "weight_abs_margin": 0.15},
        ],
    }]
    labels_path = store.labels / "labels.jsonl"
    AuditStore.write_jsonl(labels_path, labels)
    AuditStore.write_json(store.labels / "LABELS_FROZEN.json", {"labels_sha256": sha256_file(labels_path)})
    outcomes = []
    for candidate_index, successes, short in ((0, [False, False], 0.4), (1, [True, True], 1.0), (2, [False, False], 0.0)):
        for seed, success in zip((1, 2), successes):
            outcomes.append({
                "state_id": "s0", "candidate_index": candidate_index, "evaluation_seed": seed,
                "success": success, "short_progress": short, "final_progress": float(success),
                "executed_low_level_steps": 24, "status": "OK", "video_path": None,
            })
    AuditStore.write_jsonl(store.simulator / "outcomes.jsonl", outcomes)
    return cfg, store


def test_metrics_cover_pairs_top1_and_regret(tmp_path):
    cfg, store = _fixture(tmp_path)
    report = compute_metrics(cfg)
    summary = report["summary"]
    assert summary["num_emitted_pairs"] == 2
    assert summary["strict_pair_accuracy_all_pairs"] == 0.5
    assert summary["decisive_pair_accuracy"] == 1.0
    assert summary["pair_nonworse_rate"] == 1.0
    assert summary["winner_loser_success_lift"] == 0.5
    assert summary["top1_accuracy_tie_aware"] == 1.0
    assert summary["mean_top1_success_regret"] == 0.0
    assert (store.reports / "pair_metrics.csv").exists()


def test_visualization_is_available_at_partial_or_complete_stage(tmp_path):
    cfg, store = _fixture(tmp_path)
    compute_metrics(cfg)
    index = generate_inspection(cfg)
    content = index.read_text(encoding="utf-8")
    assert "Candidate 00" in content
    assert "predicted top" in content
    assert "DCCP 分支标签有效性审计" in content
    assert (store.inspection / "manifest.json").exists()


def test_rank_correlations_handle_ties():
    assert spearman([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 1.0
    assert kendall_tau_b([0.0, 1.0, 2.0], [0.0, 1.0, 2.0]) == 1.0
