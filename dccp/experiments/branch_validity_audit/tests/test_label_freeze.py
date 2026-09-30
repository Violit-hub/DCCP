from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from branch_audit.artifacts import AuditStore
from branch_audit.config import AuditConfig, CandidateConfig, PathConfig, PreferenceConfig, StateConfig, WorldModelConfig
from branch_audit.lrm_runner import score_predictions, verify_frozen_labels


class FakeScorer:
    def score_local_progress(self, frames, instruction, metadata):
        tables = {5: {0: 0.3, 1: 0.8, 2: 0.1}, 6: {0: 0.3, 1: 0.0, 2: 0.9}}
        return tables[int(metadata["wm_seed"])][int(metadata["candidate_index"])]


def make_cfg(tmp_path: Path):
    paths = PathConfig(
        dccp_root=str(tmp_path), source_state_run=str(tmp_path), source_state_config=str(tmp_path),
        policy_checkpoint=str(tmp_path), dataset_config=str(tmp_path), progress_lrm_config=str(tmp_path),
        wm_inference_config=str(tmp_path), output_root=str(tmp_path / "out"), mujoco_path=str(tmp_path),
    )
    return AuditConfig(
        task="coffee", run_name="freeze", paths=paths,
        states=StateConfig(max_states=1), candidates=CandidateConfig(total_candidates=3),
        world_model=WorldModelConfig(horizon_H=1, seeds=[5], save_video=False),
        preferences=PreferenceConfig(margin_pos=0.1, margin_neg=0.1),
    )


def make_predictions(cfg):
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    AuditStore.write_json(store.states / "manifest.json", {"states": [{"state_id": "s"}]})
    candidates, predictions = [], []
    for index in range(3):
        candidates.append({"state_id": "s", "candidate_index": index})
        for wm_seed in cfg.world_model.seeds:
            path = store.world_model / f"candidate_{index}_seed_{wm_seed}.npz"
            AuditStore.write_npz(path, frames=np.zeros((3, 8, 8, 3), dtype=np.uint8))
            predictions.append({
                "state_id": "s", "candidate_index": index, "wm_seed": wm_seed,
                "frames_path": str(path.relative_to(store.run_dir)), "status": "OK",
            })
    AuditStore.write_json(store.candidates / "manifest.json", {"candidates": candidates})
    AuditStore.write_jsonl(store.world_model / "predictions.jsonl", predictions)
    return store


def test_scores_freeze_before_simulator_and_tamper_is_detected(tmp_path):
    cfg = make_cfg(tmp_path)
    store = make_predictions(cfg)
    labels = score_predictions(cfg, scorer=FakeScorer(), freeze=True)
    assert labels[0]["predicted_top_candidates"] == [1]
    assert [(row["winner_candidate_index"], row["loser_candidate_index"]) for row in labels[0]["dccp_emitted_pairs"]] == [(1, 0), (0, 2)]
    verify_frozen_labels(cfg)
    with (store.labels / "labels.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="modified"):
        verify_frozen_labels(cfg)


def test_ground_truth_before_freeze_is_rejected(tmp_path):
    cfg = make_cfg(tmp_path)
    store = make_predictions(cfg)
    AuditStore.write_jsonl(store.simulator / "outcomes.jsonl", [{"leak": True}])
    with pytest.raises(RuntimeError, match="Simulator outcomes already exist"):
        score_predictions(cfg, scorer=FakeScorer(), freeze=True)


def test_primary_seed_labels_are_not_replaced_by_seed_ensemble(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg = replace(cfg, world_model=replace(cfg.world_model, seeds=[5, 6]))
    make_predictions(cfg)
    label = score_predictions(cfg, scorer=FakeScorer(), freeze=True)[0]
    assert label["label_aggregation"] == "primary_seed"
    assert label["primary_wm_seed"] == 5
    assert label["predicted_top_candidates"] == [1]
    assert label["mean_score_top_candidates"] == [2]
    assert set(label["dccp_pairs_by_wm_seed"]) == {"5", "6"}
