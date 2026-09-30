"""Score predicted videos, reconstruct DCCP pairs, then immutably freeze labels."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import AuditStore, sha256_file, stable_hash
from .candidates import load_candidates
from .config import AuditConfig
from .preferences import build_dccp_labels, extreme_candidates
from .source_states import load_states


@dataclass(frozen=True)
class ScoringScope:
    state_id: str | None = None
    candidate_index: int | None = None
    wm_seed: int | None = None
    max_scores: int | None = None
    rerun_completed: bool = False

    def matches(self, state_id: str, candidate_index: int, wm_seed: int) -> bool:
        return (
            (self.state_id is None or self.state_id == state_id)
            and (self.candidate_index is None or self.candidate_index == candidate_index)
            and (self.wm_seed is None or self.wm_seed == wm_seed)
        )


def _latest_success(rows: list[dict]) -> dict[tuple[str, int, int], dict]:
    latest = {}
    for row in rows:
        key = str(row["state_id"]), int(row["candidate_index"]), int(row["wm_seed"])
        if row.get("status") == "OK":
            latest[key] = row
    return latest


def score_predictions(
    cfg: AuditConfig,
    scope: ScoringScope | None = None,
    *,
    freeze: bool = True,
    scorer=None,
) -> list[dict]:
    from state_audit.progress import load_progress_scorer

    scope = scope or ScoringScope()
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    frozen_path = store.labels / "LABELS_FROZEN.json"
    if frozen_path.exists():
        if scope.rerun_completed:
            raise RuntimeError("Labels are frozen; use a new run_name instead of rerunning scores")
        return AuditStore.read_jsonl(store.labels / "labels.jsonl")
    predictions = _latest_success(AuditStore.read_jsonl(store.world_model / "predictions.jsonl"))
    expected = {
        (str(state["state_id"]), int(candidate["candidate_index"]), int(seed))
        for state in load_states(cfg)
        for candidate in load_candidates(cfg)
        if str(candidate["state_id"]) == str(state["state_id"])
        for seed in cfg.world_model.seeds
    }
    missing_predictions = sorted(expected - set(predictions))
    if missing_predictions and freeze:
        raise RuntimeError(
            f"Cannot freeze labels: {len(missing_predictions)} WM predictions missing; first={missing_predictions[:3]}"
        )
    score_path = store.labels / "prediction_scores.jsonl"
    scored = _latest_success(AuditStore.read_jsonl(score_path))
    if scorer is None:
        scorer = load_progress_scorer(
            cfg.paths.progress_lrm_config, cache_dir=store.labels / "lrm_cache"
        )
    attempted = 0
    for key in sorted(predictions):
        if key not in expected:
            continue
        state_id, candidate_index, wm_seed = key
        if not scope.matches(*key):
            continue
        if cfg.runtime.resume and key in scored and not scope.rerun_completed:
            continue
        if scope.max_scores is not None and attempted >= scope.max_scores:
            break
        attempted += 1
        prediction = predictions[key]
        try:
            frames_path = prediction.get("frames_path")
            if not frames_path:
                raise RuntimeError("LRM scoring requires world_model.save_npz=true")
            with np.load(store.run_dir / frames_path, allow_pickle=False) as data:
                frames = np.asarray(data["frames"], dtype=np.uint8)
            score = float(scorer.score_local_progress(
                frames,
                cfg.policy.instruction,
                metadata={
                    "audit": "branch_validity", "state_id": state_id,
                    "candidate_index": candidate_index, "wm_seed": wm_seed,
                    "source": "world_model_prediction",
                },
            ))
            row = {
                "state_id": state_id, "candidate_index": candidate_index,
                "wm_seed": wm_seed, "score": score, "status": "OK",
                "frames_path": frames_path,
            }
        except Exception as exc:
            row = {
                "state_id": state_id, "candidate_index": candidate_index,
                "wm_seed": wm_seed, "score": None, "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
            }
        AuditStore.append_jsonl(score_path, [row])
        if row["status"] == "OK":
            scored[key] = row
        print(f"[lrm] {key} status={row['status']} score={row.get('score')}", flush=True)

    if not freeze:
        return AuditStore.read_jsonl(score_path)
    missing_scores = sorted(expected - set(scored))
    if missing_scores:
        raise RuntimeError(
            f"Cannot freeze labels: {len(missing_scores)} LRM scores missing; first={missing_scores[:3]}"
        )
    if (store.simulator / "outcomes.jsonl").exists():
        raise RuntimeError(
            "Simulator outcomes already exist but labels are not frozen. Use a fresh run_name "
            "to preserve the prediction-before-ground-truth protocol."
        )
    label_rows = []
    for state in load_states(cfg):
        state_id = str(state["state_id"])
        by_candidate: dict[int, list[float]] = {}
        per_seed: dict[str, dict[str, float]] = {}
        for key, row in scored.items():
            if key[0] != state_id:
                continue
            by_candidate.setdefault(key[1], []).append(float(row["score"]))
            per_seed.setdefault(str(key[1]), {})[str(key[2])] = float(row["score"])
        means = {index: float(np.mean(values)) for index, values in by_candidate.items()}
        scores_by_seed = {
            int(seed): {
                candidate_index: float(per_seed[str(candidate_index)][str(seed)])
                for candidate_index in sorted(by_candidate)
            }
            for seed in cfg.world_model.seeds
        }
        primary_seed = int(cfg.world_model.seeds[0])
        label_scores = means if cfg.preferences.label_aggregation == "mean" else scores_by_seed[primary_seed]
        top, bottom = extreme_candidates(label_scores)
        pairs = build_dccp_labels(
            label_scores,
            margin_pos=cfg.preferences.margin_pos,
            margin_neg=cfg.preferences.margin_neg,
            max_pairs_per_state=cfg.preferences.max_pairs_per_state,
        )
        pairs_by_seed = {
            str(seed): build_dccp_labels(
                seed_scores,
                margin_pos=cfg.preferences.margin_pos,
                margin_neg=cfg.preferences.margin_neg,
                max_pairs_per_state=cfg.preferences.max_pairs_per_state,
            )
            for seed, seed_scores in scores_by_seed.items()
        }
        mean_top, mean_bottom = extreme_candidates(means)
        label_rows.append({
            "state_id": state_id,
            "label_aggregation": cfg.preferences.label_aggregation,
            "primary_wm_seed": primary_seed,
            "label_scores": {str(key): value for key, value in sorted(label_scores.items())},
            "candidate_mean_scores": {str(key): value for key, value in sorted(means.items())},
            "candidate_scores_by_wm_seed": per_seed,
            "predicted_top_candidates": top,
            "predicted_bottom_candidates": bottom,
            "mean_score_top_candidates": mean_top,
            "mean_score_bottom_candidates": mean_bottom,
            "dccp_emitted_pairs": pairs,
            "dccp_pairs_by_wm_seed": pairs_by_seed,
            "margin_pos": cfg.preferences.margin_pos,
            "margin_neg": cfg.preferences.margin_neg,
        })
    labels_path = store.labels / "labels.jsonl"
    AuditStore.write_jsonl(labels_path, label_rows)
    freeze_payload = {
        "schema_version": 1,
        "labels_sha256": sha256_file(labels_path),
        "prediction_scores_sha256": sha256_file(score_path),
        "config_fingerprint": stable_hash({
            "margin_pos": cfg.preferences.margin_pos,
            "margin_neg": cfg.preferences.margin_neg,
            "max_pairs_per_state": cfg.preferences.max_pairs_per_state,
            "label_aggregation": cfg.preferences.label_aggregation,
            "wm_seeds": cfg.world_model.seeds,
            "wm_mode": cfg.world_model.mode,
        }),
        "num_states": len(label_rows),
        "num_pairs": sum(len(row["dccp_emitted_pairs"]) for row in label_rows),
        "protocol": "frozen_before_simulator_ground_truth",
    }
    AuditStore.write_json(frozen_path, freeze_payload)
    return label_rows


def verify_frozen_labels(cfg: AuditConfig) -> dict:
    store = AuditStore(cfg.run_dir)
    frozen = store.labels / "LABELS_FROZEN.json"
    labels = store.labels / "labels.jsonl"
    if not frozen.exists() or not labels.exists():
        raise RuntimeError("DCCP labels are not frozen; run score_and_freeze first")
    payload = json.loads(frozen.read_text(encoding="utf-8"))
    if payload["labels_sha256"] != sha256_file(labels):
        raise RuntimeError("Frozen labels were modified after freezing")
    return payload
