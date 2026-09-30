"""Resumable state × candidate × world-model-seed prediction stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .artifacts import AuditStore, relative
from .candidates import candidates_by_state
from .config import AuditConfig
from .schemas import PredictionRecord
from .seeding import derive_seed
from .source_states import load_states
from .video import write_video
from .world_model_adapter import StandaloneWorldModel


@dataclass(frozen=True)
class PredictionScope:
    state_id: str | None = None
    candidate_index: int | None = None
    wm_seed: int | None = None
    max_predictions: int | None = None
    rerun_completed: bool = False

    def matches(self, state_id: str, candidate_index: int, wm_seed: int) -> bool:
        return (
            (self.state_id is None or self.state_id == state_id)
            and (self.candidate_index is None or self.candidate_index == candidate_index)
            and (self.wm_seed is None or self.wm_seed == wm_seed)
        )


def predict_world_model(cfg: AuditConfig, scope: PredictionScope | None = None) -> list[dict]:
    from state_audit.policy_adapter import FrozenVLAAdapter

    scope = scope or PredictionScope()
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    states = load_states(cfg)
    candidate_map = candidates_by_state(cfg)
    result_path = store.world_model / "predictions.jsonl"
    completed = {
        (str(row["state_id"]), int(row["candidate_index"]), int(row["wm_seed"]))
        for row in AuditStore.read_jsonl(result_path) if row.get("status") == "OK"
    }
    policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
    policy.load()
    model = StandaloneWorldModel(cfg, policy)
    model.load()
    attempted = 0
    for state in states:
        state_id = str(state["state_id"])
        frame = np.asarray(Image.open(store.run_dir / state["frame_path"]).convert("RGB"))
        for candidate in candidate_map[state_id]:
            candidate_index = int(candidate["candidate_index"])
            with np.load(store.run_dir / candidate["action_path"], allow_pickle=False) as data:
                normalized_actions = np.asarray(data["normalized_actions"], dtype=np.float32)
            for wm_seed in cfg.world_model.seeds:
                if not scope.matches(state_id, candidate_index, int(wm_seed)):
                    continue
                key = state_id, candidate_index, int(wm_seed)
                if cfg.runtime.resume and key in completed and not scope.rerun_completed:
                    continue
                if scope.max_predictions is not None and attempted >= scope.max_predictions:
                    return AuditStore.read_jsonl(result_path)
                attempted += 1
                # controlled: same noise for candidates; faithful: training-like independent draws.
                noise_seed = (
                    derive_seed(cfg.run_name, state_id, wm_seed, "controlled")
                    if cfg.world_model.mode == "controlled"
                    else derive_seed(cfg.run_name, state_id, candidate_index, wm_seed, "faithful")
                )
                target_dir = store.world_model / state_id / f"candidate_{candidate_index:02d}"
                frames_path = target_dir / f"wm_seed_{wm_seed}.npz"
                video_path = target_dir / f"wm_seed_{wm_seed}.mp4"
                try:
                    frames = model.predict_branch(
                        frame, normalized_actions, instruction=cfg.policy.instruction,
                        noise_seed=noise_seed,
                        continuation_seed=derive_seed(cfg.run_name, state_id, wm_seed, "continuation"),
                    )
                    if cfg.world_model.save_npz:
                        AuditStore.write_npz(frames_path, frames=frames)
                    if cfg.world_model.save_video:
                        write_video(video_path, frames, fps=cfg.world_model.fps)
                    record = PredictionRecord(
                        state_id, candidate_index, int(wm_seed), noise_seed,
                        cfg.world_model.mode,
                        relative(frames_path, store.run_dir) if cfg.world_model.save_npz else None,
                        relative(video_path, store.run_dir) if cfg.world_model.save_video else None,
                    )
                except Exception as exc:
                    record = PredictionRecord(
                        state_id, candidate_index, int(wm_seed), noise_seed,
                        cfg.world_model.mode, None, None, "FAILED", f"{type(exc).__name__}: {exc}",
                    )
                AuditStore.append_jsonl(result_path, [record.to_dict()])
                if record.status == "OK":
                    completed.add(key)
                print(f"[wm] {key} status={record.status} video={record.video_path}", flush=True)
    return AuditStore.read_jsonl(result_path)
