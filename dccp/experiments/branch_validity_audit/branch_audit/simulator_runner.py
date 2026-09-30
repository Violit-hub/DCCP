"""Execute frozen candidate actions from identical simulator snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .artifacts import AuditStore, relative
from .candidates import candidates_by_state
from .coffee_oracle import failure_reason, partial_metrics, progress_score
from .config import AuditConfig
from .lrm_runner import verify_frozen_labels
from .schemas import SimulatorOutcome
from .seeding import derive_seed, seed_everything
from .source_states import load_states
from .video import StreamingVideoRecorder


@dataclass(frozen=True)
class SimulatorScope:
    state_id: str | None = None
    candidate_index: int | None = None
    evaluation_seed: int | None = None
    max_rollouts: int | None = None
    rerun_completed: bool = False
    save_video: bool | None = None

    def matches(self, state_id: str, candidate_index: int, evaluation_seed: int) -> bool:
        return (
            (self.state_id is None or self.state_id == state_id)
            and (self.candidate_index is None or self.candidate_index == candidate_index)
            and (self.evaluation_seed is None or self.evaluation_seed == evaluation_seed)
        )


def _load_action(store: AuditStore, candidate: dict) -> np.ndarray:
    with np.load(store.run_dir / candidate["action_path"], allow_pickle=False) as data:
        return np.asarray(data["actions"], dtype=np.float32)


def _evaluate_one(cfg, store, simulator, policy, state, candidate, evaluation_seed, save_video):
    state_id = str(state["state_id"])
    candidate_index = int(candidate["candidate_index"])
    seed_everything(evaluation_seed)
    env = simulator.create_env()
    recorder = None
    video_path = None
    try:
        xml = (store.run_dir / state["model_xml_path"]).read_text(encoding="utf-8")
        with np.load(store.run_dir / state["simulator_state_path"], allow_pickle=False) as data:
            simulator_state = np.asarray(data["states"])
        obs = simulator.restore(env, xml, simulator_state)
        if save_video:
            video_path = (
                store.simulator / state_id / f"candidate_{candidate_index:02d}"
                / f"sim_seed_{evaluation_seed}.mp4"
            )
            recorder = StreamingVideoRecorder(video_path, fps=cfg.simulator.video_fps)
            recorder.append(simulator.observation_image(obs, cfg.simulator.camera_key))
        remaining = int(cfg.simulator.max_low_level_steps) - int(state["low_level_step"])
        if remaining <= 0:
            raise ValueError(f"State {state_id} is already past the episode horizon")
        executed_total = 0
        success = False
        trace = [partial_metrics(env)]
        action = _load_action(store, candidate)
        chunk_index = 0
        short_metrics = None
        while executed_total < remaining and not success:
            if chunk_index > 0:
                image = simulator.observation_image(obs, cfg.simulator.camera_key)
                continuation_seed = derive_seed(
                    cfg.run_name, state_id, evaluation_seed, "sim_continuation", chunk_index
                )
                followup = policy.generate_action(
                    image, cfg.policy.instruction, seed=continuation_seed,
                    do_sample=cfg.policy.nominal_do_sample,
                    temperature=cfg.policy.nominal_temperature, compute_entropy=False,
                )
                action = followup.actions
            obs, success, executed = simulator.step_action_chunk(
                env, action, remaining - executed_total,
                frame_callback=recorder.append if recorder is not None else None,
            )
            executed_total += int(executed)
            chunk_index += 1
            metrics = partial_metrics(env)
            trace.append(metrics)
            success = bool(success or metrics["task"])
            if chunk_index == int(cfg.world_model.horizon_H) or success:
                short_metrics = dict(metrics)
            if executed == 0:
                raise RuntimeError("Simulator executed zero actions in a continuation chunk")
        final_metrics = partial_metrics(env)
        if short_metrics is None:
            short_metrics = dict(final_metrics)
        if recorder is not None:
            recorder.close(True)
            recorder = None
        return SimulatorOutcome(
            state_id=state_id,
            candidate_index=candidate_index,
            evaluation_seed=int(evaluation_seed),
            success=bool(success or final_metrics["task"]),
            short_progress=progress_score(short_metrics),
            final_progress=progress_score(final_metrics),
            short_metrics=short_metrics,
            final_metrics=final_metrics,
            executed_low_level_steps=executed_total,
            finish_low_level_step=int(state["low_level_step"]) + executed_total,
            failure_reason=failure_reason(final_metrics, trace, success),
            video_path=relative(video_path, store.run_dir) if video_path else None,
        )
    finally:
        if recorder is not None:
            recorder.close(False)
        simulator.close_env(env)


def evaluate_in_simulator(
    cfg: AuditConfig, scope: SimulatorScope | None = None, *, policy=None, simulator=None
) -> list[dict]:
    from state_audit.policy_adapter import FrozenVLAAdapter
    from state_audit.simulator_adapter import CoffeeSimulatorAdapter

    verify_frozen_labels(cfg)
    scope = scope or SimulatorScope()
    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    states = load_states(cfg)
    candidate_map = candidates_by_state(cfg)
    result_path = store.simulator / "outcomes.jsonl"
    completed = {
        (str(row["state_id"]), int(row["candidate_index"]), int(row["evaluation_seed"]))
        for row in AuditStore.read_jsonl(result_path) if row.get("status") == "OK"
    }
    if policy is None:
        policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
        policy.load()
    if simulator is None:
        simulator = CoffeeSimulatorAdapter(cfg.paths.dataset_config, cfg.simulator)
    attempted = 0
    save_video = cfg.simulator.save_video if scope.save_video is None else scope.save_video
    for state in states:
        state_id = str(state["state_id"])
        for candidate in candidate_map[state_id]:
            candidate_index = int(candidate["candidate_index"])
            for evaluation_seed in cfg.simulator.evaluation_seeds:
                key = state_id, candidate_index, int(evaluation_seed)
                if not scope.matches(*key):
                    continue
                if cfg.runtime.resume and key in completed and not scope.rerun_completed:
                    continue
                if scope.max_rollouts is not None and attempted >= scope.max_rollouts:
                    return AuditStore.read_jsonl(result_path)
                attempted += 1
                try:
                    result = _evaluate_one(
                        cfg, store, simulator, policy, state, candidate,
                        int(evaluation_seed), bool(save_video),
                    )
                except Exception as exc:
                    result = SimulatorOutcome(
                        state_id, candidate_index, int(evaluation_seed), False, 0.0, 0.0,
                        {}, {}, 0, int(state["low_level_step"]), "evaluation_error", None,
                        "FAILED", f"{type(exc).__name__}: {exc}",
                    )
                AuditStore.append_jsonl(result_path, [result.to_dict()])
                if result.status == "OK":
                    completed.add(key)
                print(
                    f"[sim] {key} status={result.status} success={result.success} "
                    f"short={result.short_progress:.2f} video={result.video_path}", flush=True,
                )
    return AuditStore.read_jsonl(result_path)
