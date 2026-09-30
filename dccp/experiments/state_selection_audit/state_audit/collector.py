"""采集冻结策略的 nominal coffee 轨迹与可恢复状态。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .policy_adapter import FrozenVLAAdapter
from .schemas import NominalStepRecord
from .seeding import derive_seed, seed_everything
from .simulator_adapter import CoffeeSimulatorAdapter, load_initial_states
from .stage_gates import require_restore_gate


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _save_png_atomic(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(temporary, format="PNG")
    temporary.replace(path)
    path.chmod(0o660)


def _git_metadata(root: str | Path) -> dict[str, Any]:
    root = str(Path(root).resolve())

    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", root, *args], check=False, capture_output=True, text=True
        )
        return result.stdout.strip()

    return {
        "commit": run("rev-parse", "HEAD"),
        "status_short": run("status", "--short"),
    }


def write_run_manifest(cfg: AuditConfig, config_path: str | Path, store: ArtifactStore) -> None:
    source = Path(config_path).resolve()
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(source),
        "config_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "task": cfg.task,
        "run_name": cfg.run_name,
        "git": _git_metadata(cfg.paths.dccp_root),
        "scientific_contract": {
            "decision_unit": "one_8x7_action_chunk",
            "candidate_zero_is_nominal": True,
            "branch_backend": "real_simulator",
            "selector_blind_to_branch_results": True,
        },
    }
    ArtifactStore.write_json_atomic(store.run_dir / "manifest.json", payload)


def collect_nominal(cfg: AuditConfig, config_path: str | Path) -> list[dict[str, Any]]:
    """逐条采集轨迹；已经完整写出的轨迹会在 resume 模式下跳过。"""
    cfg.validate(require_inputs=True)
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    require_restore_gate(cfg)
    write_run_manifest(cfg, config_path, store)

    states = load_initial_states(
        cfg.paths.initial_states,
        trust_pickle=cfg.pilot.trust_initial_states_pickle,
    )
    if len(states) < cfg.pilot.num_trajectories:
        raise ValueError(
            f"初始状态只有 {len(states)} 个，小于 num_trajectories={cfg.pilot.num_trajectories}"
        )

    policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
    policy.load()
    simulator = CoffeeSimulatorAdapter(cfg.paths.dataset_config, cfg.simulator)
    summaries: list[dict[str, Any]] = []

    for trajectory_number in range(cfg.pilot.num_trajectories):
        trajectory_id = f"traj_{trajectory_number:04d}"
        trajectory_dir = store.nominal_dir / trajectory_id
        summary_path = trajectory_dir / "trajectory.json"
        if cfg.pilot.resume and summary_path.exists():
            print(f"[collect] {trajectory_id}: 已完成，跳过", flush=True)
            summaries.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue

        print(f"[collect] {trajectory_id}: 开始", flush=True)

        env = simulator.create_env()
        try:
            seed = cfg.pilot.nominal_seeds[
                trajectory_number % len(cfg.pilot.nominal_seeds)
            ]
            seed_everything(seed)
            initial_state = states[trajectory_number]
            obs = env.reset_to(initial_state) if initial_state is not None else env.reset()
            obs = simulator.warm_up(env, obs)
            low_level_step = 0
            decision_index = 0
            success = False
            records: list[dict[str, Any]] = []
            model_xml_path = trajectory_dir / "model.xml"

            while low_level_step < cfg.simulator.max_low_level_steps and not success:
                frame = simulator.observation_image(obs, cfg.simulator.camera_key)
                model_xml, simulator_state = simulator.snapshot(env)
                if decision_index == 0:
                    ArtifactStore.write_text_atomic(model_xml_path, model_xml)

                frame_path = trajectory_dir / "frames" / f"step_{decision_index:04d}.png"
                state_path = trajectory_dir / "states" / f"step_{decision_index:04d}.npz"
                action_path = trajectory_dir / "actions" / f"step_{decision_index:04d}.npz"
                token_path = trajectory_dir / "tokens" / f"step_{decision_index:04d}.npz"
                _save_png_atomic(frame_path, frame)
                ArtifactStore.write_npz_atomic(state_path, states=simulator_state)

                action_seed = derive_seed(cfg.run_name, trajectory_id, decision_index, seed)
                policy_action = policy.generate_action(
                    frame,
                    cfg.policy.instruction,
                    seed=action_seed,
                    do_sample=cfg.policy.nominal_do_sample,
                    temperature=cfg.policy.nominal_temperature,
                )
                ArtifactStore.write_npz_atomic(
                    action_path,
                    actions=policy_action.actions,
                    normalized_actions=policy_action.normalized_actions,
                    seed=np.asarray(policy_action.seed, dtype=np.int64),
                )
                ArtifactStore.write_npz_atomic(
                    token_path, response_tokens=policy_action.response_tokens
                )
                record = NominalStepRecord(
                    trajectory_id=trajectory_id,
                    decision_index=decision_index,
                    low_level_step=low_level_step,
                    frame_path=_relative(frame_path, store.run_dir),
                    simulator_state_path=_relative(state_path, store.run_dir),
                    nominal_action_path=_relative(action_path, store.run_dir),
                    nominal_tokens_path=_relative(token_path, store.run_dir),
                    entropy=policy_action.entropy,
                    metadata={"action_seed": action_seed},
                )
                records.append(record.to_dict())

                remaining = cfg.simulator.max_low_level_steps - low_level_step
                obs, success, executed = simulator.step_action_chunk(
                    env, policy_action.actions, remaining
                )
                low_level_step += executed
                decision_index += 1
                if executed == 0:
                    raise RuntimeError("模拟器没有执行任何 low-level action")

            summary = {
                "trajectory_id": trajectory_id,
                "initial_state_index": trajectory_number,
                "nominal_seed": seed,
                "success": bool(success),
                "finish_low_level_step": int(low_level_step),
                "num_decisions": len(records),
                "model_xml_path": _relative(model_xml_path, store.run_dir),
                "steps": records,
            }
            ArtifactStore.write_json_atomic(summary_path, summary)
            summaries.append(summary)
            print(
                f"[collect] {trajectory_id}: 完成 decisions={len(records)}, "
                f"success={bool(success)}",
                flush=True,
            )
        finally:
            simulator.close_env(env)

    ArtifactStore.write_json_atomic(store.nominal_dir / "trajectories.json", summaries)
    return summaries
