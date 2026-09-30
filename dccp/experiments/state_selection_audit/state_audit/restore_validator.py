"""实验开跑前验证 robosuite 状态恢复的一致性。"""

from __future__ import annotations

from typing import Any

import numpy as np

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .policy_adapter import FrozenVLAAdapter
from .seeding import derive_seed, seed_everything
from .simulator_adapter import CoffeeSimulatorAdapter, load_initial_states


def validate_state_restore(cfg: AuditConfig) -> dict[str, Any]:
    """同一快照和动作在两个全新环境中必须得到相同观测与结果。"""
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    states = load_initial_states(
        cfg.paths.initial_states,
        trust_pickle=cfg.pilot.trust_initial_states_pickle,
    )
    simulator = CoffeeSimulatorAdapter(cfg.paths.dataset_config, cfg.simulator)
    policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
    policy.load()
    seed = derive_seed(cfg.run_name, "restore_gate")
    seed_everything(seed)

    source_env = simulator.create_env()
    try:
        obs = source_env.reset_to(states[0])
        obs = simulator.warm_up(source_env, obs)
        first = policy.generate_action(
            simulator.observation_image(obs, cfg.simulator.camera_key),
            cfg.policy.instruction,
            seed=derive_seed(seed, "reach_snapshot"),
            do_sample=False,
            temperature=1.0,
            compute_entropy=False,
        )
        obs, reached_success, _ = simulator.step_action_chunk(
            source_env, first.actions, cfg.simulator.max_low_level_steps
        )
        if reached_success:
            raise RuntimeError("恢复闸门的首个动作已完成任务，无法构造中间状态")
        model_xml, simulator_state = simulator.snapshot(source_env)
        snapshot_image = simulator.observation_image(obs, cfg.simulator.camera_key)
        probe = policy.generate_action(
            snapshot_image,
            cfg.policy.instruction,
            seed=derive_seed(seed, "probe_action"),
            do_sample=False,
            temperature=1.0,
            compute_entropy=False,
        )
    finally:
        simulator.close_env(source_env)

    trials = []
    for repeat in range(2):
        seed_everything(derive_seed(seed, "repeat", repeat))
        env = simulator.create_env()
        try:
            restored_obs = simulator.restore(env, model_xml, simulator_state)
            restored_image = simulator.observation_image(restored_obs, cfg.simulator.camera_key)
            next_obs, success, executed = simulator.step_action_chunk(
                env, probe.actions, cfg.policy.action_chunk_length
            )
            next_image = simulator.observation_image(next_obs, cfg.simulator.camera_key)
            _, next_state = simulator.snapshot(env)
            trials.append(
                {
                    "restored_image": restored_image,
                    "next_image": next_image,
                    "success": bool(success),
                    "executed": int(executed),
                    "next_state": next_state,
                }
            )
        finally:
            simulator.close_env(env)

    atol = float(cfg.simulator.image_checksum_atol)
    # 首次创建离屏 renderer 时可能有极少数像素初始化差异；真正的闸门是
    # 两个全新环境从同一 state 恢复后彼此一致，并产生相同物理转移。
    restore_pair_match = np.allclose(
        trials[0]["restored_image"], trials[1]["restored_image"], atol=atol, rtol=0.0
    )
    transition_match = np.allclose(
        trials[0]["next_image"], trials[1]["next_image"], atol=atol, rtol=0.0
    )
    scalar_match = (
        trials[0]["success"] == trials[1]["success"]
        and trials[0]["executed"] == trials[1]["executed"]
    )
    state_diff = float(np.max(np.abs(trials[0]["next_state"] - trials[1]["next_state"])))
    state_match = state_diff <= float(cfg.simulator.simulator_state_atol)
    passed = bool(restore_pair_match and transition_match and scalar_match and state_match)
    report = {
        "passed": passed,
        "seed": seed,
        "restore_pair_image_match": bool(restore_pair_match),
        "transition_image_match": bool(transition_match),
        "transition_state_match": bool(state_match),
        "scalar_match": bool(scalar_match),
        "source_to_restore_mean_abs_diff": float(
            np.mean(np.abs(trials[0]["restored_image"].astype(float) - snapshot_image))
        ),
        "source_to_restore_max_abs_diff": float(
            np.max(np.abs(trials[0]["restored_image"].astype(float) - snapshot_image))
        ),
        "max_restore_pair_abs_diff": float(
            np.max(np.abs(trials[0]["restored_image"].astype(float) - trials[1]["restored_image"].astype(float)))
        ),
        "max_transition_abs_diff": float(
            np.max(np.abs(trials[0]["next_image"].astype(float) - trials[1]["next_image"].astype(float)))
        ),
        "max_transition_state_abs_diff": state_diff,
    }
    ArtifactStore.write_json_atomic(store.report_dir / "state_restore_gate.json", report)
    if not passed:
        raise RuntimeError(f"状态恢复一致性闸门失败: {report}")
    return report
