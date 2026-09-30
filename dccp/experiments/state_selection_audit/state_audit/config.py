"""实验配置读取与严格校验。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    dccp_root: str
    policy_checkpoint: str
    dataset_config: str
    initial_states: str
    progress_lrm_config: str
    output_root: str
    mujoco_path: str = "/path/to/mujoco210"


@dataclass(frozen=True)
class PolicyConfig:
    unnorm_key: str = "coffee_d0_300_demos"
    instruction: str = "coffee"
    dtype: str = "bfloat16"
    device: str = "cuda:0"
    center_crop: bool = True
    action_chunk_length: int = 8
    action_dim: int = 7
    nominal_do_sample: bool = True
    nominal_temperature: float = 1.6
    candidate_temperature: float = 1.0
    candidate_max_attempts: int = 64


@dataclass(frozen=True)
class SimulatorConfig:
    num_steps_wait: int = 10
    max_low_level_steps: int = 256
    success_reward_threshold: float = 0.0
    camera_key: str = "agentview_image"
    image_checksum_atol: float = 0.0
    simulator_state_atol: float = 1e-10


@dataclass(frozen=True)
class SelectionConfig:
    horizon_H: int = 3
    state_budget: int = 2
    nms_gap: int = 2
    lambda_curvature: float = 1.0
    lambda_entropy: float = 1.0
    random_draws: int = 1000


@dataclass(frozen=True)
class PilotConfig:
    num_trajectories: int = 5
    total_candidates: int = 8
    candidate_seed: int = 1701
    nominal_seeds: list[int] = field(default_factory=lambda: [11, 23, 37, 53, 71])
    evaluation_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])
    bootstrap_samples: int = 2000
    trust_initial_states_pickle: bool = False
    resume: bool = True


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_memory_threshold_mib: int = 4096
    allow_existing_gpu_processes_below_threshold: bool = False
    skip_gpu_tests_when_busy: bool = True
    progress_health_timeout_sec: float = 5.0


@dataclass(frozen=True)
class AuditConfig:
    task: str
    run_name: str
    paths: PathConfig
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    pilot: PilotConfig = field(default_factory=PilotConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.output_root).expanduser().resolve() / self.run_name

    def validate(self, *, require_inputs: bool = False) -> None:
        if self.task != "coffee":
            raise ValueError(f"小规模版本当前只支持 coffee，收到: {self.task!r}")
        if self.policy.action_chunk_length <= 0 or self.policy.action_dim <= 0:
            raise ValueError("action_chunk_length 和 action_dim 必须为正数")
        if self.pilot.total_candidates < 2:
            raise ValueError("total_candidates 至少为 2（包含 nominal）")
        if self.pilot.num_trajectories <= 0:
            raise ValueError("num_trajectories 必须为正数")
        if not self.pilot.nominal_seeds:
            raise ValueError("nominal_seeds 不能为空")
        if not self.pilot.evaluation_seeds:
            raise ValueError("evaluation_seeds 不能为空")
        if self.selection.state_budget <= 0:
            raise ValueError("state_budget 必须为正数")
        if self.selection.random_draws <= 0:
            raise ValueError("random_draws 必须为正数")
        if self.selection.horizon_H <= 0 or self.selection.nms_gap < 0:
            raise ValueError("horizon_H 必须为正数，nms_gap 不能为负数")
        if self.simulator.max_low_level_steps < self.policy.action_chunk_length:
            raise ValueError("max_low_level_steps 不能小于一个动作块")
        if require_inputs:
            for label, value in {
                "dccp_root": self.paths.dccp_root,
                "policy_checkpoint": self.paths.policy_checkpoint,
                "dataset_config": self.paths.dataset_config,
                "initial_states": self.paths.initial_states,
                "progress_lrm_config": self.paths.progress_lrm_config,
                "mujoco_path": self.paths.mujoco_path,
            }.items():
                if not Path(value).expanduser().exists():
                    raise FileNotFoundError(f"配置路径不存在: {label}={value}")


def _build_dataclass(cls, values: dict[str, Any] | None):
    return cls(**dict(values or {}))


def load_config(path: str | Path, *, require_inputs: bool = False) -> AuditConfig:
    """从 YAML 加载配置，并拒绝缺少核心字段的配置。"""
    config_path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for key in ("task", "run_name", "paths"):
        if key not in raw:
            raise ValueError(f"配置缺少字段: {key}")
    cfg = AuditConfig(
        task=str(raw["task"]),
        run_name=str(raw["run_name"]),
        paths=_build_dataclass(PathConfig, raw["paths"]),
        policy=_build_dataclass(PolicyConfig, raw.get("policy")),
        simulator=_build_dataclass(SimulatorConfig, raw.get("simulator")),
        selection=_build_dataclass(SelectionConfig, raw.get("selection")),
        pilot=_build_dataclass(PilotConfig, raw.get("pilot")),
        runtime=_build_dataclass(RuntimeConfig, raw.get("runtime")),
    )
    cfg.validate(require_inputs=require_inputs)
    return cfg
