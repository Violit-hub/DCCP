"""Strict configuration schema for the standalone branch-validity audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    dccp_root: str
    source_state_run: str
    source_state_config: str
    policy_checkpoint: str
    dataset_config: str
    progress_lrm_config: str
    wm_inference_config: str
    output_root: str
    mujoco_path: str = "/path/to/mujoco210"


@dataclass(frozen=True)
class StateConfig:
    selection_method: str = "joint"
    max_states: int = 2
    # Empty means use selection_method from the source state audit.
    targets: list[dict[str, Any]] = field(default_factory=list)


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
class CandidateConfig:
    total_candidates: int = 8
    seed: int = 1701
    # True is easier to inspect; false is closer to the current training sampler.
    deduplicate: bool = False


@dataclass(frozen=True)
class WorldModelConfig:
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    horizon_H: int = 3
    queue_len: int = 4
    seeds: list[int] = field(default_factory=lambda: [401, 402, 403])
    mode: str = "controlled"
    fps: int = 12
    save_npz: bool = True
    save_video: bool = True


@dataclass(frozen=True)
class PreferenceConfig:
    margin_pos: float = 0.10
    margin_neg: float = 0.10
    max_pairs_per_state: int | None = None
    label_aggregation: str = "primary_seed"


@dataclass(frozen=True)
class SimulatorConfig:
    num_steps_wait: int = 10
    max_low_level_steps: int = 256
    success_reward_threshold: float = 0.0
    camera_key: str = "agentview_image"
    evaluation_seeds: list[int] = field(default_factory=lambda: [101, 202, 303])
    save_video: bool = True
    video_fps: int = 20


@dataclass(frozen=True)
class MetricsConfig:
    bootstrap_samples: int = 2000
    bootstrap_seed: int = 90210
    actual_gap_epsilon: float = 1e-9


@dataclass(frozen=True)
class RuntimeConfig:
    gpu_memory_threshold_mib: int = 4096
    skip_gpu_stages_when_busy: bool = True
    progress_health_timeout_sec: float = 5.0
    resume: bool = True


@dataclass(frozen=True)
class AuditConfig:
    task: str
    run_name: str
    paths: PathConfig
    states: StateConfig = field(default_factory=StateConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    preferences: PreferenceConfig = field(default_factory=PreferenceConfig)
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    @property
    def run_dir(self) -> Path:
        return Path(self.paths.output_root).expanduser().resolve() / self.run_name

    def validate(self, *, require_inputs: bool = False) -> None:
        if self.task != "coffee":
            raise ValueError(f"This pilot currently supports coffee only, got {self.task!r}")
        if self.states.max_states <= 0:
            raise ValueError("states.max_states must be positive")
        if self.candidates.total_candidates < 2:
            raise ValueError("candidates.total_candidates must include nominal and be >= 2")
        if self.world_model.horizon_H <= 0 or self.world_model.queue_len <= 0:
            raise ValueError("world_model horizon_H and queue_len must be positive")
        if not self.world_model.seeds:
            raise ValueError("world_model.seeds cannot be empty")
        if self.world_model.mode not in {"controlled", "faithful"}:
            raise ValueError("world_model.mode must be controlled or faithful")
        if not self.simulator.evaluation_seeds:
            raise ValueError("simulator.evaluation_seeds cannot be empty")
        if self.preferences.margin_pos < 0 or self.preferences.margin_neg < 0:
            raise ValueError("preference margins must be non-negative")
        if self.preferences.label_aggregation not in {"primary_seed", "mean"}:
            raise ValueError("preferences.label_aggregation must be primary_seed or mean")
        if self.policy.action_chunk_length <= 0 or self.policy.action_dim <= 0:
            raise ValueError("policy action shape must be positive")
        if require_inputs:
            checks = {
                "dccp_root": self.paths.dccp_root,
                "source_state_run": self.paths.source_state_run,
                "source_state_config": self.paths.source_state_config,
                "policy_checkpoint": self.paths.policy_checkpoint,
                "dataset_config": self.paths.dataset_config,
                "progress_lrm_config": self.paths.progress_lrm_config,
                "wm_inference_config": self.paths.wm_inference_config,
                "mujoco_path": self.paths.mujoco_path,
            }
            missing = [f"{name}={value}" for name, value in checks.items() if not Path(value).expanduser().exists()]
            if missing:
                raise FileNotFoundError("Configured inputs do not exist:\n  " + "\n  ".join(missing))


def _make(cls, value: dict[str, Any] | None):
    return cls(**dict(value or {}))


def load_config(path: str | Path, *, require_inputs: bool = False) -> AuditConfig:
    source = Path(path).expanduser().resolve()
    raw = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    for key in ("task", "run_name", "paths"):
        if key not in raw:
            raise ValueError(f"Missing configuration field: {key}")
    cfg = AuditConfig(
        task=str(raw["task"]),
        run_name=str(raw["run_name"]),
        paths=_make(PathConfig, raw["paths"]),
        states=_make(StateConfig, raw.get("states")),
        policy=_make(PolicyConfig, raw.get("policy")),
        candidates=_make(CandidateConfig, raw.get("candidates")),
        world_model=_make(WorldModelConfig, raw.get("world_model")),
        preferences=_make(PreferenceConfig, raw.get("preferences")),
        simulator=_make(SimulatorConfig, raw.get("simulator")),
        metrics=_make(MetricsConfig, raw.get("metrics")),
        runtime=_make(RuntimeConfig, raw.get("runtime")),
    )
    cfg.validate(require_inputs=require_inputs)
    return cfg
