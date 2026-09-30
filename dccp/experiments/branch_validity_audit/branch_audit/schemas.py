"""Stable records written by the branch-validity stages."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class StateRecord:
    state_id: str
    trajectory_id: str
    decision_index: int
    low_level_step: int
    frame_path: str
    simulator_state_path: str
    model_xml_path: str
    nominal_action_path: str
    nominal_tokens_path: str
    source_paths: dict[str, str] = field(default_factory=dict)
    hashes: dict[str, str] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class CandidateRecord:
    state_id: str
    candidate_index: int
    is_nominal: bool
    action_path: str
    seed: int
    token_hash: str

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class PredictionRecord:
    state_id: str
    candidate_index: int
    wm_seed: int
    noise_seed: int
    mode: str
    frames_path: str | None
    video_path: str | None
    status: str = "OK"
    error: str | None = None

    @property
    def key(self):
        return self.state_id, self.candidate_index, self.wm_seed

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class SimulatorOutcome:
    state_id: str
    candidate_index: int
    evaluation_seed: int
    success: bool
    short_progress: float
    final_progress: float
    short_metrics: dict[str, bool]
    final_metrics: dict[str, bool]
    executed_low_level_steps: int
    finish_low_level_step: int
    failure_reason: str
    video_path: str | None
    status: str = "OK"
    error: str | None = None

    @property
    def key(self):
        return self.state_id, self.candidate_index, self.evaluation_seed

    def to_dict(self):
        return asdict(self)
