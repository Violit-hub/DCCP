"""审计产物使用的稳定数据结构。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class NominalStepRecord:
    trajectory_id: str
    decision_index: int
    low_level_step: int
    frame_path: str
    simulator_state_path: str
    nominal_action_path: str
    nominal_tokens_path: str
    entropy: float
    progress: float | None = None
    curvature: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SelectionRecord:
    trajectory_id: str
    method: str
    selected_indices: list[int]
    scores: list[float]
    selection_seed: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BranchResult:
    trajectory_id: str
    decision_index: int
    candidate_index: int
    is_nominal: bool
    evaluation_seed: int
    success: bool
    finish_low_level_step: int
    executed_low_level_steps: int
    candidate_tokens_path: str
    candidate_action_path: str
    video_path: str | None = None
    status: str = "OK"
    error: str | None = None

    @property
    def key(self) -> tuple[str, int, int, int]:
        return (
            self.trajectory_id,
            self.decision_index,
            self.candidate_index,
            self.evaluation_seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
