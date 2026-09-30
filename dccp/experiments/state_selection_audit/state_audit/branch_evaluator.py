"""从保存状态出发，在真实模拟器中穷举全部有效状态和候选首动作。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .policy_adapter import FrozenVLAAdapter, PolicyAction
from .schemas import BranchResult
from .seeding import derive_seed, seed_everything
from .simulator_adapter import CoffeeSimulatorAdapter
from .video import BranchVideoRecorder


@dataclass(frozen=True)
class EvaluationScope:
    """把长评估拆成可人工验收的单轨迹、单状态或单候选任务。"""

    trajectory_id: str | None = None
    decision_index: int | None = None
    candidate_index: int | None = None
    evaluation_seed: int | None = None
    max_branches: int | None = None
    save_video: bool = False
    video_fps: int = 20
    rerun_completed: bool = False

    def matches(
        self,
        trajectory_id: str,
        decision_index: int,
        candidate_index: int,
        evaluation_seed: int,
    ) -> bool:
        return (
            (self.trajectory_id is None or trajectory_id == self.trajectory_id)
            and (self.decision_index is None or decision_index == self.decision_index)
            and (self.candidate_index is None or candidate_index == self.candidate_index)
            and (self.evaluation_seed is None or evaluation_seed == self.evaluation_seed)
        )


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _load_candidates(
    cfg: AuditConfig,
    store: ArtifactStore,
    policy: FrozenVLAAdapter,
    trajectory_id: str,
    step: dict[str, Any],
) -> list[tuple[PolicyAction, Path]]:
    decision_index = int(step["decision_index"])
    candidate_dir = store.branch_dir / "candidate_sets" / trajectory_id / f"step_{decision_index:04d}"
    manifest_path = candidate_dir / "candidates.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest["total_candidates"]) != cfg.pilot.total_candidates:
            raise ValueError("已缓存候选数与当前配置不同，请使用新的 run_name")
        output = []
        for item in manifest["candidates"]:
            path = store.run_dir / item["path"]
            with np.load(path, allow_pickle=False) as data:
                output.append(
                    (
                        PolicyAction(
                            actions=data["actions"],
                            normalized_actions=data["normalized_actions"],
                            response_tokens=data["response_tokens"],
                            entropy=float("nan"),
                            seed=int(data["seed"]),
                        ),
                        path,
                    )
                )
        return output

    with np.load(store.run_dir / step["nominal_action_path"], allow_pickle=False) as action_data:
        nominal_actions = action_data["actions"]
        normalized_actions = action_data["normalized_actions"]
        nominal_seed = int(action_data["seed"])
    with np.load(store.run_dir / step["nominal_tokens_path"], allow_pickle=False) as token_data:
        nominal_tokens = token_data["response_tokens"]
    nominal = PolicyAction(
        actions=nominal_actions,
        normalized_actions=normalized_actions,
        response_tokens=nominal_tokens,
        entropy=float(step["entropy"]),
        seed=nominal_seed,
    )
    frame = np.asarray(Image.open(store.run_dir / step["frame_path"]).convert("RGB"))
    base_seed = derive_seed(cfg.pilot.candidate_seed, trajectory_id, decision_index)
    candidates = policy.sample_candidate_set(
        frame,
        cfg.policy.instruction,
        nominal,
        total_candidates=cfg.pilot.total_candidates,
        base_seed=base_seed,
    )
    manifest_rows = []
    output = []
    for candidate_index, candidate in enumerate(candidates):
        path = candidate_dir / f"candidate_{candidate_index:02d}.npz"
        ArtifactStore.write_npz_atomic(
            path,
            actions=candidate.actions,
            normalized_actions=candidate.normalized_actions,
            response_tokens=candidate.response_tokens,
            seed=np.asarray(candidate.seed, dtype=np.int64),
        )
        manifest_rows.append(
            {
                "candidate_index": candidate_index,
                "is_nominal": candidate_index == 0,
                "path": _relative(path, store.run_dir),
            }
        )
        output.append((candidate, path))
    ArtifactStore.write_json_atomic(
        manifest_path,
        {"total_candidates": len(candidates), "candidates": manifest_rows},
    )
    return output


def _evaluate_one(
    cfg: AuditConfig,
    simulator: CoffeeSimulatorAdapter,
    policy: FrozenVLAAdapter,
    store: ArtifactStore,
    trajectory: dict[str, Any],
    step: dict[str, Any],
    candidate_index: int,
    candidate: PolicyAction,
    candidate_path: Path,
    evaluation_seed: int,
    *,
    save_video: bool,
    video_fps: int,
) -> BranchResult:
    trajectory_id = str(trajectory["trajectory_id"])
    decision_index = int(step["decision_index"])
    start_step = int(step["low_level_step"])
    seed_everything(evaluation_seed)
    env = simulator.create_env()
    executed_total = 0
    success = False
    recorder = None
    video_path = None
    try:
        model_xml = (store.run_dir / trajectory["model_xml_path"]).read_text(encoding="utf-8")
        with np.load(store.run_dir / step["simulator_state_path"], allow_pickle=False) as data:
            simulator_state = data["states"]
        obs = simulator.restore(env, model_xml, simulator_state)
        if save_video:
            video_path = (
                store.branch_dir
                / "videos"
                / trajectory_id
                / f"step_{decision_index:04d}"
                / f"candidate_{candidate_index:02d}_seed_{evaluation_seed}.mp4"
            )
            recorder = BranchVideoRecorder(video_path, fps=video_fps)
            recorder.append(simulator.observation_image(obs, cfg.simulator.camera_key))
        remaining = cfg.simulator.max_low_level_steps - start_step
        obs, success, executed = simulator.step_action_chunk(
            env,
            candidate.actions,
            remaining,
            frame_callback=recorder.append if recorder is not None else None,
        )
        executed_total += executed

        continuation_index = 0
        while not success and executed_total < remaining:
            image = simulator.observation_image(obs, cfg.simulator.camera_key)
            # 同一状态、同一 evaluation seed 下，各候选共享完全相同的后续随机数序列。
            continuation_seed = derive_seed(
                cfg.run_name,
                trajectory_id,
                decision_index,
                evaluation_seed,
                "continuation",
                continuation_index,
            )
            followup = policy.generate_action(
                image,
                cfg.policy.instruction,
                seed=continuation_seed,
                do_sample=cfg.policy.nominal_do_sample,
                temperature=cfg.policy.nominal_temperature,
                compute_entropy=False,
            )
            obs, success, executed = simulator.step_action_chunk(
                env,
                followup.actions,
                remaining - executed_total,
                frame_callback=recorder.append if recorder is not None else None,
            )
            executed_total += executed
            continuation_index += 1
            if executed == 0:
                raise RuntimeError("continuation 没有执行任何 low-level action")

        relative_candidate = _relative(candidate_path, store.run_dir)
        relative_video = _relative(video_path, store.run_dir) if video_path else None
        if recorder is not None:
            recorder.close(commit=True)
            recorder = None
        return BranchResult(
            trajectory_id=trajectory_id,
            decision_index=decision_index,
            candidate_index=candidate_index,
            is_nominal=candidate_index == 0,
            evaluation_seed=evaluation_seed,
            success=bool(success),
            finish_low_level_step=start_step + executed_total,
            executed_low_level_steps=executed_total,
            candidate_tokens_path=relative_candidate,
            candidate_action_path=relative_candidate,
            video_path=relative_video,
        )
    finally:
        if recorder is not None:
            recorder.close(commit=False)
        simulator.close_env(env)


def evaluate_all_valid_states(
    cfg: AuditConfig,
    scope: EvaluationScope | None = None,
) -> list[dict[str, Any]]:
    """评估冻结选择域内的每个状态；JSONL 主键支持安全断点续跑。"""
    scope = scope or EvaluationScope()
    if scope.max_branches is not None and scope.max_branches <= 0:
        raise ValueError("max_branches 必须为正数")
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    frozen = store.selection_dir / "SELECTIONS_FROZEN.json"
    if not frozen.exists():
        raise RuntimeError("选择尚未冻结，请先运行 03_select_states.py")
    selection_rows = ArtifactStore.read_jsonl(store.selection_dir / "selected_states.jsonl")
    eligible_by_trajectory: dict[str, set[int]] = {}
    for row in selection_rows:
        eligible_by_trajectory.setdefault(str(row["trajectory_id"]), set()).update(
            int(index) for index in row["eligible_indices"]
        )
    trajectories = json.loads(
        (store.nominal_dir / "trajectories.json").read_text(encoding="utf-8")
    )
    trajectory_ids = {str(item["trajectory_id"]) for item in trajectories}
    if scope.trajectory_id is not None and scope.trajectory_id not in trajectory_ids:
        raise ValueError(
            f"trajectory_id={scope.trajectory_id!r} 不存在；可用值: {sorted(trajectory_ids)}"
        )
    if scope.candidate_index is not None and not (
        0 <= scope.candidate_index < cfg.pilot.total_candidates
    ):
        raise ValueError(
            f"candidate_index 必须在 [0, {cfg.pilot.total_candidates - 1}] 内"
        )
    if (
        scope.evaluation_seed is not None
        and scope.evaluation_seed not in cfg.pilot.evaluation_seeds
    ):
        raise ValueError(
            f"evaluation_seed={scope.evaluation_seed} 不在配置 seeds={cfg.pilot.evaluation_seeds} 中"
        )
    eligible_targets = [
        (trajectory_id, decision_index)
        for trajectory_id, indices in eligible_by_trajectory.items()
        for decision_index in sorted(indices)
        if (scope.trajectory_id is None or trajectory_id == scope.trajectory_id)
        and (scope.decision_index is None or decision_index == scope.decision_index)
    ]
    if not eligible_targets:
        raise ValueError(
            "没有状态匹配指定 trajectory/decision 过滤条件；"
            "请先运行 inspect 并从页面选择 eligible 状态"
        )
    candidate_count = 1 if scope.candidate_index is not None else cfg.pilot.total_candidates
    seed_count = 1 if scope.evaluation_seed is not None else len(cfg.pilot.evaluation_seeds)
    print(
        f"[evaluate] 计划匹配 {len(eligible_targets) * candidate_count * seed_count} 个 branch; "
        f"本次上限={scope.max_branches}, video={scope.save_video}",
        flush=True,
    )
    result_path = store.branch_dir / "branch_results.jsonl"
    # FAILED 行保留诊断，但不能阻止下次续跑重新尝试同一主键。
    completed = {
        (
            str(row["trajectory_id"]),
            int(row["decision_index"]),
            int(row["candidate_index"]),
            int(row["evaluation_seed"]),
        )
        for row in ArtifactStore.read_jsonl(result_path)
        if row.get("status", "OK") == "OK"
    }

    policy = FrozenVLAAdapter(cfg.paths.policy_checkpoint, cfg.policy)
    policy.load()
    simulator = CoffeeSimulatorAdapter(cfg.paths.dataset_config, cfg.simulator)
    attempted = 0
    matched = 0

    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        if scope.trajectory_id is not None and trajectory_id != scope.trajectory_id:
            continue
        eligible = eligible_by_trajectory.get(trajectory_id, set())
        for step in trajectory["steps"]:
            decision_index = int(step["decision_index"])
            if decision_index not in eligible:
                continue
            if scope.decision_index is not None and decision_index != scope.decision_index:
                continue
            candidates = _load_candidates(cfg, store, policy, trajectory_id, step)
            for candidate_index, (candidate, candidate_path) in enumerate(candidates):
                if scope.candidate_index is not None and candidate_index != scope.candidate_index:
                    continue
                for evaluation_seed in cfg.pilot.evaluation_seeds:
                    if not scope.matches(
                        trajectory_id, decision_index, candidate_index, evaluation_seed
                    ):
                        continue
                    matched += 1
                    key = (trajectory_id, decision_index, candidate_index, evaluation_seed)
                    if cfg.pilot.resume and key in completed and not scope.rerun_completed:
                        print(f"[evaluate] 已完成，跳过 key={key}", flush=True)
                        continue
                    if scope.max_branches is not None and attempted >= scope.max_branches:
                        print(
                            f"[evaluate] 达到 max_branches={scope.max_branches}，安全停止",
                            flush=True,
                        )
                        return ArtifactStore.read_jsonl(result_path)
                    print(f"[evaluate] 开始 key={key}, video={scope.save_video}", flush=True)
                    attempted += 1
                    try:
                        result = _evaluate_one(
                            cfg,
                            simulator,
                            policy,
                            store,
                            trajectory,
                            step,
                            candidate_index,
                            candidate,
                            candidate_path,
                            evaluation_seed,
                            save_video=scope.save_video,
                            video_fps=scope.video_fps,
                        )
                    except Exception as exc:
                        relative_candidate = _relative(candidate_path, store.run_dir)
                        result = BranchResult(
                            trajectory_id=trajectory_id,
                            decision_index=decision_index,
                            candidate_index=candidate_index,
                            is_nominal=candidate_index == 0,
                            evaluation_seed=evaluation_seed,
                            success=False,
                            finish_low_level_step=int(step["low_level_step"]),
                            executed_low_level_steps=0,
                            candidate_tokens_path=relative_candidate,
                            candidate_action_path=relative_candidate,
                            video_path=None,
                            status="FAILED",
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    ArtifactStore.append_jsonl(result_path, [result.to_dict()])
                    if result.status == "OK":
                        completed.add(key)
                    print(
                        f"[evaluate] 完成 key={key}, status={result.status}, "
                        f"success={result.success}, video={result.video_path}",
                        flush=True,
                    )
    if matched == 0:
        raise ValueError(
            "没有分支匹配指定过滤条件；请在 inspection/index.html 中确认 eligible 状态和 seed"
        )
    return ArtifactStore.read_jsonl(result_path)
