"""在查看 simulator 分支结果之前冻结四种状态选择所需信息。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .artifact_store import ArtifactStore
from .config import AuditConfig
from .progress import load_progress_scorer, score_nominal_progress
from .seeding import derive_seed
from .selectors import draw_random_selections, select_scored_methods


def _load_trajectories(store: ArtifactStore) -> list[dict[str, Any]]:
    path = store.nominal_dir / "trajectories.json"
    if not path.exists():
        raise FileNotFoundError("尚未采集 nominal 轨迹，请先运行 02_collect_nominal.py")
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_selections(cfg: AuditConfig, *, force: bool = False) -> list[dict[str, Any]]:
    """冻结选择；branch 结果存在后禁止首次或强制重算，防止信息泄漏。"""
    store = ArtifactStore(cfg.run_dir)
    store.ensure_layout()
    selection_path = store.selection_dir / "selected_states.jsonl"
    frozen_path = store.selection_dir / "SELECTIONS_FROZEN.json"
    branch_path = store.branch_dir / "branch_results.jsonl"

    if frozen_path.exists() and selection_path.exists() and not force:
        return ArtifactStore.read_jsonl(selection_path)
    if branch_path.exists() and branch_path.stat().st_size > 0:
        raise RuntimeError("已有 branch 结果，禁止重算选择；请新建 run_name")

    trajectories = _load_trajectories(store)
    scorer = load_progress_scorer(
        cfg.paths.progress_lrm_config,
        cache_dir=store.selection_dir / "lrm_cache",
    )
    rows: list[dict[str, Any]] = []

    for trajectory in trajectories:
        trajectory_id = str(trajectory["trajectory_id"])
        print(f"[select] {trajectory_id}: 请求 Progress LRM", flush=True)
        steps = list(trajectory["steps"])
        frames = np.stack(
            [
                np.asarray(Image.open(store.run_dir / step["frame_path"]).convert("RGB"))
                for step in steps
            ],
            axis=0,
        )
        starts, progress = score_nominal_progress(
            frames,
            cfg.policy.instruction,
            scorer,
            horizon_H=cfg.selection.horizon_H,
            trajectory_id=trajectory_id,
        )
        entropy = np.asarray([float(steps[int(index)]["entropy"]) for index in starts])

        from verl.utils.dccp_mining import DCCPMiningConfig, select_decision_sensitive_states

        diagnostic = select_decision_sensitive_states(
            progress,
            entropy,
            DCCPMiningConfig(
                state_budget_per_traj=cfg.selection.state_budget,
                nms_gap=cfg.selection.nms_gap,
                lambda_curvature=cfg.selection.lambda_curvature,
                lambda_entropy=cfg.selection.lambda_entropy,
                require_entropy=True,
            ),
        )
        eligible_indices = [
            int(starts[index]) for index in np.flatnonzero(diagnostic.valid_mask)
        ]
        methods = select_scored_methods(
            progress,
            entropy,
            starts,
            state_budget=cfg.selection.state_budget,
            nms_gap=cfg.selection.nms_gap,
            lambda_curvature=cfg.selection.lambda_curvature,
            lambda_entropy=cfg.selection.lambda_entropy,
        )
        for method_name, selection in methods.items():
            rows.append(
                {
                    "trajectory_id": trajectory_id,
                    "method": method_name,
                    "selected_indices": selection.decision_indices,
                    "scores": selection.scores,
                    "eligible_indices": eligible_indices,
                    "selection_seed": None,
                }
            )

        # 保存一次可检查的随机盲选；最终随机基线会用大量合法抽样估计期望。
        random_seed = derive_seed(cfg.run_name, trajectory_id, "random_selection")
        random_draw = draw_random_selections(
            diagnostic.valid_mask,
            budget=cfg.selection.state_budget,
            nms_gap=cfg.selection.nms_gap,
            draws=1,
            seed=random_seed,
        )
        random_local = list(random_draw[0]) if random_draw else []
        rows.append(
            {
                "trajectory_id": trajectory_id,
                "method": "random",
                "selected_indices": [int(starts[index]) for index in random_local],
                "scores": [],
                "eligible_indices": eligible_indices,
                "selection_seed": random_seed,
            }
        )
        ArtifactStore.write_npz_atomic(
            store.selection_dir / f"{trajectory_id}_scores.npz",
            valid_start_indices=starts,
            progress_scores=progress,
            entropy_scores=entropy,
            curvature_scores=diagnostic.curvature_scores,
            decision_scores=diagnostic.decision_scores,
            valid_mask=diagnostic.valid_mask,
        )
        print(
            f"[select] {trajectory_id}: eligible={len(eligible_indices)}, "
            + ", ".join(
                f"{name}={selection.decision_indices}" for name, selection in methods.items()
            ),
            flush=True,
        )

    ArtifactStore.write_jsonl_atomic(selection_path, rows)
    ArtifactStore.write_json_atomic(
        frozen_path,
        {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "num_rows": len(rows),
            "branch_results_existed_at_freeze": False,
            "warning": "冻结后不得依据 simulator branch 结果修改选择",
        },
    )
    return rows
