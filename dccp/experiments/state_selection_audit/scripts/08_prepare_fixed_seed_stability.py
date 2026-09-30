#!/usr/bin/env python3
"""从已完成的审计中冻结状态和候选，创建全新的 seed 稳定性实验目录。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from state_audit.artifact_store import ArtifactStore  # noqa: E402
from state_audit.config import load_config  # noqa: E402


PREPARATION_FILE = "fixed_seed_stability.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {label}: {path}")
    return path


def _chmod_shared(root: Path) -> None:
    for directory, _, filenames in os.walk(root):
        Path(directory).chmod(0o2775)
        for filename in filenames:
            path = Path(directory) / filename
            path.chmod(path.stat().st_mode | 0o660)


def _expected_identity(
    *,
    cfg,
    source_run: Path,
    trajectory_id: str,
    decision_indices: list[int],
) -> dict[str, Any]:
    return {
        "kind": "fixed_state_fixed_candidate_seed_stability",
        "run_name": cfg.run_name,
        "source_run_dir": str(source_run),
        "trajectory_id": trajectory_id,
        "decision_indices": decision_indices,
        "evaluation_seeds": list(cfg.pilot.evaluation_seeds),
        "total_candidates": int(cfg.pilot.total_candidates),
    }


def _validate_existing(target: Path, expected: dict[str, Any]) -> bool:
    preparation = target / PREPARATION_FILE
    if not target.exists():
        return False
    if not preparation.is_file():
        raise FileExistsError(
            f"目标目录已存在但不是本脚本准备的实验，拒绝覆盖: {target}"
        )
    payload = _read_json(preparation)
    actual = {key: payload.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(
            "目标目录的冻结定义与本次请求不一致，拒绝复用。\n"
            f"expected={expected}\nactual={actual}"
        )
    print(f"[prepare-stability] 已准备且定义一致，直接复用: {target}")
    return True


def _validate_source(
    *,
    source_run: Path,
    trajectory_id: str,
    decision_indices: list[int],
    total_candidates: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, dict[str, Any]]]:
    trajectories_path = _require_file(
        source_run / "nominal" / "trajectories.json", "source trajectories"
    )
    trajectories = _read_json(trajectories_path)
    matches = [row for row in trajectories if str(row["trajectory_id"]) == trajectory_id]
    if len(matches) != 1:
        raise ValueError(
            f"source 中 trajectory_id={trajectory_id!r} 应恰好出现一次，实际 {len(matches)}"
        )
    trajectory = matches[0]
    steps = {int(step["decision_index"]): step for step in trajectory.get("steps", [])}
    missing_steps = sorted(set(decision_indices) - set(steps))
    if missing_steps:
        raise ValueError(f"source nominal 缺少 decision indices: {missing_steps}")

    manifests: dict[int, dict[str, Any]] = {}
    for decision_index in decision_indices:
        manifest_path = _require_file(
            source_run
            / "branches"
            / "candidate_sets"
            / trajectory_id
            / f"step_{decision_index:04d}"
            / "candidates.json",
            f"candidate manifest for state {decision_index}",
        )
        manifest = _read_json(manifest_path)
        if int(manifest.get("total_candidates", -1)) != total_candidates:
            raise ValueError(
                f"状态 {decision_index} 的候选数为 {manifest.get('total_candidates')}，"
                f"目标配置要求 {total_candidates}"
            )
        rows = manifest.get("candidates", [])
        indices = [int(row["candidate_index"]) for row in rows]
        if indices != list(range(total_candidates)):
            raise ValueError(f"状态 {decision_index} 的候选编号不完整: {indices}")
        for row in rows:
            relative = Path(str(row["path"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"不安全的候选相对路径: {relative}")
            _require_file(source_run / relative, f"candidate {relative}")
        manifests[decision_index] = manifest
    return trajectory, trajectories, manifests


def prepare(
    *,
    config_path: Path,
    source_run: Path,
    trajectory_id: str,
    decision_indices: list[int],
) -> Path:
    cfg = load_config(config_path, require_inputs=True)
    source_run = source_run.expanduser().resolve()
    target = cfg.run_dir
    decision_indices = sorted(set(int(value) for value in decision_indices))
    if not decision_indices:
        raise ValueError("至少需要一个 decision index")
    if source_run == target:
        raise ValueError("source 和 target run_dir 不能相同")
    if not source_run.is_dir():
        raise FileNotFoundError(f"source run 不存在: {source_run}")

    expected = _expected_identity(
        cfg=cfg,
        source_run=source_run,
        trajectory_id=trajectory_id,
        decision_indices=decision_indices,
    )
    if _validate_existing(target, expected):
        return target

    _, trajectories, manifests = _validate_source(
        source_run=source_run,
        trajectory_id=trajectory_id,
        decision_indices=decision_indices,
        total_candidates=cfg.pilot.total_candidates,
    )
    restore_report = _require_file(
        source_run / "reports" / "state_restore_gate.json", "source restore gate"
    )
    if _read_json(restore_report).get("passed") is not True:
        raise RuntimeError("source 的状态恢复闸门没有通过")
    score_path = _require_file(
        source_run / "selections" / f"{trajectory_id}_scores.npz",
        "source selection scores",
    )

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{cfg.run_name}.prepare-", dir=target.parent)
    )
    try:
        store = ArtifactStore(temporary)
        store.ensure_layout()
        shutil.copytree(
            source_run / "nominal", store.nominal_dir, dirs_exist_ok=True, copy_function=shutil.copy2
        )
        shutil.copy2(restore_report, store.report_dir / restore_report.name)
        shutil.copy2(score_path, store.selection_dir / score_path.name)

        candidate_hashes: dict[str, str] = {}
        for decision_index, manifest in manifests.items():
            source_dir = (
                source_run
                / "branches"
                / "candidate_sets"
                / trajectory_id
                / f"step_{decision_index:04d}"
            )
            destination_dir = (
                store.branch_dir
                / "candidate_sets"
                / trajectory_id
                / f"step_{decision_index:04d}"
            )
            shutil.copytree(source_dir, destination_dir, copy_function=shutil.copy2)
            for item in manifest["candidates"]:
                relative = Path(str(item["path"]))
                source_hash = _sha256(source_run / relative)
                destination_hash = _sha256(temporary / relative)
                if source_hash != destination_hash:
                    raise RuntimeError(f"候选复制后哈希不一致: {relative}")
                candidate_hashes[str(relative)] = source_hash

        selection_row = {
            "trajectory_id": trajectory_id,
            "method": "fixed_previous_oracle",
            "selected_indices": decision_indices,
            "scores": [],
            "eligible_indices": decision_indices,
            "selection_seed": None,
            "source_run_dir": str(source_run),
            "purpose": "post-hoc stability check; not an unbiased state-selection evaluation",
        }
        ArtifactStore.write_jsonl_atomic(
            store.selection_dir / "selected_states.jsonl", [selection_row]
        )
        ArtifactStore.write_json_atomic(
            store.selection_dir / "SELECTIONS_FROZEN.json",
            {
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "num_rows": 1,
                "branch_results_existed_at_freeze": False,
                "source_run_dir": str(source_run),
                "fixed_decision_indices": decision_indices,
                "warning": "post-hoc stability check; states and candidates must not be changed",
            },
        )
        preparation = {
            **expected,
            "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
            "expected_branches": (
                len(decision_indices)
                * int(cfg.pilot.total_candidates)
                * len(cfg.pilot.evaluation_seeds)
            ),
            "candidate_sha256": candidate_hashes,
            "source_trajectories_sha256": _sha256(
                source_run / "nominal" / "trajectories.json"
            ),
            "config_path": str(config_path.resolve()),
            "source_trajectory_count": len(trajectories),
        }
        ArtifactStore.write_json_atomic(temporary / PREPARATION_FILE, preparation)
        ArtifactStore.write_json_atomic(
            temporary / "manifest.json",
            {
                "task": cfg.task,
                "run_name": cfg.run_name,
                "experiment": "fixed-state/fixed-candidate 25-seed stability",
                "provenance": PREPARATION_FILE,
            },
        )
        _chmod_shared(temporary)
        if target.exists():
            raise FileExistsError(f"准备期间目标目录被其他进程创建，拒绝覆盖: {target}")
        temporary.rename(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    print(
        f"[prepare-stability] 已创建独立实验: {target}\n"
        f"[prepare-stability] fixed states={decision_indices}, "
        f"seeds={len(cfg.pilot.evaluation_seeds)}, "
        f"expected branches={preparation['expected_branches']}"
    )
    return target


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-run-dir", required=True)
    parser.add_argument("--trajectory-id", default="traj_0000")
    parser.add_argument("--decision-index", type=int, action="append", required=True)
    args = parser.parse_args()
    prepare(
        config_path=Path(args.config).expanduser().resolve(),
        source_run=Path(args.source_run_dir),
        trajectory_id=args.trajectory_id,
        decision_indices=args.decision_index,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
