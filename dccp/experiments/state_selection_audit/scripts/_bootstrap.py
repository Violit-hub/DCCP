"""各阶段脚本共享的路径、MuJoCo 和 GPU 防护。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
if str(AUDIT_ROOT) not in sys.path:
    sys.path.insert(0, str(AUDIT_ROOT))

from state_audit.config import load_config  # noqa: E402
from state_audit.gpu import (  # noqa: E402
    gpu_is_busy,
    query_compute_processes,
    query_gpu_status,
)


def prepare(config_path: str, *, require_inputs: bool = True):
    cfg = load_config(config_path, require_inputs=require_inputs)
    canonical_dccp_package = str(Path(cfg.paths.dccp_root).resolve() / "dccp")
    if canonical_dccp_package not in sys.path:
        sys.path.insert(0, canonical_dccp_package)
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", cfg.paths.mujoco_path)
    additions = [str(Path(cfg.paths.mujoco_path) / "bin"), "/usr/lib/nvidia"]
    old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(additions + ([old] if old else []))
    return cfg


def gpu_guard(cfg, *, stage: str) -> bool:
    statuses = query_gpu_status()
    processes = query_compute_processes()
    payload = {
        "stage": stage,
        "gpus": [item.__dict__ for item in statuses],
        "compute_processes": processes,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not statuses:
        raise RuntimeError("没有检测到 NVIDIA GPU")
    device_index = int(str(cfg.policy.device).split(":")[-1])
    status = next((item for item in statuses if item.index == device_index), None)
    if status is None:
        raise RuntimeError(f"配置 GPU {device_index} 不存在")
    memory_busy = gpu_is_busy(status, cfg.runtime.gpu_memory_threshold_mib)
    process_busy = bool(processes) and not cfg.runtime.allow_existing_gpu_processes_below_threshold
    busy = memory_busy or process_busy
    if busy and cfg.runtime.skip_gpu_tests_when_busy:
        print(f"SKIPPED: GPU {device_index} 正被占用，未执行 {stage}", flush=True)
        return False
    if busy:
        raise RuntimeError(f"GPU {device_index} 正被占用，拒绝执行 {stage}")
    return True


def refresh_inspection(cfg) -> str:
    """阶段完成后自动刷新检查页，同时保留手动 inspect 入口。"""
    from state_audit.visualization import generate_inspection_site

    path = generate_inspection_site(cfg)
    print(f"[inspect] 已刷新: {path}", flush=True)
    return str(path)
