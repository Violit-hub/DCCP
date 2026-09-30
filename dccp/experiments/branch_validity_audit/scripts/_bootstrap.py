"""Shared path, environment, GPU guard, and inspection refresh."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


AUDIT_ROOT = Path(__file__).resolve().parents[1]
STATE_AUDIT_ROOT = AUDIT_ROOT.parent / "state_selection_audit"
for path in (AUDIT_ROOT, STATE_AUDIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from branch_audit.config import load_config  # noqa: E402


def prepare(config_path: str, *, require_inputs: bool = True):
    cfg = load_config(config_path, require_inputs=require_inputs)
    dccp_package = str(Path(cfg.paths.dccp_root).resolve() / "dccp")
    opensora = str(Path(cfg.paths.dccp_root).resolve() / "dccp" / "dependencies" / "opensora")
    for path in (dccp_package, opensora):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("MUJOCO_PY_MUJOCO_PATH", cfg.paths.mujoco_path)
    additions = [str(Path(cfg.paths.mujoco_path) / "bin"), "/usr/lib/nvidia"]
    old = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(additions + ([old] if old else []))
    os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
    os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])
    return cfg


def gpu_guard(cfg, *, stage: str, device: str) -> bool:
    from state_audit.gpu import gpu_is_busy, query_compute_processes, query_gpu_status

    statuses = query_gpu_status()
    payload = {
        "stage": stage,
        "requested_device": device,
        "gpus": [item.__dict__ for item in statuses],
        "compute_processes": query_compute_processes(),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2), flush=True)
    if not statuses:
        if cfg.runtime.skip_gpu_stages_when_busy:
            print(f"SKIPPED: no NVIDIA GPU detected; {stage} was not run", flush=True)
            return False
        raise RuntimeError("No NVIDIA GPU detected")
    index = int(str(device).split(":")[-1])
    status = next((item for item in statuses if item.index == index), None)
    if status is None:
        raise RuntimeError(f"Configured GPU {index} does not exist")
    busy = gpu_is_busy(status, cfg.runtime.gpu_memory_threshold_mib)
    if busy and cfg.runtime.skip_gpu_stages_when_busy:
        print(
            f"SKIPPED: GPU {index} busy (used={status.memory_used_mib} MiB, util={status.utilization_percent}%); "
            f"{stage} was not run",
            flush=True,
        )
        return False
    if busy:
        raise RuntimeError(f"GPU {index} is busy; refusing to run {stage}")
    return True


def refresh(cfg):
    from branch_audit.visualization import generate_inspection
    path = generate_inspection(cfg)
    print(f"[inspect] {path}", flush=True)
    return path
