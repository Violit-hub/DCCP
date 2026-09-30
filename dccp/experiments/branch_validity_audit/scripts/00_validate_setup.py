#!/usr/bin/env python3
"""Validate paths and report service/GPU state without loading large models."""

import argparse
import json
from pathlib import Path

from _bootstrap import prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config, require_inputs=True)
    from branch_audit.artifacts import AuditStore
    from state_audit.gpu import query_compute_processes, query_gpu_status
    from state_audit.progress import check_progress_service

    store = AuditStore(cfg.run_dir)
    store.ensure_layout()
    service = None
    try:
        service = check_progress_service(cfg.paths.progress_lrm_config, cfg.runtime.progress_health_timeout_sec)
    except Exception as exc:
        service = {"healthy": False, "error": f"{type(exc).__name__}: {exc}"}
    report = {
        "passed_static_paths": True,
        "run_dir": str(cfg.run_dir),
        "config": str(Path(args.config).resolve()),
        "gpus": [row.__dict__ for row in query_gpu_status()],
        "compute_processes": query_compute_processes(),
        "progress_service": service,
        "note": "LRM service may be started later; an unhealthy service does not fail static setup.",
    }
    AuditStore.write_json(store.reports / "setup.json", report)
    refresh(cfg)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
