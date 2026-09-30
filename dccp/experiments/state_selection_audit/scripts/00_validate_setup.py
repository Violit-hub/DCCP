#!/usr/bin/env python3
"""检查路径、GPU 和 Progress LRM 服务，不加载大模型。"""

import argparse
import json
import sys
from pathlib import Path

from _bootstrap import prepare


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-progress", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    from state_audit.gpu import query_compute_processes, query_gpu_status
    from state_audit.progress import check_progress_service

    progress = None
    progress_error = None
    try:
        progress = check_progress_service(
            cfg.paths.progress_lrm_config, cfg.runtime.progress_health_timeout_sec
        )
    except Exception as exc:
        progress_error = f"{type(exc).__name__}: {exc}"
    payload = {
        "config": str(Path(args.config).resolve()),
        "run_dir": str(cfg.run_dir),
        "python": sys.executable,
        "gpus": [item.__dict__ for item in query_gpu_status()],
        "compute_processes": query_compute_processes(),
        "progress_service": progress,
        "progress_error": progress_error,
        "status": "OK" if not (args.require_progress and progress_error) else "FAILED",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 1 if args.require_progress and progress_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
