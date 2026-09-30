#!/usr/bin/env python3
"""采集 nominal coffee 轨迹。"""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="collect_nominal"):
        return 0
    from state_audit.collector import collect_nominal

    trajectories = collect_nominal(cfg, args.config)
    refresh_inspection(cfg)
    print(json.dumps({"num_trajectories": len(trajectories)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
