#!/usr/bin/env python3
"""加载冻结策略并执行状态恢复一致性闸门。"""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="state_restore_gate"):
        return 0
    from state_audit.restore_validator import validate_state_restore

    print(json.dumps(validate_state_restore(cfg), ensure_ascii=False, indent=2))
    refresh_inspection(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
