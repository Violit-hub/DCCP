#!/usr/bin/env python3
"""检查分支完整性并生成主表。"""

import argparse
import json

from _bootstrap import prepare, refresh_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    from state_audit.reporting import build_reports

    print(json.dumps(build_reports(cfg), ensure_ascii=False, indent=2))
    refresh_inspection(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
