#!/usr/bin/env python3
"""生成固定 candidate 与 nominal 的确认性25-seed报告。"""

import argparse
import json

from _bootstrap import prepare


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    from state_audit.stability_reporting import build_fixed_seed_stability_report

    report = build_fixed_seed_stability_report(cfg)
    if report is None:
        raise RuntimeError("当前 run 不是 fixed-seed stability 实验")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
