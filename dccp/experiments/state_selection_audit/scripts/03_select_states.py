#!/usr/bin/env python3
"""调用 Progress LRM 并冻结状态选择。"""

import argparse
import json

from _bootstrap import prepare, refresh_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    from state_audit.selection_runner import freeze_selections

    rows = freeze_selections(cfg, force=args.force)
    refresh_inspection(cfg)
    print(json.dumps({"num_selection_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
