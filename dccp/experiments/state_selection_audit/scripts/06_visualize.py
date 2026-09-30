#!/usr/bin/env python3
"""根据当前已有产物生成可离线打开的人工检查页面。"""

import argparse
import json

from _bootstrap import prepare


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config, require_inputs=False)
    from state_audit.visualization import generate_inspection_site

    index = generate_inspection_site(cfg)
    print(json.dumps({"inspection_html": str(index)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
