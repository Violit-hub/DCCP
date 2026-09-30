#!/usr/bin/env python3
"""Copy and freeze selected restorable real states."""

import argparse
import json

from _bootstrap import prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    from branch_audit.source_states import prepare_states
    rows = prepare_states(cfg)
    refresh(cfg)
    print(json.dumps({"num_states": len(rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
