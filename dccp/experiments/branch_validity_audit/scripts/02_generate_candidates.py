#!/usr/bin/env python3
"""Generate and freeze the exact VLA candidate set."""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="generate_candidates", device=cfg.policy.device):
        refresh(cfg)
        return
    from branch_audit.candidates import generate_candidates
    rows = generate_candidates(cfg)
    refresh(cfg)
    print(json.dumps({"num_candidates": len(rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
