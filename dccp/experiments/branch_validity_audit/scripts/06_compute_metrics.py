#!/usr/bin/env python3
"""Compute label validity, ranking, calibration, and bootstrap metrics."""

import argparse
import json

from _bootstrap import prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = prepare(args.config)
    from branch_audit.metrics import compute_metrics
    report = compute_metrics(cfg)
    refresh(cfg)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
