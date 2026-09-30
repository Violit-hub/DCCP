#!/usr/bin/env python3
"""Score WM videos with Progress LRM and freeze DCCP labels before simulator truth."""

import argparse
import json

from _bootstrap import prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-id")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--wm-seed", type=int)
    parser.add_argument("--max-scores", type=int)
    parser.add_argument("--score-only", action="store_true", help="do not freeze until all scores are inspected")
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    from branch_audit.lrm_runner import ScoringScope, score_predictions
    rows = score_predictions(cfg, ScoringScope(
        args.state_id, args.candidate_index, args.wm_seed,
        args.max_scores, args.rerun_completed,
    ), freeze=not args.score_only)
    refresh(cfg)
    print(json.dumps({"num_rows": len(rows), "frozen": not args.score_only}, ensure_ascii=False))


if __name__ == "__main__": main()
