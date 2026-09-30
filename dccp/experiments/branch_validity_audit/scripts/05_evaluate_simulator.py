#!/usr/bin/env python3
"""Execute frozen actions in the real simulator after label freeze."""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-id")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--max-rollouts", type=int)
    parser.add_argument("--rerun-completed", action="store_true")
    video = parser.add_mutually_exclusive_group()
    video.add_argument("--save-video", action="store_true")
    video.add_argument("--no-video", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="evaluate_simulator", device=cfg.policy.device):
        refresh(cfg)
        return
    from branch_audit.simulator_runner import SimulatorScope, evaluate_in_simulator
    save_video = True if args.save_video else (False if args.no_video else None)
    rows = evaluate_in_simulator(cfg, SimulatorScope(
        args.state_id, args.candidate_index, args.evaluation_seed,
        args.max_rollouts, args.rerun_completed, save_video,
    ))
    refresh(cfg)
    print(json.dumps({"num_outcome_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
