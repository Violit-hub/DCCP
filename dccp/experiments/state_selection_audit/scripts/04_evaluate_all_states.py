#!/usr/bin/env python3
"""在真实模拟器中评估全部 eligible 状态。"""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh_inspection


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--trajectory-id")
    parser.add_argument("--decision-index", type=int)
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--max-branches", type=int)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="evaluate_all_states"):
        return 0
    from state_audit.branch_evaluator import EvaluationScope, evaluate_all_valid_states

    scope = EvaluationScope(
        trajectory_id=args.trajectory_id,
        decision_index=args.decision_index,
        candidate_index=args.candidate_index,
        evaluation_seed=args.evaluation_seed,
        max_branches=args.max_branches,
        save_video=args.save_video,
        video_fps=args.video_fps,
        rerun_completed=args.rerun_completed,
    )
    rows = evaluate_all_valid_states(cfg, scope)
    refresh_inspection(cfg)
    print(json.dumps({"num_branch_rows": len(rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
