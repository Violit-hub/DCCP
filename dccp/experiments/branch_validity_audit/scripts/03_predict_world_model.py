#!/usr/bin/env python3
"""Generate WM videos with fine-grained resumable filters."""

import argparse
import json

from _bootstrap import gpu_guard, prepare, refresh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--state-id")
    parser.add_argument("--candidate-index", type=int)
    parser.add_argument("--wm-seed", type=int)
    parser.add_argument("--max-predictions", type=int)
    parser.add_argument("--rerun-completed", action="store_true")
    args = parser.parse_args()
    cfg = prepare(args.config)
    if not gpu_guard(cfg, stage="predict_world_model", device=cfg.world_model.device):
        refresh(cfg)
        return
    from branch_audit.world_model_runner import PredictionScope, predict_world_model
    rows = predict_world_model(cfg, PredictionScope(
        args.state_id, args.candidate_index, args.wm_seed,
        args.max_predictions, args.rerun_completed,
    ))
    refresh(cfg)
    print(json.dumps({"num_prediction_rows": len(rows)}, ensure_ascii=False))


if __name__ == "__main__": main()
