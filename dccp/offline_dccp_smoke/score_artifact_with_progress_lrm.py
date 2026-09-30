#!/usr/bin/env python
"""离线调用 progress LRM，并把 branch artifact 转成 packed pref_* batch。

这个脚本只需要 progress server，不需要 world model / actor / ref 常驻显存。
它实现论文中的局部分支打分和 high-margin preference construction：

    margin = S_prog(alternative_branch) - S_prog(nominal_branch)

    margin > delta_plus    => alternative 赢 nominal
    margin < -delta_minus  => nominal 赢 alternative
    其他情况               => pref_valid=False，丢弃该 pair
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.utils.dccp_lrm_client import DCCPLRMClient, DCCPLRMClientConfig
from verl.utils.dccp_scorer import DCCPScorer, DCCPScorerConfig
from verl.utils.dccp_schema import PREF_KEYS


def _as_pair_dim(tensor: torch.Tensor) -> torch.Tensor:
    """把单个 pair 的 tensor 打包成 [B=1, P=1, ...]。"""
    return tensor.unsqueeze(0).unsqueeze(0)


def build_pref_batch(artifact: dict, nominal_score: float, alternative_score: float, delta_plus: float, delta_minus: float) -> dict:
    """根据 progress margin 构造一个 packed pref_* batch。"""
    margin = float(alternative_score) - float(nominal_score)
    valid = False

    nominal_tokens = artifact["nominal"]["response_tokens"].detach().clone().long()
    alternative_tokens = artifact["alternative"]["response_tokens"].detach().clone().long()

    if margin > float(delta_plus):
        # alternative branch 局部进展更高，因此 alternative 是 winner。
        winner = alternative_tokens
        loser = nominal_tokens
        valid = True
    elif margin < -float(delta_minus):
        # nominal branch 局部进展更高，因此 nominal 是 winner。
        winner = nominal_tokens
        loser = alternative_tokens
        valid = True
    else:
        # 小 margin 不可靠，保留 slot 但标记 invalid。
        winner = torch.zeros_like(nominal_tokens)
        loser = torch.zeros_like(nominal_tokens)

    context = artifact["context"]
    response_mask = context.get(PREF_KEYS.response_mask, torch.ones_like(nominal_tokens, dtype=torch.bool))

    batch = {
        PREF_KEYS.input_ids: _as_pair_dim(context[PREF_KEYS.input_ids].long()),
        PREF_KEYS.attention_mask: _as_pair_dim(context[PREF_KEYS.attention_mask].long()),
        PREF_KEYS.pixel_values: _as_pair_dim(context[PREF_KEYS.pixel_values].float()),
        PREF_KEYS.winner_responses: _as_pair_dim(winner.long()),
        PREF_KEYS.loser_responses: _as_pair_dim(loser.long()),
        PREF_KEYS.response_mask: _as_pair_dim(response_mask.bool()),
        PREF_KEYS.weight: torch.tensor([[abs(margin) if valid else 0.0]], dtype=torch.float32),
        # 单独离线打分阶段不加载 ref policy，因此这里默认 ref gap 为 0。
        PREF_KEYS.delta_ref: torch.zeros((1, 1), dtype=torch.float32),
        PREF_KEYS.margin: torch.tensor([[margin]], dtype=torch.float32),
        PREF_KEYS.valid: torch.tensor([[valid]], dtype=torch.bool),
        PREF_KEYS.nominal_score: torch.tensor([[float(nominal_score)]], dtype=torch.float32),
        PREF_KEYS.alternative_score: torch.tensor([[float(alternative_score)]], dtype=torch.float32),
        PREF_KEYS.state_index: torch.tensor([[int(artifact.get("state_index", -1))]], dtype=torch.long),
        PREF_KEYS.candidate_index: torch.tensor([[int(artifact.get("candidate_index", -1))]], dtype=torch.long),
    }
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="用 progress LRM 给离线 DCCP artifact 打分并生成 pref_* batch")
    parser.add_argument("--artifact", type=Path, required=True, help="输入 artifact.pt")
    parser.add_argument("--output", type=Path, required=True, help="输出 pref_batch.pt")
    parser.add_argument("--progress-endpoint", default="http://127.0.0.1:8002/progress")
    parser.add_argument("--timeout-sec", type=float, default=300.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--cache-dir", default="./cache/lrm_scores")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--progress-num-keyframes", type=int, default=4)
    parser.add_argument("--delta-plus", type=float, default=0.02)
    parser.add_argument("--delta-minus", type=float, default=0.02)
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu")
    if artifact.get("format") != "dccp_offline_branch_artifact_v1":
        raise ValueError(f"unsupported artifact format: {artifact.get('format')}")

    client = DCCPLRMClient(
        DCCPLRMClientConfig(
            completion_endpoint="",
            progress_endpoint=str(args.progress_endpoint),
            timeout_sec=float(args.timeout_sec),
            max_retries=int(args.max_retries),
            retry_sleep_sec=1.0,
            use_cache=True,
            cache_dir=str(args.cache_dir),
        )
    )
    scorer = DCCPScorer(
        client=client,
        config=DCCPScorerConfig(
            completion_num_keyframes=8,
            progress_num_keyframes=int(args.progress_num_keyframes),
            image_size=int(args.image_size),
        ),
    )

    instruction = str(artifact["instruction"])
    nominal_score = scorer.score_local_progress(
        branch_or_suffix_video=artifact["nominal"]["branch_video"],
        instruction=instruction,
        metadata={"score_type": "offline_nominal_branch_progress", "rollout_id": artifact.get("rollout_id", "")},
    )
    alternative_score = scorer.score_local_progress(
        branch_or_suffix_video=artifact["alternative"]["branch_video"],
        instruction=instruction,
        metadata={"score_type": "offline_alternative_branch_progress", "rollout_id": artifact.get("rollout_id", "")},
    )

    pref_batch = build_pref_batch(
        artifact=artifact,
        nominal_score=float(nominal_score),
        alternative_score=float(alternative_score),
        delta_plus=float(args.delta_plus),
        delta_minus=float(args.delta_minus),
    )

    payload = {
        "format": "dccp_offline_pref_batch_v1",
        "pref_batch": pref_batch,
        "scores": {
            "nominal_score": float(nominal_score),
            "alternative_score": float(alternative_score),
            "margin": float(alternative_score) - float(nominal_score),
            "valid_pairs": int(pref_batch[PREF_KEYS.valid].sum().item()),
        },
        "source_artifact": str(args.artifact),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)

    print(f"[offline-dccp] nominal_score={float(nominal_score):.6f}")
    print(f"[offline-dccp] alternative_score={float(alternative_score):.6f}")
    print(f"[offline-dccp] margin={payload['scores']['margin']:.6f}")
    print(f"[offline-dccp] valid_pairs={payload['scores']['valid_pairs']}")
    print(f"[offline-dccp] saved pref batch: {args.output}")


if __name__ == "__main__":
    main()
