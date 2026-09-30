#!/usr/bin/env python
"""读取离线 pref_* batch，并验证 B 部分 local DPO-style loss 接口。

默认使用 fake logprob，不加载 OpenVLA 大模型，目的是便宜地检查：
  - pref_* packed schema 是否完整；
  - pref_valid=False 时是否不炸；
  - 有效 pair 时 loss 是否为 scalar；
  - use_ref_gap=false 时是否稳定忽略 pref_delta_ref。
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verl.workers.actor.dp_rob import RobDataParallelPPOActor
from verl.utils.dccp_schema import PREF_KEYS


class AttrDict(dict):
    """让 dict 支持 config.xxx 访问，匹配 actor 里 OmegaConf 的使用习惯。"""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def str_to_bool(value: str) -> bool:
    return str(value).lower() in {"1", "true", "yes", "y"}


def make_fake_actor(use_ref_gap: bool) -> RobDataParallelPPOActor:
    """构造不加载大模型的 actor，只复用 _compute_dccp_pref_loss 逻辑。"""
    actor = RobDataParallelPPOActor.__new__(RobDataParallelPPOActor)
    actor.config = AttrDict(
        {
            "use_dccp_branch": True,
            "vla": "openvla-oft",
            "dccp": {
                "enable_loss": True,
                "beta": 0.5,
                "lambda_pref": 0.3,
                "lambda_pref_warmup_steps": 0,
                "use_ref_gap": bool(use_ref_gap),
                "packed_pref_interface": True,
            },
        }
    )
    actor.pad_token_id = 0

    def fake_logprob(self, input_ids, attention_mask, pixel_values, responses, response_mask, temperature):
        del input_ids, attention_mask, pixel_values, temperature
        # 简单、确定性的 sequence logprob：只依赖 action token 和 response mask。
        values = responses.float().reshape(responses.shape[0], -1).sum(dim=-1)
        mask_scale = response_mask.float().reshape(response_mask.shape[0], -1).mean(dim=-1)
        return values * mask_scale / 100.0

    actor._forward_dccp_sequence_logprob = MethodType(fake_logprob, actor)
    return actor


def main() -> None:
    parser = argparse.ArgumentParser(description="验证离线 pref_* batch 能被 DCCP preference loss 消费")
    parser.add_argument("--pref-batch", type=Path, required=True, help="score_artifact_with_progress_lrm.py 输出的 pref_batch.pt")
    parser.add_argument("--use-ref-gap", default="false", help="true/false；离线单卡 smoke 通常用 false")
    args = parser.parse_args()

    payload = torch.load(args.pref_batch, map_location="cpu")
    if payload.get("format") != "dccp_offline_pref_batch_v1":
        raise ValueError(f"unsupported pref batch format: {payload.get('format')}")

    batch = payload["pref_batch"]
    required = [
        PREF_KEYS.input_ids,
        PREF_KEYS.attention_mask,
        PREF_KEYS.pixel_values,
        PREF_KEYS.winner_responses,
        PREF_KEYS.loser_responses,
        PREF_KEYS.response_mask,
        PREF_KEYS.weight,
        PREF_KEYS.delta_ref,
        PREF_KEYS.margin,
        PREF_KEYS.valid,
    ]
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(f"pref batch missing keys: {missing}")

    actor = make_fake_actor(use_ref_gap=str_to_bool(args.use_ref_gap))
    loss, metrics = actor._compute_dccp_pref_loss(batch, temperature=1.0)

    print(f"[offline-dccp] loss scalar: shape={tuple(loss.shape)} value={float(loss.detach()):.6f}")
    for key in sorted(metrics):
        if key.startswith("dccp/") or key.startswith("loss/"):
            print(f"[offline-dccp] {key} = {metrics[key]}")


if __name__ == "__main__":
    main()
