#!/usr/bin/env python
"""生成一个最小 DCCP branch artifact。

这个脚本不加载 world model，也不加载 LRM。它只构造一个和真实离线流程
兼容的最小 artifact，用于验证后续：
  1. progress LRM 能对 nominal / alternative branch videos 打分；
  2. margin 规则能构造 winner-loser pair；
  3. pref_* batch 能被 actor preference loss 消费。

后续如果要接真实 world model，只需要生成同样结构的 artifact.pt。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def make_video(num_frames: int, image_size: int, brightness_start: int, brightness_step: int) -> np.ndarray:
    """构造简单 RGB 视频；亮度变化只是为了让 LRM 输入不是空图。"""
    frames = []
    for idx in range(num_frames):
        value = np.uint8(np.clip(brightness_start + idx * brightness_step, 0, 255))
        frame = np.full((image_size, image_size, 3), value, dtype=np.uint8)
        # 加一条颜色条，避免所有帧完全均匀。
        frame[:, idx % image_size : (idx % image_size) + 2, 1] = 255 - value
        frames.append(frame)
    return np.stack(frames, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 toy DCCP branch artifact")
    parser.add_argument("--output", type=Path, required=True, help="输出 artifact.pt 路径")
    parser.add_argument("--image-size", type=int, default=96, help="toy 视频帧大小")
    parser.add_argument("--frames", type=int, default=5, help="每个 branch 的帧数")
    parser.add_argument("--prompt-len", type=int, default=8, help="fake prompt token 长度")
    parser.add_argument("--action-len", type=int, default=4, help="fake action token 长度")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    prompt_len = int(args.prompt_len)
    action_len = int(args.action_len)

    # context 对应 DCCP preference 中的 x：同一状态下比较两个动作。
    context = {
        "pref_input_ids": torch.arange(prompt_len, dtype=torch.long),
        "pref_attention_mask": torch.ones(prompt_len, dtype=torch.long),
        "pref_pixel_values": torch.zeros(3, 224, 224, dtype=torch.float32),
        "pref_response_mask": torch.ones(action_len, dtype=torch.bool),
    }

    artifact = {
        "format": "dccp_offline_branch_artifact_v1",
        "instruction": "Move the robot to complete the coffee task.",
        "rollout_id": "toy_rollout_0",
        "state_index": 1,
        "candidate_index": 1,
        "context": context,
        # nominal / alternative 对应论文里的 a^(0) 与 a^(k)。
        "nominal": {
            "response_tokens": torch.tensor([10, 11, 12, 13], dtype=torch.long),
            "branch_video": make_video(args.frames, args.image_size, brightness_start=40, brightness_step=5),
        },
        "alternative": {
            "response_tokens": torch.tensor([20, 21, 22, 23], dtype=torch.long),
            "branch_video": make_video(args.frames, args.image_size, brightness_start=80, brightness_step=12),
        },
        "metadata": {
            "note": "toy artifact; replace this file with real world-model branch artifact later",
        },
    }

    torch.save(artifact, args.output)
    print(f"[offline-dccp] saved toy artifact: {args.output}")
    print("[offline-dccp] next: start progress LRM and run score_artifact_with_progress_lrm.py")


if __name__ == "__main__":
    main()
