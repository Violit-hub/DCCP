"""冻结 OpenVLA-OFT 的单卡推理适配器。"""

from __future__ import annotations

import contextlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PolicyConfig
from .seeding import temporary_seed


@dataclass(frozen=True)
class PolicyAction:
    actions: np.ndarray
    normalized_actions: np.ndarray
    response_tokens: np.ndarray
    entropy: float
    seed: int


class FrozenVLAAdapter:
    """直接加载冻结策略，不创建 optimizer，也不接入 Ray/FSDP。"""

    def __init__(self, checkpoint: str | Path, config: PolicyConfig):
        self.checkpoint = Path(checkpoint).expanduser().resolve()
        self.config = config
        self.model = None
        self.processor = None
        self.device = None
        self.torch_dtype = None
        self.unnorm_key = config.unnorm_key

    def load(self) -> None:
        import torch
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from verl.utils.vla_utils.openvla_oft.configuration_prismatic import OpenVLAConfig
        from verl.utils.vla_utils.openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from verl.utils.vla_utils.openvla_oft.processing_prismatic import (
            PrismaticImageProcessor,
            PrismaticProcessor,
        )

        for register in (
            lambda: AutoConfig.register("openvla", OpenVLAConfig),
            lambda: AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor),
            lambda: AutoProcessor.register(OpenVLAConfig, PrismaticProcessor),
            lambda: AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction),
        ):
            try:
                register()
            except ValueError:
                # 同一 Python 进程重复创建 adapter 时允许已注册。
                pass

        dtype_by_name = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        if self.config.dtype not in dtype_by_name:
            raise ValueError(f"不支持的 policy dtype: {self.config.dtype}")
        self.device = torch.device(self.config.device)
        self.torch_dtype = dtype_by_name[self.config.dtype]
        self.processor = AutoProcessor.from_pretrained(
            str(self.checkpoint), trust_remote_code=True, local_files_only=True
        )
        model_config = AutoConfig.from_pretrained(
            str(self.checkpoint), trust_remote_code=True, local_files_only=True
        )
        self.model = AutoModelForVision2Seq.from_pretrained(
            str(self.checkpoint),
            config=model_config,
            torch_dtype=self.torch_dtype,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if hasattr(self.model, "vision_backbone"):
            self.model.vision_backbone.set_num_images_in_input(1)
        statistics_path = self.checkpoint / "dataset_statistics.json"
        if statistics_path.exists():
            self.model.norm_stats = json.loads(statistics_path.read_text(encoding="utf-8"))
        if (
            self.unnorm_key not in self.model.norm_stats
            and f"{self.unnorm_key}_no_noops" in self.model.norm_stats
        ):
            self.unnorm_key = f"{self.unnorm_key}_no_noops"
        if self.unnorm_key not in self.model.norm_stats:
            raise KeyError(
                f"checkpoint 不含 unnorm_key={self.unnorm_key!r}; "
                f"可用键: {sorted(self.model.norm_stats)}"
            )

    def _require_loaded(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("请先调用 FrozenVLAAdapter.load()")

    def prepare_inputs(self, image: np.ndarray, instruction: str) -> dict[str, Any]:
        """使用与训练 rollout 一致的 prompt、中心裁剪和空 token。"""
        self._require_loaded()
        import torch
        from PIL import Image
        from verl.workers.rollout.rob_rollout import center_crop_image

        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).convert("RGB")
        if self.config.center_crop:
            pil_image = center_crop_image(pil_image)
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        feature = self.processor(prompt, pil_image)
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        if not torch.all(input_ids[:, -1] == 29871):
            empty_token = torch.tensor([[29871]], dtype=input_ids.dtype)
            input_ids = torch.cat([input_ids, empty_token], dim=1)
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(empty_token, dtype=attention_mask.dtype)], dim=1
            )
        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "pixel_values": feature["pixel_values"].to(self.device),
        }

    def generate_action(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        seed: int,
        do_sample: bool,
        temperature: float,
        compute_entropy: bool = True,
    ) -> PolicyAction:
        """生成一个完整 8x7 动作块，并按训练定义计算动作 token 熵。"""
        self._require_loaded()
        import torch

        inputs = self.prepare_inputs(image, instruction)
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with temporary_seed(seed), torch.inference_mode(), autocast:
            actions, response_tokens, normalized_actions = self.model.generate_action_verl(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                attention_mask=inputs["attention_mask"],
                padding_idx=self.processor.tokenizer.pad_token_id,
                do_sample=bool(do_sample),
                unnorm_key=self.unnorm_key,
                temperature=float(temperature),
            )
        response_tensor = response_tokens.detach().clone()
        entropy = self.compute_entropy(inputs, response_tensor) if compute_entropy else float("nan")
        actions_array = np.asarray(actions, dtype=np.float32).reshape(
            -1, self.config.action_dim
        )
        normalized_array = np.asarray(
            normalized_actions.detach().cpu() if hasattr(normalized_actions, "detach") else normalized_actions,
            dtype=np.float32,
        ).reshape(-1, self.config.action_dim)
        expected_shape = (self.config.action_chunk_length, self.config.action_dim)
        if actions_array.shape != expected_shape:
            raise ValueError(f"动作块形状错误: {actions_array.shape}，期望 {expected_shape}")
        return PolicyAction(
            actions=actions_array,
            normalized_actions=normalized_array,
            response_tokens=response_tensor.detach().cpu().numpy().reshape(-1).astype(np.int64),
            entropy=float(entropy),
            seed=int(seed),
        )

    def compute_entropy(self, inputs: dict[str, Any], response_tokens) -> float:
        """在最后 256 个动作 token 上计算每个位置的平均熵。"""
        import torch
        from verl.utils.dccp_action_logprob import (
            build_action_vocab_mask,
            compute_action_entropy_from_logits,
        )

        response_tokens = response_tokens.to(self.device)
        autocast = (
            torch.autocast(device_type="cuda", dtype=self.torch_dtype)
            if self.device.type == "cuda"
            else contextlib.nullcontext()
        )
        with torch.inference_mode(), autocast:
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs["pixel_values"],
            )
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits = outputs[0]
        else:
            logits = outputs
        prompt_length = int(inputs["input_ids"].shape[1])
        response_length = int(response_tokens.shape[1])
        if logits.shape[1] == response_length:
            action_logits = logits
        else:
            start = prompt_length - 1
            end = start + response_length
            action_logits = logits[:, start:end, :]
        if action_logits.shape[1] != response_length:
            raise ValueError(
                f"无法对齐 action logits: logits={tuple(logits.shape)}, "
                f"responses={tuple(response_tokens.shape)}"
            )
        tokenizer_vocab_size = int(getattr(self.model, "vocab_size", action_logits.shape[-1]))
        action_begin = tokenizer_vocab_size - 256
        action_mask = build_action_vocab_mask(
            vocab_size=int(action_logits.shape[-1]),
            device=action_logits.device,
            action_token_begin=action_begin,
            action_token_end=tokenizer_vocab_size,
        )
        entropy_result = compute_action_entropy_from_logits(
            logits=action_logits,
            response_mask=torch.ones_like(response_tokens, dtype=torch.bool),
            action_vocab_mask=action_mask,
        )
        return float(entropy_result.sequence_entropies.item())

    def sample_candidate_set(
        self,
        image: np.ndarray,
        instruction: str,
        nominal: PolicyAction,
        *,
        total_candidates: int,
        base_seed: int,
    ) -> list[PolicyAction]:
        """返回 nominal + 去重后的 alternatives；候选重复时自动重采样。"""
        candidates = [nominal]
        seen = {tuple(int(token) for token in nominal.response_tokens.tolist())}
        attempts = 0
        while len(candidates) < int(total_candidates) and attempts < self.config.candidate_max_attempts:
            candidate_seed = int(base_seed) + attempts
            candidate = self.generate_action(
                image,
                instruction,
                seed=candidate_seed,
                do_sample=True,
                temperature=self.config.candidate_temperature,
                compute_entropy=False,
            )
            key = tuple(int(token) for token in candidate.response_tokens.tolist())
            attempts += 1
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        if len(candidates) != int(total_candidates):
            raise RuntimeError(
                f"候选动作去重后只有 {len(candidates)}/{total_candidates} 个；"
                f"已尝试 {attempts} 次"
            )
        return candidates
