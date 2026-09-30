"""Minimal frozen OpenSora world-model inference, extracted from DCCP rollout code."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

from .config import AuditConfig
from .seeding import derive_seed, seed_everything


class StandaloneWorldModel:
    """Loads VAE/model/scheduler only; no Ray, FSDP, optimizer, or training state."""

    def __init__(self, cfg: AuditConfig, policy):
        self.audit_cfg = cfg
        self.policy = policy
        self.device = None
        self.dtype = None
        self.vae = None
        self.model = None
        self.scheduler = None
        self.model_args = None
        self.latent_size = None
        self.image_size = None

    def load(self) -> None:
        import torch

        opensora_root = Path(self.audit_cfg.paths.dccp_root).resolve() / "dccp" / "dependencies" / "opensora"
        if str(opensora_root) not in sys.path:
            sys.path.insert(0, str(opensora_root))
        from opensora.datasets.aspect import get_image_size, get_num_frames
        from opensora.registry import MODELS, SCHEDULERS, build_module
        from opensora.utils.config_utils import read_config
        from opensora.utils.inference_utils import prepare_multi_resolution_info

        if not torch.cuda.is_available() and str(self.audit_cfg.world_model.device).startswith("cuda"):
            raise RuntimeError("World-model inference requires CUDA with the configured checkpoint")
        self.device = torch.device(self.audit_cfg.world_model.device)
        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        if self.audit_cfg.world_model.dtype not in dtype_map:
            raise ValueError(f"Unsupported world-model dtype: {self.audit_cfg.world_model.dtype}")
        self.dtype = dtype_map[self.audit_cfg.world_model.dtype]
        model_cfg = read_config(self.audit_cfg.paths.wm_inference_config)
        self.vae = build_module(model_cfg.vae, MODELS).to(self.device, self.dtype).eval()
        self.image_size = model_cfg.get("image_size") or get_image_size(
            model_cfg.resolution, model_cfg.aspect_ratio
        )
        num_frames = get_num_frames(model_cfg.num_frames)
        self.latent_size = self.vae.get_latent_size((num_frames, *self.image_size))
        self.model = build_module(
            model_cfg.model, MODELS, input_size=self.latent_size, in_channels=self.vae.out_channels
        ).to(self.device, self.dtype).eval()
        for module in (self.vae, self.model):
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        self.scheduler = build_module(model_cfg.scheduler, SCHEDULERS)
        self.model_args = prepare_multi_resolution_info(
            model_cfg.get("multi_resolution"), 1, self.image_size, num_frames,
            model_cfg.fps, self.device, self.dtype,
        )

    def _require_loaded(self):
        if self.vae is None or self.model is None:
            raise RuntimeError("Call StandaloneWorldModel.load() before inference")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(frame, dtype=np.uint8)).convert("RGB")
        height, width = int(self.image_size[0]), int(self.image_size[1])
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def predict_branch(
        self,
        start_frame: np.ndarray,
        normalized_first_action: np.ndarray,
        *,
        instruction: str,
        noise_seed: int,
        continuation_seed: int,
    ) -> np.ndarray:
        """Predict start frame plus H chunks, matching `_rollout_dccp_counterfactual_branch`."""
        self._require_loaded()
        import torch

        chunk = int(self.audit_cfg.policy.action_chunk_length)
        action_dim = int(self.audit_cfg.policy.action_dim)
        current_frame = self._resize(start_frame)
        start_tensor = torch.from_numpy(current_frame.copy()).permute(2, 0, 1).float()
        start_tensor = (start_tensor / 255.0 * 2.0 - 1.0).to(self.device, self.dtype)
        with torch.inference_mode():
            history = self.vae.encode(start_tensor.unsqueeze(0).unsqueeze(2))
            history = history.repeat(1, 1, int(self.audit_cfg.world_model.queue_len), 1, 1)
            output = [current_frame[None]]
            for local_step in range(int(self.audit_cfg.world_model.horizon_H)):
                if local_step == 0:
                    actions = np.asarray(normalized_first_action, dtype=np.float32).reshape(chunk, action_dim)
                else:
                    action = self.policy.generate_action(
                        current_frame,
                        instruction,
                        seed=derive_seed(continuation_seed, "vla", local_step),
                        do_sample=self.audit_cfg.policy.nominal_do_sample,
                        temperature=self.audit_cfg.policy.nominal_temperature,
                        compute_entropy=False,
                    )
                    actions = np.asarray(action.normalized_actions, dtype=np.float32).reshape(chunk, action_dim)
                y = torch.from_numpy(actions).to(self.device, self.dtype).reshape(1, chunk, action_dim)
                # Reset per step so controlled mode supplies identical diffusion noise to all candidates.
                seed_everything(derive_seed(noise_seed, "diffusion", local_step))
                z = torch.randn(
                    1, self.vae.out_channels, chunk, *self.latent_size[1:],
                    device=self.device, dtype=self.dtype,
                )
                combined = torch.cat([history, z], dim=2)
                mask = torch.zeros(1, history.shape[2] + chunk, device=self.device, dtype=torch.long)
                mask[:, -chunk:] = 1
                samples = self.scheduler.sample(
                    self.model, z=combined, y=y, device=self.device,
                    additional_args=self.model_args, progress=False, mask=mask,
                )
                predicted_latents = samples[:, :, -chunk:].to(self.dtype)
                history = predicted_latents[:, :, -int(self.audit_cfg.world_model.queue_len):].clone()
                decoded = self.vae.decode(predicted_latents)
                frames = ((decoded.float().cpu().permute(0, 2, 3, 4, 1).numpy() * 0.5 + 0.5) * 255.0)
                frames = np.clip(frames, 0, 255).astype(np.uint8)[0]
                output.append(frames)
                current_frame = frames[-1]
        return np.concatenate(output, axis=0)
