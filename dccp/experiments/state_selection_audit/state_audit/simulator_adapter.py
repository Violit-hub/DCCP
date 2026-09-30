"""Coffee robosuite 环境、快照和恢复适配器。"""

from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from .config import SimulatorConfig


class CoffeeSimulatorAdapter:
    def __init__(self, dataset_config: str | Path, config: SimulatorConfig):
        self.dataset_config = Path(dataset_config).expanduser().resolve()
        self.config = config
        self.ext_cfg = json.loads(self.dataset_config.read_text(encoding="utf-8"))

    def create_env(self):
        from robomimic.config import config_factory
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.obs_utils as ObsUtils
        import mimicgen.envs.robosuite  # noqa: F401

        cfg = config_factory(self.ext_cfg["algo_name"])
        with cfg.values_unlocked():
            cfg.update(self.ext_cfg)
        cfg.lock()
        ObsUtils.initialize_obs_utils_with_config(cfg)
        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=cfg.train.data)
        shape_meta = FileUtils.get_shape_metadata_from_dataset(
            dataset_path=cfg.train.data,
            all_obs_keys=cfg.all_obs_keys,
            verbose=False,
        )
        if cfg.experiment.env is not None:
            env_meta["env_name"] = cfg.experiment.env
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=True,
            use_image_obs=shape_meta["use_images"],
            use_depth_obs=shape_meta["use_depths"],
        )
        return EnvUtils.wrap_env_from_config(env, config=cfg)

    @staticmethod
    def close_env(env) -> None:
        inner = getattr(env, "env", env)
        close = getattr(inner, "close", None)
        if callable(close):
            close()

    @staticmethod
    def snapshot(env) -> tuple[str, np.ndarray]:
        state = env.get_state()
        return str(state["model"]), np.asarray(state["states"]).copy()

    @staticmethod
    def restore(env, model_xml: str, simulator_state: np.ndarray):
        return env.reset_to({"model": model_xml, "states": np.asarray(simulator_state)})

    def warm_up(self, env, obs=None):
        for _ in range(int(self.config.num_steps_wait)):
            obs, _, _, _ = env.step(np.zeros(self.config_action_dim, dtype=np.float32))
        return obs

    @property
    def config_action_dim(self) -> int:
        return 7

    @staticmethod
    def observation_image(obs: dict[str, Any], camera_key: str = "agentview_image") -> np.ndarray:
        image = np.asarray(obs[camera_key])
        if image.ndim != 3:
            raise ValueError(f"相机图像必须是 3D，收到 {image.shape}")
        if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
            image = image.transpose(1, 2, 0)
        if np.issubdtype(image.dtype, np.floating):
            if float(np.nanmax(image)) <= 1.0 + 1e-6:
                image = image * 255.0
        return np.clip(image, 0, 255).astype(np.uint8)

    def step_action_chunk(
        self,
        env,
        actions: np.ndarray,
        remaining_steps: int,
        *,
        frame_callback=None,
    ):
        obs = None
        success = False
        executed = 0
        for action in np.asarray(actions, dtype=np.float32):
            if executed >= int(remaining_steps):
                break
            obs, reward, _, _ = env.step(action.tolist())
            executed += 1
            if frame_callback is not None:
                frame_callback(self.observation_image(obs, self.config.camera_key))
            if float(reward) > float(self.config.success_reward_threshold):
                success = True
                break
        return obs, success, executed


def load_initial_states(path: str | Path, *, trust_pickle: bool) -> list[Any]:
    """读取项目已有初始状态；必须显式确认 pickle 信任边界。"""
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() not in {".pkl", ".pickle"}:
        raise ValueError("当前 initial state loader 只支持项目已有的 pickle 文件")
    if not trust_pickle:
        raise PermissionError(
            "pickle 反序列化可能执行代码。确认该文件来自可信项目后，"
            "在配置中设置 pilot.trust_initial_states_pickle=true"
        )
    warnings.warn(f"正在反序列化可信项目 pickle: {source}", RuntimeWarning, stacklevel=2)
    with source.open("rb") as handle:
        states = pickle.load(handle)
    if not isinstance(states, (list, tuple)) or not states:
        raise ValueError("初始状态文件必须包含非空 list/tuple")
    return list(states)
