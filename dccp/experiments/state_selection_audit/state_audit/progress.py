"""Progress LRM 健康检查与 nominal 进度评分。"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def load_progress_scorer(config_path: str | Path, *, cache_dir: str | Path | None = None):
    from verl.utils.dccp_scorer import DCCPScorer

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cache_dir is not None:
        raw.setdefault("scorer", {})["cache_dir"] = str(Path(cache_dir).resolve())
    return DCCPScorer.from_config_dict(raw)


def check_progress_service(config_path: str | Path, timeout_sec: float) -> dict[str, Any]:
    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    endpoint = str(raw["scorer"]["progress_endpoint"])
    health_endpoint = endpoint.rsplit("/", 1)[0] + "/health"
    request = urllib.request.Request(health_endpoint, method="GET")
    host = urllib.parse.urlparse(health_endpoint).hostname
    if host in {"127.0.0.1", "localhost", "::1"}:
        # 本机服务不能经过用户 shell 中配置的 HTTP(S)_PROXY。
        open_request = urllib.request.build_opener(
            urllib.request.ProxyHandler({})
        ).open
    else:
        open_request = urllib.request.urlopen
    with open_request(request, timeout=float(timeout_sec)) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "endpoint": endpoint,
        "health_endpoint": health_endpoint,
        "status": int(response.status),
        "payload": payload,
    }


def score_nominal_progress(
    frames: np.ndarray,
    instruction: str,
    scorer,
    *,
    horizon_H: int,
    trajectory_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    """以 VLA 决策帧为单位评分，frames_per_action 固定为 1。"""
    from verl.utils.dccp_mining import compute_nominal_progress_scores

    starts, progress = compute_nominal_progress_scores(
        rollout_video=np.asarray(frames),
        instruction=instruction,
        scorer=scorer,
        horizon_H=int(horizon_H),
        frames_per_action=1,
        decision_indices=list(range(len(frames))),
        metadata={"trajectory_id": trajectory_id, "audit": "state_selection"},
    )
    return starts.astype(np.int64), progress.astype(np.float32)
