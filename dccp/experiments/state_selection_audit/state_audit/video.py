"""使用项目环境已有的 imageio-ffmpeg 原子写入 MP4。"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


class BranchVideoRecorder:
    """逐帧编码，避免一个长 rollout 的全部图像常驻内存。"""

    def __init__(self, path: str | Path, *, fps: int = 20):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(f"{self.path.stem}.tmp{self.path.suffix}")
        self.writer = imageio.get_writer(
            str(self.temporary),
            fps=int(fps),
            codec="libx264",
            quality=7,
            macro_block_size=1,
        )
        self.frame_count = 0

    def append(self, frame: np.ndarray) -> None:
        image = np.asarray(frame)
        if np.issubdtype(image.dtype, np.floating):
            if float(np.nanmax(image)) <= 1.0 + 1e-6:
                image = image * 255.0
        self.writer.append_data(np.clip(image, 0, 255).astype(np.uint8))
        self.frame_count += 1

    def close(self, *, commit: bool = True) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        if commit and self.frame_count > 0:
            self.temporary.replace(self.path)
            self.path.chmod(0o660)
        elif self.temporary.exists():
            self.temporary.unlink()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(commit=exc_type is None)
        return False
