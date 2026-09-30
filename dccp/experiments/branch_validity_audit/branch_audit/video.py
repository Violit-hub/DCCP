"""Atomic MP4 writing and reading helpers."""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    image = np.asarray(frame)
    if np.issubdtype(image.dtype, np.floating) and image.size and float(np.nanmax(image)) <= 1.0 + 1e-6:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def write_video(path: str | Path, frames: np.ndarray, *, fps: int) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.stem}.tmp{target.suffix}")
    writer = imageio.get_writer(str(temporary), fps=int(fps), codec="libx264", quality=7, macro_block_size=1)
    count = 0
    try:
        for frame in np.asarray(frames):
            writer.append_data(normalize_frame(frame))
            count += 1
    finally:
        writer.close()
    if count == 0:
        temporary.unlink(missing_ok=True)
        raise ValueError("Refusing to write an empty video")
    temporary.replace(target)
    target.chmod(0o660)


class StreamingVideoRecorder:
    def __init__(self, path: str | Path, *, fps: int):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(f"{self.path.stem}.tmp{self.path.suffix}")
        self.writer = imageio.get_writer(str(self.temporary), fps=int(fps), codec="libx264", quality=7, macro_block_size=1)
        self.count = 0

    def append(self, frame: np.ndarray) -> None:
        self.writer.append_data(normalize_frame(frame))
        self.count += 1

    def close(self, commit: bool = True) -> None:
        if self.writer is None:
            return
        self.writer.close()
        self.writer = None
        if commit and self.count:
            self.temporary.replace(self.path)
            self.path.chmod(0o660)
        else:
            self.temporary.unlink(missing_ok=True)
