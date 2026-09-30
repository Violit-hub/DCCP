#!/usr/bin/env python3
"""Merge a PEFT LoRA adapter into a full OpenVLA / HuggingFace model."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForVision2Seq


def _copy_support_files(base_model_path: Path, output_path: Path) -> None:
    """Copy tokenizer / processor / custom-code files that save_pretrained may omit."""
    skip_prefixes = ("model-",)
    skip_names = {
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "adapter_model.safetensors",
        "adapter_config.json",
    }

    for src in base_model_path.iterdir():
        if not src.is_file():
            continue
        if src.name in skip_names:
            continue
        if src.name.startswith(skip_prefixes) and src.suffix == ".safetensors":
            continue
        dst = output_path / src.name
        if not dst.exists():
            shutil.copy2(src, dst)


def _read_base_model_from_adapter(adapter_path: Path) -> Path:
    config_path = adapter_path / "adapter_config.json"
    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    base = cfg.get("base_model_name_or_path")
    if not base:
        raise ValueError(f"{config_path} does not contain base_model_name_or_path")
    return Path(base)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True, type=Path)
    parser.add_argument("--base-model-path", default=None, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    parser.add_argument("--max-shard-size", default="5GB")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    adapter_path = args.adapter_path.resolve()
    base_model_path = (args.base_model_path or _read_base_model_from_adapter(adapter_path)).resolve()
    output_path = args.output_path.resolve()

    if not (adapter_path / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing adapter_config.json under {adapter_path}")
    if not base_model_path.is_dir():
        raise FileNotFoundError(f"Base model path does not exist: {base_model_path}")
    if output_path.exists() and any(output_path.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output path is not empty: {output_path}. Pass --overwrite to replace files.")

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[merge] base model: {base_model_path}", flush=True)
    print(f"[merge] adapter:    {adapter_path}", flush=True)
    print(f"[merge] output:     {output_path}", flush=True)

    model = AutoModelForVision2Seq.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(model, str(adapter_path))
    model = model.merge_and_unload()

    model.save_pretrained(
        str(output_path),
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
    )
    _copy_support_files(base_model_path, output_path)

    print("[merge] done", flush=True)


if __name__ == "__main__":
    main()
