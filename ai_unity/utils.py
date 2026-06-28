from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def default_cuda_visible_devices() -> None:
    """Reserve GPU 0 for this project unless the caller already chose a GPU."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(device: str = "auto", gpu_index: int = 0) -> torch.device:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unknown device '{device}'. Use auto, cpu, or cuda.")
    if device == "cpu":
        return torch.device("cpu")
    if device in {"auto", "cuda"} and torch.cuda.is_available():
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"gpu_index={gpu_index} is invalid; visible CUDA devices: {torch.cuda.device_count()}"
            )
        torch.cuda.set_device(gpu_index)
        return torch.device(f"cuda:{gpu_index}")
    if device == "cuda":
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def model_summary(model: nn.Module) -> int:
    params = count_params(model)
    name = getattr(model, "name", model.__class__.__name__)
    print(f"  {name}: {params:,} parameters")
    if hasattr(model, "sparsity"):
        print(f"    Weight sparsity: {model.sparsity():.1%}")
    return params


def output_tuple(output: Any, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Normalize model outputs to logits, auxiliary loss, and diagnostics."""
    if isinstance(output, tuple):
        logits = output[0]
        extra = output[1] if len(output) > 1 else None
    else:
        logits = output
        extra = None

    aux_loss = torch.zeros((), device=device)
    info: dict[str, Any] = {}

    if isinstance(extra, torch.Tensor):
        aux_loss = extra.to(device=device)
    elif isinstance(extra, dict):
        info = extra
        candidate = extra.get("aux_loss")
        if isinstance(candidate, torch.Tensor):
            aux_loss = candidate.to(device=device)
    elif extra is not None:
        raise TypeError(f"Unsupported model auxiliary output: {type(extra)!r}")

    return logits, aux_loss, info


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def json_safe(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2)
