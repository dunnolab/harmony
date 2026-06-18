from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    ema: Any | None = None,
    ema_weights: dict[str, torch.Tensor] | None = None,
    **metadata: Any,
) -> None:
    payload = {"state_dict": model.state_dict(), **metadata}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
    if ema is not None:
        payload["ema"] = ema.state_dict()
    if ema_weights is not None:
        payload["ema_weights"] = ema_weights
    torch.save(payload, Path(path).expanduser())


def load_model_state(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    strict: bool = True,
    state_dict_key: str = "state_dict",
    prefer_ema: bool = False,
):
    checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu")
    if prefer_ema and isinstance(checkpoint, dict) and "ema_weights" in checkpoint:
        state_dict = checkpoint["ema_weights"]
    else:
        state_dict = checkpoint.get(state_dict_key, checkpoint)
    state_dict = strip_known_prefixes(state_dict)
    if not strict:
        return model.load_state_dict(state_dict, strict=False)

    result = model.load_state_dict(state_dict, strict=False)
    missing = [key for key in result.missing_keys if not tensor_product_backend_key(key)]
    unexpected = [key for key in result.unexpected_keys if not tensor_product_backend_key(key)]
    if missing or unexpected:
        raise RuntimeError(state_dict_error_message(model, missing, unexpected))
    return result


def strip_known_prefixes(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            key = key[len("model.") :]
        cleaned[key] = value
    return cleaned


def tensor_product_backend_key(key: str) -> bool:
    return key.startswith("tp.") or ".tp." in key


def state_dict_error_message(
    model: torch.nn.Module,
    missing: list[str],
    unexpected: list[str],
) -> str:
    lines = [f"Error(s) in loading state_dict for {model.__class__.__name__}:"]
    if missing:
        lines.append(f"\tMissing key(s) in state_dict: {quoted_keys(missing)}.")
    if unexpected:
        lines.append(f"\tUnexpected key(s) in state_dict: {quoted_keys(unexpected)}.")
    return "\n".join(lines)


def quoted_keys(keys: list[str]) -> str:
    return ", ".join(f'"{key}"' for key in keys)
