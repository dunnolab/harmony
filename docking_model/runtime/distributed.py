from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def setup_distributed(timeout_hours: int = 5) -> bool:
    if not dist.is_available() or int(os.environ.get("WORLD_SIZE", "1")) <= 1:
        return False
    if dist.is_initialized():
        return True

    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "nccl":
        torch.cuda.set_device(get_local_rank())

    dist.init_process_group(
        backend=backend,
        init_method="env://",
        timeout=timedelta(hours=timeout_hours),
    )
    return True


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_rank() -> int:
    if is_distributed():
        return dist.get_rank()
    return int(os.environ.get("RANK", "0"))


def get_local_rank() -> int:
    for key in ("LOCAL_RANK", "SLURM_LOCALID", "OMPI_COMM_WORLD_LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            return int(value)
    return 0


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def any_rank_has(value: bool, device: torch.device | str | None = None) -> bool:
    if not is_distributed():
        return bool(value)
    tensor_device = torch.device(device) if device is not None else torch.device("cpu")
    flag = torch.tensor([1 if value else 0], device=tensor_device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    return bool(flag.item())


def wrap_model_for_distributed(
    model: torch.nn.Module,
    device: torch.device,
    *,
    find_unused_parameters: bool = True,
) -> torch.nn.Module:
    if not is_distributed():
        return model

    from torch.nn.parallel import DistributedDataParallel

    kwargs: dict[str, Any] = {
        "broadcast_buffers": False,
        "find_unused_parameters": find_unused_parameters,
    }
    if device.type == "cuda":
        device_index = device.index if device.index is not None else get_local_rank()
        kwargs["device_ids"] = [device_index]
        kwargs["output_device"] = device_index

    return DistributedDataParallel(model, **kwargs)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def all_gather_object(value: Any) -> list[Any]:
    if not is_distributed():
        return [value]
    values: list[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(values, value)
    return values


def broadcast_object(value: Any, src: int = 0) -> Any:
    if not is_distributed():
        return value
    values = [value if get_rank() == src else None]
    dist.broadcast_object_list(values, src=src)
    return values[0]
