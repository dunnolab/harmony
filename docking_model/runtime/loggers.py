from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from docking_model.config.serialization import config_to_dict
from docking_model.runtime.distributed import is_main_process


@dataclass
class NullExperimentLogger:
    run_id: str | None = None
    wandb: Any = None

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del metrics, step

    def log_artifacts(self, artifacts: dict[str, Any]) -> None:
        del artifacts

    def finish(self) -> None:
        return


class WandbExperimentLogger:
    def __init__(self, cfg, *, job_type: str):
        try:
            import wandb
        except ImportError as exc:
            raise ImportError("logger.wandb=true requires the 'wandb' package to be installed.") from exc

        logger_cfg = cfg.logger
        name = logger_cfg.name or cfg.run_name
        run_id = logger_cfg.run_id or os.environ.get("WANDB_RUN_ID")

        init_kwargs = {
            "entity": logger_cfg.entity,
            "project": logger_cfg.project,
            "name": name,
            "tags": list(logger_cfg.tags or []),
            "group": logger_cfg.group,
            "mode": logger_cfg.mode,
            "dir": logger_cfg.dir,
            "job_type": job_type,
            "config": config_to_dict(cfg) if logger_cfg.log_config else None,
        }
        if run_id is not None:
            init_kwargs["id"] = run_id
            init_kwargs["resume"] = logger_cfg.resume

        run = wandb.init(**init_kwargs)
        self._wandb = wandb
        self.wandb = wandb
        self.run = run
        self.run_id = getattr(run, "id", None)

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        cleaned = {
            key: value
            for key, value in (metric_item(item) for item in metrics.items())
            if value is not None
        }
        if cleaned:
            self._wandb.log(cleaned, step=step)

    def log_artifacts(self, artifacts: dict[str, Any]) -> None:
        if artifacts:
            self._wandb.log(artifacts)

    def finish(self) -> None:
        self._wandb.finish()


def build_experiment_logger(cfg, *, job_type: str):
    if not is_main_process() or not bool(getattr(cfg.logger, "wandb", False)):
        return NullExperimentLogger()
    return WandbExperimentLogger(cfg, job_type=job_type)


def metric_item(item: tuple[str, Any]) -> tuple[str, Any | None]:
    key, value = item
    if torch.is_tensor(value):
        if value.numel() != 1:
            return key, None
        value = value.detach().cpu().item()
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return key, None
        if value.size == 1:
            value = value.reshape(-1)[0].item()
        else:
            values = value.astype(float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                return key, None
            value = float(values.mean())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return key, float(value)
    if isinstance(value, int):
        return key, value
    if isinstance(value, float):
        return key, value if math.isfinite(value) else None
    return key, None
