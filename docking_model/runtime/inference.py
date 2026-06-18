from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from docking_model.config.loader import load_config, merge_overrides
from docking_model.config.serialization import config_to_dict
from docking_model.config.schema import DockingConfig


def load_inference_config(
    model_parameters_path: str | Path,
    overrides: Mapping[str, Any] | Iterable[str] | None = None,
) -> DockingConfig:
    """Load training config from model_parameters.yml and apply inference overrides."""

    source_path = str(Path(model_parameters_path).expanduser())
    cfg = load_config(model_parameters_path)
    if overrides is None:
        return cfg

    raw = config_to_dict(cfg)
    merged = DockingConfig.from_dict(merge_overrides(raw, overrides))
    merged.source_path = source_path
    return merged
