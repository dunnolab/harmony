from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import yaml


def config_to_dict(config: Any) -> dict[str, Any]:
    if is_dataclass(config):
        value = asdict(config)
    elif isinstance(config, dict):
        value = dict(config)
    else:
        value = dict(getattr(config, "__dict__", {}))
    return plain(value)


def save_config(config: Any, path: str | Path) -> None:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)


def plain(value: Any) -> Any:
    if is_dataclass(value):
        return plain(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value
