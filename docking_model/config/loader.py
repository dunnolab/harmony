from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping
import re

import yaml

from docking_model.config.schema import DockingConfig


def load_config(path: str | Path, overrides: Mapping[str, Any] | Iterable[str] | None = None) -> DockingConfig:
    """Load a YAML config and apply optional nested or dotlist overrides."""

    config_path = Path(path).expanduser()
    raw = load_yaml_mapping(config_path)

    merged = merge_overrides(raw, overrides)
    cfg = DockingConfig.from_dict(merged)
    cfg.source_path = str(config_path)
    return cfg


def merge_overrides(
    base: Mapping[str, Any],
    overrides: Mapping[str, Any] | Iterable[str] | None,
) -> dict[str, Any]:
    merged = deepcopy(dict(base))
    if overrides is None:
        return merged

    if isinstance(overrides, Mapping):
        return deep_merge(merged, dict(overrides))

    for item in overrides:
        key, value = parse_dotlist_item(item)
        set_nested(merged, key.split("."), value)
    return merged


def deep_merge(base: dict[str, Any], update: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            base[key] = deep_merge(dict(base[key]), value)
        else:
            base[key] = deepcopy(value)
    return base


def parse_dotlist_item(item: str) -> tuple[str, Any]:
    if "=" not in item:
        raise ValueError(f"Override must use key=value syntax, got {item!r}.")
    key, raw_value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise ValueError("Override key cannot be empty.")
    return key, yaml.safe_load(raw_value)


def set_nested(target: dict[str, Any], path: list[str], value: Any) -> None:
    cursor = target
    for key in path[:-1]:
        existing = cursor.get(key)
        if not isinstance(existing, dict):
            existing = {}
            cursor[key] = existing
        cursor = existing
    cursor[path[-1]] = value


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        from omegaconf import OmegaConf

        cfg = OmegaConf.load(path)
        return dict(OmegaConf.to_container(cfg, resolve=True) or {})
    except Exception:
        with path.open("r") as handle:
            raw = yaml.safe_load(handle) or {}
        return resolve_interpolations(raw)


_INTERPOLATION_RE = re.compile(r"\$\{([^}]+)\}")


def resolve_interpolations(raw: dict[str, Any]) -> dict[str, Any]:
    root = deepcopy(raw)

    def resolve(value: Any, stack: tuple[str, ...] = ()) -> Any:
        if isinstance(value, dict):
            return {key: resolve(item, stack + (str(key),)) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item, stack + (str(idx),)) for idx, item in enumerate(value)]
        if not isinstance(value, str):
            return value

        full = _INTERPOLATION_RE.fullmatch(value.strip())
        if full:
            looked_up = lookup(root, full.group(1), stack)
            if looked_up == value:
                return value
            return resolve(looked_up, stack)

        def replace(match: re.Match[str]) -> str:
            looked_up = lookup(root, match.group(1), stack)
            resolved = looked_up if looked_up == match.group(0) else resolve(looked_up, stack)
            return str(resolved)

        return _INTERPOLATION_RE.sub(replace, value)

    return resolve(root)


def lookup(root: Mapping[str, Any], expression: str, stack: tuple[str, ...]) -> Any:
    if expression.startswith("not:"):
        return not bool(lookup(root, expression[len("not:") :], stack))

    path = tuple(part for part in expression.split(".") if part)
    if path in {stack[: len(path)], stack}:
        raise ValueError(f"Self-referential config interpolation: {expression}")

    cursor: Any = root
    for part in path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            return "${" + expression + "}"
        cursor = cursor[part]
    return cursor
