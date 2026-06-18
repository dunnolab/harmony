from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import torch

from docking_model.config.schema import DockingConfig
from docking_model.data.feature.featurizer import Featurizer, FeaturizerConfig
from docking_model.data.parse.parser import ComplexParser


def ensure_preprocessed_cache(cfg: DockingConfig) -> None:
    if not cfg.data.preprocess_raw:
        return
    if cfg.data.input_csv is None:
        raise ValueError("data.preprocess_raw=true requires data.input_csv.")
    if cfg.data.cache_path is None:
        raise ValueError("data.preprocess_raw=true requires data.cache_path.")

    input_csv = Path(cfg.data.input_csv).expanduser()
    cache_path = Path(cfg.data.cache_path).expanduser()
    cache_path.mkdir(parents=True, exist_ok=True)

    rows = load_rows(input_csv)
    if cfg.data.limit_complexes:
        rows = rows[: cfg.data.limit_complexes]

    parser = ComplexParser(esm_embeddings_path=cfg.model.esm_embeddings_path)
    featurizer = Featurizer.from_config(
        FeaturizerConfig(
            matching=cfg.data.matching,
            popsize=cfg.data.matching_popsize,
            maxiter=cfg.data.matching_maxiter,
            keep_original=cfg.data.keep_original,
            remove_hs=cfg.data.remove_hs,
            num_conformers=cfg.data.num_conformers,
            max_lig_size=cfg.data.max_lig_size,
            all_atoms=cfg.pocket.all_atoms,
            flexible_backbone=cfg.protein.flexible_backbone,
            flexible_sidechains=cfg.protein.flexible_sidechains,
        )
    )

    processed_names: list[str] = []
    split_names: dict[str, list[str]] = {"train": [], "val": []}
    for row in rows:
        input_dict = row_to_complex_input(row)
        name = input_dict["name"]
        if (cfg.protein.flexible_backbone or cfg.protein.flexible_sidechains) and input_dict["holo_rec_path"] is None:
            raise ValueError(f"{name}: flexible raw preprocessing requires a holo protein path.")

        output_file = cache_path / f"heterograph-{name}.pt"
        if output_file.exists():
            processed_names.append(name)
            record_split(split_names, row, name)
            continue

        try:
            complex_inputs = parser.parse_complex(input_dict)
            features = featurizer.featurize_complex(complex_inputs)
        except Exception as exc:
            logging.exception("%s: preprocessing failed due to %s", name, exc)
            continue
        if features is None:
            logging.warning("%s: preprocessing produced no graph", name)
            continue

        graph = features["complex_graph"]
        for key, value in input_dict.items():
            if value is not None and key not in {"ligand"}:
                graph[key] = str(value)
        torch.save(graph, output_file)
        processed_names.append(name)
        record_split(split_names, row, name)

    if not processed_names:
        raise ValueError(f"No complexes were preprocessed from {input_csv}.")

    ensure_split_files(cfg, cache_path, processed_names, split_names)


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def row_to_complex_input(row: dict[str, Any]) -> dict[str, Any]:
    base_dir = optional_csv_value(row, "base_dir")
    name = required_csv_value(row, "pdbid", "name", "complex_name")
    ligand_input = required_csv_value(row, "ligand_input", "ligand_file", "ligand")
    ligand_description = optional_csv_value(row, "ligand_description") or "filename"
    apo_rec_path = resolve_csv_path(
        base_dir,
        required_csv_value(row, "apo_protein_file", "apo_rec_path", "protein_file"),
    )
    holo_rec_path = resolve_csv_path(
        base_dir,
        optional_csv_value(row, "holo_protein_file", "holo_rec_path", "match_protein_file"),
    )
    return {
        "name": name,
        "base_dir": base_dir,
        "apo_rec_path": apo_rec_path,
        "holo_rec_path": holo_rec_path,
        "holo_rec_path_for_metrics": holo_rec_path,
        "ligand_input": ligand_input,
        "ligand_true_file": optional_csv_value(row, "ligand_true_file"),
        "pocket_ligand_file": optional_csv_value(row, "pocket_ligand_file"),
        "ligand_description": ligand_description,
    }


def required_csv_value(row: dict[str, Any], *keys: str) -> str:
    value = optional_csv_value(row, *keys)
    if value is None:
        raise ValueError(f"Input CSV row is missing one of {keys}: {row}")
    return value


def optional_csv_value(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value and value.lower() not in {"none", "nan", "null"}:
            return value
    return None


def resolve_csv_path(base_dir: str | None, value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((Path(base_dir).expanduser() / path).resolve())


def record_split(split_names: dict[str, list[str]], row: dict[str, Any], name: str) -> None:
    split = (optional_csv_value(row, "split", "subset") or "").lower()
    if split in split_names:
        split_names[split].append(name)


def ensure_split_files(
    cfg: DockingConfig,
    cache_path: Path,
    processed_names: list[str],
    split_names: dict[str, list[str]],
) -> None:
    if cfg.data.split_train is None:
        train_names = split_names["train"] or processed_names
        cfg.data.split_train = str(cache_path / "split_train.txt")
        write_split(Path(cfg.data.split_train), train_names)
    if cfg.data.split_val is None and split_names["val"]:
        cfg.data.split_val = str(cache_path / "split_val.txt")
        write_split(Path(cfg.data.split_val), split_names["val"])


def write_split(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n")
