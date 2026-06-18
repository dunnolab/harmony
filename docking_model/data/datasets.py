from __future__ import annotations

import csv
import logging
import math
from pathlib import Path
from typing import Callable, Sequence

import torch
from torch_geometric.data import Dataset


class ListDataset(Dataset):
    def __init__(self, items: Sequence, transform: Callable | None = None):
        super().__init__(root=None, transform=transform)
        self.items = list(items)

    def len(self) -> int:
        return len(self.items)

    def get(self, idx: int):
        return self.items[idx]


class CachedComplexDataset(Dataset):
    """Loads one processed heterograph per complex from a cache directory."""

    def __init__(
        self,
        cache_path: str | Path,
        split_path: str | Path,
        transform: Callable | None = None,
        affinity_csv: str | Path | None = None,
        esm_embeddings_path: str | Path | None = None,
        limit_complexes: int | None = None,
        multiplicity: int = 1,
    ):
        super().__init__(root=None, transform=transform)
        self.cache_path = Path(cache_path).expanduser()
        self.split_path = Path(split_path).expanduser()
        self.limit_complexes = limit_complexes
        self.multiplicity = max(int(multiplicity), 1)
        self.affinity_map = load_affinity_map(affinity_csv)
        self.metadata_map = load_metadata_map(affinity_csv)
        self.esm_embeddings_path = (
            Path(esm_embeddings_path).expanduser()
            if esm_embeddings_path is not None
            else None
        )
        self.complex_files = self.collect_complex_files()

    def len(self) -> int:
        return len(self.complex_files) * self.multiplicity

    def get(self, idx: int):
        idx = idx % len(self.complex_files)
        data = torch.load(self.cache_path / self.complex_files[idx], weights_only=False)
        data = attach_esm_embeddings(fix_0d_tensors(data), self.esm_embeddings_path)
        data = attach_affinity(data, self.affinity_map)
        return attach_metadata(data, self.metadata_map)

    def collect_complex_files(self) -> list[str]:
        if not self.cache_path.exists():
            raise FileNotFoundError(f"Cache directory does not exist: {self.cache_path}")
        if not self.split_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {self.split_path}")

        names = [
            line.strip()
            for line in self.split_path.read_text().splitlines()
            if line.strip()
        ]
        if self.limit_complexes:
            names = names[: self.limit_complexes]

        files = []
        for name in names:
            filename = f"heterograph-{name}.pt"
            if (self.cache_path / filename).exists():
                files.append(filename)

        if not files:
            raise ValueError(
                f"No cached complexes from {self.split_path} were found in {self.cache_path}."
            )
        return files


def load_affinity_map(path: str | Path | None) -> dict[str, tuple[float, bool]] | None:
    if path is None:
        return None

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        logging.warning("Affinity CSV not found: %s", csv_path)
        return None

    values: dict[str, tuple[float, bool]] = {}
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = (row.get("pdb_id") or "").strip().lower()
            if not pdb_id:
                continue
            try:
                p_value = float(row.get("p_value", ""))
            except ValueError:
                continue
            if not math.isfinite(p_value):
                continue
            p_type = (row.get("p_type") or "").strip().lower()
            affinity_type = (row.get("affinity_type") or "").strip().lower()
            is_ic50 = p_type == "pic50" or affinity_type == "ic50"
            values[pdb_id] = (p_value, is_ic50)
    return values


def load_metadata_map(path: str | Path | None) -> dict[str, dict[str, str]] | None:
    if path is None:
        return None

    csv_path = Path(path).expanduser()
    if not csv_path.exists():
        return None

    values: dict[str, dict[str, str]] = {}
    with csv_path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            pdb_id = csv_value(row, "pdb_id", "pdbid", "name", "complex_name")
            if not pdb_id:
                continue

            metadata: dict[str, str] = {}
            ligand_path = resolve_csv_path(
                row,
                csv_value(
                    row,
                    "ligand_sdf",
                    "ligand_mol2",
                    "ligand_input",
                    "ligand_file",
                    "ligand",
                ),
            )
            protein_path = resolve_csv_path(
                row,
                csv_value(
                    row,
                    "protein_pdb",
                    "holo_protein_file",
                    "holo_rec_path",
                    "holo_path",
                ),
            )

            if ligand_path:
                metadata["ligand_input"] = ligand_path
                metadata["ligand_true_file"] = ligand_path
                metadata["ligand_description"] = "filename"
                metadata["base_dir"] = str(Path(ligand_path).expanduser().parent)
            if protein_path:
                metadata["holo_rec_path_for_metrics"] = protein_path
                metadata.setdefault(
                    "base_dir",
                    str(Path(protein_path).expanduser().parent),
                )

            if metadata:
                values[pdb_id.strip().lower()] = metadata
    return values


def csv_value(row: dict[str, str], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def resolve_csv_path(row: dict[str, str], value: str | None) -> str | None:
    if value is None:
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    base_dir = csv_value(row, "base_dir")
    if base_dir:
        return str((Path(base_dir).expanduser() / path).resolve())
    return str(path)


def attach_affinity(data, affinity_map: dict[str, tuple[float, bool]] | None):
    if affinity_map is None:
        return data

    name = normalize_name(data["name"] if "name" in data else None)
    entry = affinity_map.get(name)
    if entry is None:
        data.affinity = torch.tensor([0.0], dtype=torch.float32)
        data.affinity_mask = torch.tensor([0.0], dtype=torch.float32)
        return data

    affinity_value, is_ic50 = entry
    data.affinity = torch.tensor([affinity_value], dtype=torch.float32)
    data.affinity_mask = torch.tensor([0.0 if is_ic50 else 1.0], dtype=torch.float32)
    return data


def attach_metadata(data, metadata_map: dict[str, dict[str, str]] | None):
    if metadata_map is None:
        return data

    name = normalize_name(data["name"] if "name" in data else None)
    entry = metadata_map.get(name)
    if entry is None:
        return data

    for key, value in entry.items():
        if metadata_missing(data, key):
            data[key] = value
    return data


def metadata_missing(data, key: str) -> bool:
    if not hasattr(data, "_global_store") or key not in data._global_store:
        return True

    value = data._global_store[key]
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, (list, tuple)):
        return len(value) == 0
    if torch.is_tensor(value):
        return value.numel() == 0
    return False


def attach_esm_embeddings(data, esm_embeddings_path: Path | None):
    if esm_embeddings_path is None:
        return data
    name = normalize_name(data["name"] if "name" in data else None)
    if not name:
        return data
    path = esm_embeddings_path / f"{name}.pt"
    if not path.exists():
        logging.warning("ESM embedding file not found for %s: %s", name, path)
        return data

    embeddings = torch.load(path, weights_only=False)
    if isinstance(embeddings, dict):
        embeddings = torch.cat([embeddings[key] for key in sorted(embeddings)], dim=0)
    if not torch.is_floating_point(data["receptor"].x):
        data["receptor"].x = data["receptor"].x.float()
    embeddings = torch.as_tensor(embeddings, dtype=data["receptor"].x.dtype, device=data["receptor"].x.device)
    if embeddings.shape[0] != data["receptor"].x.shape[0]:
        logging.warning(
            "Skipping ESM embeddings for %s because residue counts differ: embeddings=%s receptor=%s",
            name,
            embeddings.shape[0],
            data["receptor"].x.shape[0],
        )
        return data
    if data["receptor"].x.shape[1] >= embeddings.shape[1]:
        tail = data["receptor"].x[:, -embeddings.shape[1] :]
        if torch.allclose(tail, embeddings.to(dtype=tail.dtype), atol=1.0e-5, rtol=1.0e-4):
            return data
    data["receptor"].x = torch.cat([data["receptor"].x, embeddings], dim=1)
    return data


def normalize_name(value) -> str | None:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if hasattr(value, "item") and callable(value.item):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        return value.strip().lower()
    return None


def fix_0d_tensors(data):
    def fix_store(store):
        for key in list(store.keys()):
            value = store[key]
            if torch.is_tensor(value) and value.dim() == 0:
                store[key] = value.view(1)

    for node_type in data.node_types:
        fix_store(data[node_type])
    for edge_type in data.edge_types:
        fix_store(data[edge_type])
    if hasattr(data, "_global_store"):
        fix_store(data._global_store)
    return data
