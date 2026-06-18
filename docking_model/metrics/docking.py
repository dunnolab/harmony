from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from docking_model.geometry.ops import rigid_transform_kabsch_numpy


_WARNED_INVALID_PLI_LDDT: set[str] = set()


def align_proteins(
    true_atom_pos,
    pred_atom_pos,
    ca_mask=None,
    nearby_atom_mask=None,
    mode: str = "nearby_atoms",
):
    if mode == "nearby_atoms":
        if nearby_atom_mask is None:
            raise ValueError("nearby_atom_mask is required for nearby_atoms alignment.")
        return rigid_transform_kabsch_numpy(true_atom_pos[nearby_atom_mask], pred_atom_pos[nearby_atom_mask])
    if mode == "calpha":
        if ca_mask is None:
            raise ValueError("ca_mask is required for calpha alignment.")
        return rigid_transform_kabsch_numpy(true_atom_pos[ca_mask], pred_atom_pos[ca_mask])
    if mode == "all_atoms":
        return rigid_transform_kabsch_numpy(true_atom_pos, pred_atom_pos)
    if mode == "noalign":
        return np.eye(3), np.zeros((1, 3)), 0.0
    raise ValueError(f"Unsupported protein alignment mode: {mode}")


def compute_ligand_rmsd(true_ligand_pos, pred_ligand_pos, name=None, mol=None):
    del name
    if mol is not None:
        try:
            from docking_model.data.conformers.molecule import get_symmetry_rmsd

            return get_symmetry_rmsd(mol, true_ligand_pos, [pred_ligand_pos])[0]
        except Exception:
            pass
    return np.sqrt(((pred_ligand_pos - true_ligand_pos) ** 2).sum(axis=1).mean(axis=0))


def pli_lddt_score(
    rec_coords_predicted: torch.Tensor,
    lig_coords_predicted: torch.Tensor,
    rec_coords_true: torch.Tensor,
    lig_coords_true: torch.Tensor,
    pli_distance_threshold: float = 6.0,
    lddt_thresholds: Sequence[float] = (0.5, 1.0, 2.0, 4.0),
) -> torch.Tensor:
    """Compute the protein-ligand interface LDDT score for each predicted pose."""

    if rec_coords_predicted.dim() != 3 or lig_coords_predicted.dim() != 3:
        raise ValueError("Predicted receptor and ligand coordinates must have shape [B, N, 3].")

    batch_size = rec_coords_predicted.shape[0]
    rec_coords_true = expand_true_coords(rec_coords_true, batch_size)
    lig_coords_true = expand_true_coords(lig_coords_true, batch_size)

    dmat_predicted = torch.cdist(rec_coords_predicted, lig_coords_predicted)
    dmat_true = torch.cdist(rec_coords_true, lig_coords_true)
    dists_to_score = (dmat_true < pli_distance_threshold).float()
    dist_l1 = torch.abs(dmat_true - dmat_predicted)

    if torch.sum(dists_to_score) == 0:
        raise ValueError(
            "No protein-ligand atom pairs are below the PLI-LDDT distance threshold. "
            f"Minimum true receptor-ligand distance: {dmat_true.min()}"
        )

    score = torch.mean(torch.stack([(dist_l1 < threshold).float() for threshold in lddt_thresholds]), dim=0)
    norm = 1.0 / (1e-10 + torch.sum(dists_to_score, dim=(-2, -1)))
    return norm * (1e-10 + torch.sum(dists_to_score * score, dim=(-2, -1)))


def compute_inference_sample_metrics(reference, predictions: list) -> dict[str, np.ndarray | float]:
    """Compute per-sample docking metrics against one reference complex."""

    if not predictions:
        return {}

    reference = first_graph(reference)
    name = graph_name(reference)

    if "reference_ligand_loaded" in reference and not bool(reference["reference_ligand_loaded"]):
        logging.warning("Skipping inference metrics for %s because the reference ligand was not loaded.", name)
        return {}
    if "reference_protein_loaded" in reference and not bool(reference["reference_protein_loaded"]):
        logging.warning("Skipping inference metrics for %s because the reference holo protein was not loaded.", name)
        return {}
    if "orig_pos" not in reference["ligand"]:
        logging.warning("Skipping inference metrics for %s because reference ligand coordinates are missing.", name)
        return {}
    if "orig_holo_pos" not in reference["atom"]:
        logging.warning("Skipping inference metrics for %s because reference holo protein coordinates are missing.", name)
        return {}

    filter_hs = as_numpy(reference["ligand"].x[:, 0] != 0).astype(bool)
    true_ligand_pos = filter_ligand_pos(as_numpy(unwrap(reference["ligand"].orig_pos)), filter_hs)
    true_atom_pos = as_numpy(unwrap(reference["atom"].orig_holo_pos))
    nearby_atoms = nearby_atom_mask(reference, true_atom_pos.shape[0])
    mol = getattr(reference, "mol", None)

    aligned_atom_pos = []
    aligned_ligand_pos = []
    rmsds_before_alignment = []
    rmsds = []
    centroid_distances = []

    for prediction in predictions:
        pred_atom_pos, pred_ligand_pos = prediction_positions(prediction)
        pred_ligand_pos = filter_ligand_pos(pred_ligand_pos, filter_hs)
        rmsd_before_alignment = compute_ligand_rmsd(true_ligand_pos, pred_ligand_pos, name=name, mol=mol)

        try:
            R, t, _ = rigid_transform_kabsch_numpy(true_atom_pos[nearby_atoms], pred_atom_pos[nearby_atoms])
        except Exception as exc:
            logging.warning("Skipping a validation-inference pose because protein alignment failed: %s", exc)
            continue

        pred_atom_pos = (R @ pred_atom_pos.T).T + t
        pred_ligand_pos = (R @ pred_ligand_pos.T).T + t

        aligned_atom_pos.append(pred_atom_pos)
        aligned_ligand_pos.append(pred_ligand_pos)
        rmsds_before_alignment.append(rmsd_before_alignment)
        rmsds.append(compute_ligand_rmsd(true_ligand_pos, pred_ligand_pos, name=name, mol=mol))
        centroid_distances.append(float(np.linalg.norm(pred_ligand_pos.mean(axis=0) - true_ligand_pos.mean(axis=0))))

    if not aligned_atom_pos:
        return {}

    atom_pos = np.asarray(aligned_atom_pos)
    ligand_pos = np.asarray(aligned_ligand_pos)
    native_min_distance, native_contact_pairs = native_interface_stats(true_atom_pos, true_ligand_pos)
    sample_metrics: dict[str, np.ndarray | float] = {
        "rmsds": np.asarray(rmsds, dtype=float),
        "rmsds_before_alignment": np.asarray(rmsds_before_alignment, dtype=float),
        "centroid_distances": np.asarray(centroid_distances, dtype=float),
        "native_min_distance": native_min_distance,
        "native_contact_pairs": float(native_contact_pairs),
    }

    ca_mask = as_numpy(reference["atom"].ca_mask).astype(bool)
    if ca_mask.any():
        sample_metrics["bb_rmsds"] = rmsd(atom_pos[:, ca_mask], true_atom_pos[None, ca_mask])

    sample_metrics["aa_rmsds"] = np.asarray(
        [sidechain_rmsd(nearby_atoms, atom_sample, true_atom_pos) for atom_sample in atom_pos],
        dtype=float,
    )

    if native_contact_pairs == 0:
        warning_key = f"native:{name}"
        if warning_key not in _WARNED_INVALID_PLI_LDDT:
            logging.warning(
                "Skipping validation-inference PLI-LDDT for %s because the cached native holo protein/ligand "
                "has no atom pairs below 6 A. Minimum native receptor-ligand distance: %.3f A.",
                name,
                native_min_distance,
            )
            _WARNED_INVALID_PLI_LDDT.add(warning_key)
        sample_metrics["pli_lddt"] = np.full(len(ligand_pos), np.nan, dtype=float)
        sample_metrics["pli_lddt_valid"] = 0.0
    else:
        try:
            pli_lddt = pli_lddt_score(
                rec_coords_predicted=torch.from_numpy(atom_pos).float(),
                lig_coords_predicted=torch.from_numpy(ligand_pos).float(),
                rec_coords_true=torch.from_numpy(true_atom_pos).float(),
                lig_coords_true=torch.from_numpy(true_ligand_pos).float(),
            )
            sample_metrics["pli_lddt"] = (100.0 * pli_lddt).detach().cpu().numpy().astype(float)
            sample_metrics["pli_lddt_valid"] = 1.0
        except ValueError as exc:
            warning_key = f"metric:{name}"
            if warning_key not in _WARNED_INVALID_PLI_LDDT:
                logging.warning("Skipping validation-inference PLI-LDDT for %s because metric computation failed: %s", name, exc)
                _WARNED_INVALID_PLI_LDDT.add(warning_key)
            sample_metrics["pli_lddt"] = np.full(len(ligand_pos), np.nan, dtype=float)
            sample_metrics["pli_lddt_valid"] = 0.0

    return sample_metrics


def compute_valinf_metrics(reference, predictions: list) -> dict[str, float]:
    """Compute validation-inference docking metrics for one complex."""

    sample_metrics = compute_inference_sample_metrics(reference, predictions)
    if not sample_metrics:
        return {}

    ligand_rmsd = np.asarray(sample_metrics["rmsds"], dtype=float)
    metrics = {
        "rmsds_lt1": percent_below(ligand_rmsd, 1.0),
        "rmsds_lt2": percent_below(ligand_rmsd, 2.0),
        "rmsds_lt5": percent_below(ligand_rmsd, 5.0),
        "mean_rmsd": mean(ligand_rmsd),
        "native_min_distance": float(sample_metrics["native_min_distance"]),
        "native_contact_pairs": float(sample_metrics["native_contact_pairs"]),
    }

    if "bb_rmsds" in sample_metrics:
        ca_rmsd = np.asarray(sample_metrics["bb_rmsds"], dtype=float)
        metrics.update(
            {
                "bb_rmsds_lt2": percent_below(ca_rmsd, 2.0),
                "bb_rmsds_lt1": percent_below(ca_rmsd, 1.0),
                "bb_rmsds_lt05": percent_below(ca_rmsd, 0.5),
                "mean_bb_rmsd": mean(ca_rmsd),
            }
        )

    aa_rmsd = np.asarray(sample_metrics["aa_rmsds"], dtype=float)
    metrics.update(
        {
            "aa_rmsds_lt2": percent_below(aa_rmsd, 2.0),
            "aa_rmsds_lt1": percent_below(aa_rmsd, 1.0),
            "aa_rmsds_lt05": percent_below(aa_rmsd, 0.5),
            "mean_aa_rmsd": mean(aa_rmsd),
        }
    )

    if "pli_lddt" in sample_metrics:
        metrics["pli_lddt"] = mean(np.asarray(sample_metrics["pli_lddt"], dtype=float))
        metrics["pli_lddt_valid"] = float(sample_metrics["pli_lddt_valid"])

    return metrics


def expand_true_coords(coords: torch.Tensor, batch_size: int) -> torch.Tensor:
    if coords.dim() == 2:
        return coords.unsqueeze(0).expand(batch_size, -1, -1)
    if coords.dim() == 3 and coords.shape[0] == 1 and batch_size > 1:
        return coords.expand(batch_size, -1, -1)
    if coords.dim() == 3 and coords.shape[0] == batch_size:
        return coords
    raise ValueError(f"True coordinate batch shape {tuple(coords.shape)} is incompatible with batch size {batch_size}.")


def first_graph(data):
    if isinstance(data, list):
        return data[0]
    if hasattr(data, "to_data_list"):
        items = data.to_data_list()
        if len(items) != 1:
            raise ValueError("Validation-inference metrics currently expect batch_size=1.")
        return items[0]
    return data


def prediction_positions(prediction) -> tuple[np.ndarray, np.ndarray]:
    prediction = first_graph(prediction)
    if isinstance(prediction, dict) and "atom_pos" in prediction and "ligand_pos" in prediction:
        return as_numpy(prediction["atom_pos"]), as_numpy(prediction["ligand_pos"])
    return as_numpy(prediction["atom"].pos), as_numpy(prediction["ligand"].pos)


def unwrap(value):
    if isinstance(value, list):
        return value[0]
    return value


def graph_name(data) -> str:
    value = None
    try:
        if "name" in data:
            value = data["name"]
    except Exception:
        value = getattr(data, "name", None)
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return "<unknown>" if value is None else str(value)


def as_numpy(value) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def filter_ligand_pos(pos: np.ndarray, filter_hs: np.ndarray) -> np.ndarray:
    if pos.shape[0] == filter_hs.shape[0]:
        return pos[filter_hs]
    return pos


def nearby_atom_mask(reference, atom_count: int) -> np.ndarray:
    if "nearby_atoms" in reference["atom"]:
        mask = as_numpy(reference["atom"].nearby_atoms).astype(bool)
        if mask.shape[0] == atom_count and mask.any():
            return mask
    return np.ones(atom_count, dtype=bool)


def rmsd(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(((predicted - target) ** 2).sum(axis=2).mean(axis=1))


def sidechain_rmsd(atom_mask: np.ndarray, predicted: np.ndarray, target: np.ndarray) -> float:
    if atom_mask.sum() == 0:
        return 0.0
    return float(np.sqrt(np.sum((predicted[atom_mask] - target[atom_mask]) ** 2) / atom_mask.sum()))


def native_interface_stats(atom_pos: np.ndarray, ligand_pos: np.ndarray, threshold: float = 6.0) -> tuple[float, int]:
    if atom_pos.size == 0 or ligand_pos.size == 0:
        return float("inf"), 0
    dists = np.sqrt(np.sum((atom_pos[:, None, :] - ligand_pos[None, :, :]) ** 2, axis=-1))
    return float(np.min(dists)), int(np.sum(dists < threshold))


def percent_below(values: np.ndarray, threshold: float) -> float:
    values = finite(values)
    if values.size == 0:
        return float("nan")
    return 100.0 * float(np.mean(values < threshold))


def mean(values: np.ndarray) -> float:
    values = finite(values)
    return float(np.mean(values)) if values.size else float("nan")


def finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]
