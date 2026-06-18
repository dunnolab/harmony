import csv
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from docking_model.data.parse.molecule import (
    read_mols,
    read_mols_v2,
    read_single_mol,
    resolve_ligand_path,
)


def to_numpy(value):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def denormalize_coords(coords, original_center):
    coords = np.asarray(coords, dtype=np.float32)
    if original_center is None:
        return coords
    return coords + np.asarray(original_center, dtype=np.float32).reshape(1, 3)


def normalize_optional_string(value):
    if value is None:
        return None
    if isinstance(value, dict) and not value:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        value = value[0]
    value = str(value).strip()
    if value in {"", "{}", "None", "none", "nan", "NaN"}:
        return None
    return value


def load_ligand_from_base_dir(base_dir, name):
    if base_dir is None or name is None:
        return None

    base_dir = Path(base_dir)
    names_to_try = [str(name)]
    if str(name).lower() not in names_to_try:
        names_to_try.append(str(name).lower())
    if str(name).upper() not in names_to_try:
        names_to_try.append(str(name).upper())

    for name_to_try in names_to_try:
        if (base_dir / name_to_try).is_dir():
            ligands = read_mols(str(base_dir), name_to_try, remove_hs=False)
            if ligands:
                return ligands[0]

    if base_dir.is_dir():
        ligands = read_mols_v2(str(base_dir), remove_hs=False)
        if ligands:
            return ligands[0]

    return None


def load_reference_ligand(docking_outputs):
    name = normalize_optional_string(docking_outputs.get("name"))
    base_dir = normalize_optional_string(docking_outputs.get("base_dir"))
    ligand_input = normalize_optional_string(
        docking_outputs.get("ligand_true_file")
    )
    if ligand_input is None:
        ligand_input = normalize_optional_string(
            docking_outputs.get("pocket_ligand_file")
        )
    if ligand_input is None:
        ligand_input = normalize_optional_string(docking_outputs.get("ligand_input"))
    ligand_description = normalize_optional_string(
        docking_outputs.get("ligand_description")
    )
    if ligand_description is None:
        ligand_description = "filename"

    if ligand_description == "smiles" and ligand_input is not None:
        mol = Chem.MolFromSmiles(ligand_input)
        return Chem.AddHs(mol) if mol is not None else None

    if ligand_input is not None:
        try:
            ligand_path = resolve_ligand_path(
                base_dir=base_dir,
                name=name,
                ligand_input=ligand_input,
            )
            ligand = read_single_mol(ligand_path, remove_hs=False)
            if ligand is not None:
                return ligand
        except FileNotFoundError:
            logging.warning(
                "%s: ligand_input=%s could not be resolved for trajectory export; "
                "falling back to base_dir/name ligand discovery",
                name,
                ligand_input,
            )

    ligand = load_ligand_from_base_dir(base_dir=base_dir, name=name)
    if ligand is not None:
        return ligand

    return None


def reference_ligand_coords(mol):
    if mol is None:
        return None
    mol_no_h = Chem.RemoveAllHs(Chem.Mol(mol))
    if mol_no_h.GetNumConformers() == 0:
        return None
    return np.asarray(mol_no_h.GetConformer().GetPositions(), dtype=np.float32)


def compute_raw_ligand_rmsd_to_reference(ligand_coords, reference_coords, filter_hs):
    if reference_coords is None:
        return None
    ligand_coords = ligand_coords_without_hydrogens(ligand_coords, filter_hs)
    if ligand_coords.shape != reference_coords.shape:
        return None
    return float(
        np.sqrt(np.mean(np.sum((ligand_coords - reference_coords) ** 2, axis=-1)))
    )


def compute_ligand_rmsd_to_reference(
    mol,
    ligand_coords,
    reference_coords,
    filter_hs,
):
    if reference_coords is None:
        return None
    ligand_coords = ligand_coords_without_hydrogens(ligand_coords, filter_hs)
    if ligand_coords.shape != reference_coords.shape:
        return None

    try:
        from docking_model.data.conformers.molecule import get_symmetry_rmsd

        mol_no_h = Chem.RemoveAllHs(Chem.Mol(mol))
        return float(get_symmetry_rmsd(mol_no_h, reference_coords, [ligand_coords])[0])
    except Exception:
        return compute_raw_ligand_rmsd_to_reference(
            ligand_coords,
            reference_coords,
            None,
        )


def load_reference_protein_coords(docking_outputs, atom_mask):
    from docking_model.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd

    reference_protein_path = (
        normalize_optional_string(docking_outputs.get("holo_rec_path_for_metrics"))
        or normalize_optional_string(docking_outputs.get("holo_rec_path"))
        or normalize_optional_string(docking_outputs.get("holo_protein_file"))
        or normalize_optional_string(docking_outputs.get("apo_rec_path"))
    )
    if reference_protein_path is None or atom_mask is None:
        return None, None, None

    struct = parse_pdb_pmd(reference_protein_path, remove_hs=True, reorder=True)
    reference_atom_coords = np.asarray(struct.get_coordinates(0), dtype=np.float32)
    backbone_atom_mask = np.asarray(
        [atom.name in {"N", "CA", "C", "O", "OXT"} for atom in struct.atoms],
        dtype=bool,
    )
    atom_mask = np.asarray(atom_mask, dtype=bool).reshape(-1)
    if atom_mask.shape[0] == reference_atom_coords.shape[0]:
        reference_atom_coords = reference_atom_coords[atom_mask]
        backbone_atom_mask = backbone_atom_mask[atom_mask]
    return reference_atom_coords, reference_protein_path, backbone_atom_mask


def valid_bool_mask(mask, n_atoms):
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    if mask.shape[0] != n_atoms:
        return None
    return mask


def sidechain_rmsd_mask(
    n_atoms,
    nearby_atom_mask,
    ca_mask,
    c_mask=None,
    n_mask=None,
    backbone_atom_mask=None,
):
    mask = valid_bool_mask(nearby_atom_mask, n_atoms)
    if mask is None:
        mask = np.ones(n_atoms, dtype=bool)

    backbone_mask = valid_bool_mask(backbone_atom_mask, n_atoms)
    if backbone_mask is None:
        backbone_mask = np.zeros(n_atoms, dtype=bool)
        for candidate in [ca_mask, c_mask, n_mask]:
            candidate_mask = valid_bool_mask(candidate, n_atoms)
            if candidate_mask is not None:
                backbone_mask |= candidate_mask

    sidechain_mask = mask & ~backbone_mask
    if not np.any(sidechain_mask):
        return None
    return sidechain_mask


def compute_sidechain_rmsd_to_reference(
    aligned_pred_atom_coords,
    reference_atom_coords,
    sidechain_mask,
):
    if (
        aligned_pred_atom_coords is None
        or reference_atom_coords is None
        or sidechain_mask is None
    ):
        return None
    if aligned_pred_atom_coords.shape != reference_atom_coords.shape:
        return None
    sidechain_mask = np.asarray(sidechain_mask, dtype=bool).reshape(-1)
    if sidechain_mask.shape[0] != aligned_pred_atom_coords.shape[0]:
        return None
    if not np.any(sidechain_mask):
        return None
    return float(
        np.sqrt(
            np.mean(
                np.sum(
                    (
                        aligned_pred_atom_coords[sidechain_mask]
                        - reference_atom_coords[sidechain_mask]
                    )
                    ** 2,
                    axis=-1,
                )
            )
        )
    )


def align_prediction_to_reference_protein(
    pred_atom_coords,
    pred_ligand_coords,
    reference_atom_coords,
    ca_mask,
    nearby_atom_mask,
):
    if reference_atom_coords is None or pred_atom_coords is None:
        return pred_atom_coords, pred_ligand_coords, "raw"
    if reference_atom_coords.shape != pred_atom_coords.shape:
        return pred_atom_coords, pred_ligand_coords, "raw"

    try:
        from docking_model.metrics.docking import align_proteins

        n_atoms = pred_atom_coords.shape[0]
        nearby_atom_mask = valid_bool_mask(nearby_atom_mask, n_atoms)
        ca_mask = valid_bool_mask(ca_mask, n_atoms)
        if nearby_atom_mask is not None and np.any(nearby_atom_mask):
            alignment_mode = "nearby_atoms"
        elif ca_mask is not None and np.any(ca_mask):
            alignment_mode = "calpha"
        else:
            alignment_mode = "all_atoms"

        R, t, _ = align_proteins(
            reference_atom_coords,
            pred_atom_coords,
            ca_mask=ca_mask,
            nearby_atom_mask=nearby_atom_mask,
            mode=alignment_mode,
        )
        return (
            (R @ pred_atom_coords.T).T + t,
            (R @ pred_ligand_coords.T).T + t,
            f"protein_{alignment_mode}",
        )
    except Exception:
        logging.warning(
            "Failed to align trajectory frame to reference protein; using raw ligand coordinates",
            exc_info=True,
        )
        return pred_atom_coords, pred_ligand_coords, "raw"


def sort_key(value):
    if value is None:
        return float("inf")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("inf")
    if np.isnan(value):
        return float("inf")
    return value


def assign_metric_ranks(summary_rows, metric_key, rank_key):
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (sort_key(row.get(metric_key)), int(row["sample_rank"])),
    )
    for rank_idx, row in enumerate(ranked_rows):
        row[rank_key] = rank_idx if sort_key(row.get(metric_key)) < float("inf") else None


def assign_combined_ranks(summary_rows):
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: (
            sort_key(row.get("final_ligand_rmsd_to_reference")),
            sort_key(row.get("final_sidechain_rmsd_to_reference")),
            int(row["sample_rank"]),
        ),
    )
    for rank_idx, row in enumerate(ranked_rows):
        if (
            sort_key(row.get("final_ligand_rmsd_to_reference")) < float("inf")
            or sort_key(row.get("final_sidechain_rmsd_to_reference")) < float("inf")
        ):
            row["combined_rmsd_rank"] = rank_idx
        else:
            row["combined_rmsd_rank"] = None


def write_rank_summary_csv(path, rows, sort_key_fields):
    if not rows:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_rows = sorted(
        rows,
        key=lambda row: tuple(sort_key(row.get(field)) for field in sort_key_fields)
        + (int(row["sample_rank"]),),
    )
    fieldnames = [
        "complex_id",
        "sample_rank",
        "confidence",
        "final_ligand_rmsd_to_reference",
        "final_raw_ligand_rmsd_to_reference",
        "final_sidechain_rmsd_to_reference",
        "final_raw_sidechain_rmsd_to_reference",
        "ligand_rmsd_rank",
        "sidechain_rmsd_rank",
        "combined_rmsd_rank",
        "rmsd_alignment",
        "rmsd_reference_protein_path",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sorted_rows)
    return path


def make_mol_with_coordinates(mol, coords):
    mol = deepcopy(mol)
    mol = Chem.RemoveAllHs(mol)
    mol.RemoveAllConformers()

    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coord in enumerate(coords):
        conf.SetAtomPosition(
            atom_idx,
            Point3D(float(coord[0]), float(coord[1]), float(coord[2])),
        )
    mol.AddConformer(conf, assignId=True)
    return mol


def write_mol_file(mol, coords, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mol_with_coords = make_mol_with_coordinates(mol, coords)
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol_with_coords)
    writer.close()
    return output_path


def ligand_coords_without_hydrogens(ligand_coords, filter_hs):
    if filter_hs is None:
        return ligand_coords
    filter_hs = np.asarray(filter_hs, dtype=bool).reshape(-1)
    if filter_hs.shape[0] != ligand_coords.shape[0]:
        return ligand_coords
    return ligand_coords[filter_hs]


def write_ligand_frame(mol, ligand_coords, filter_hs, output_path):
    ligand_coords = ligand_coords_without_hydrogens(ligand_coords, filter_hs)
    expected_atoms = Chem.RemoveAllHs(Chem.Mol(mol)).GetNumAtoms()
    if ligand_coords.shape[0] != expected_atoms:
        raise ValueError(
            "ligand atom count mismatch: "
            f"trajectory has {ligand_coords.shape[0]} atoms, "
            f"template has {expected_atoms} heavy atoms"
        )
    write_mol_file(mol, ligand_coords, output_path)
    return output_path


def write_protein_frame(apo_rec_path, atom_mask, atom_coords, output_path):
    from docking_model.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd

    if apo_rec_path is None:
        raise ValueError("apo_rec_path is not available")
    if atom_mask is None:
        raise ValueError("atom_mask is not available")

    atom_mask = np.asarray(atom_mask, dtype=bool).reshape(-1)
    atom_coords = np.asarray(atom_coords, dtype=np.float32)
    if int(atom_mask.sum()) != atom_coords.shape[0]:
        raise ValueError(
            "protein atom count mismatch: "
            f"trajectory has {atom_coords.shape[0]} atoms, "
            f"atom_mask selects {int(atom_mask.sum())} atoms"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    struct = parse_pdb_pmd(apo_rec_path, remove_hs=True, reorder=True)
    full_atom_pos = struct.get_coordinates(0)
    full_atom_pos[atom_mask] = atom_coords

    for atom, coord in zip(struct.atoms, full_atom_pos):
        atom.xx = float(coord[0])
        atom.xy = float(coord[1])
        atom.xz = float(coord[2])

    struct.save(str(output_path), overwrite=True)
    return output_path


def write_docking_trajectory_frames(
    docking_outputs,
    output_dir,
    max_ranks=None,
):
    ligand_trajectory = to_numpy(docking_outputs.get("ligand_trajectory"))
    atom_trajectory = to_numpy(docking_outputs.get("atom_trajectory"))
    if ligand_trajectory is None and atom_trajectory is None:
        return None

    output_dir = Path(output_dir)
    trajectory_dir = output_dir / "trajectory"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

    n_ranks = 0
    if ligand_trajectory is not None:
        n_ranks = ligand_trajectory.shape[0]
    elif atom_trajectory is not None:
        n_ranks = atom_trajectory.shape[0]
    if max_ranks is not None:
        n_ranks = min(n_ranks, int(max_ranks))

    original_center = to_numpy(docking_outputs.get("original_center"))
    filter_hs = docking_outputs.get("filterHs")
    atom_mask = docking_outputs.get("atom_mask")
    apo_rec_path = docking_outputs.get("apo_rec_path")
    ca_mask = docking_outputs.get("ca_mask")
    c_mask = docking_outputs.get("c_mask")
    n_mask = docking_outputs.get("n_mask")
    nearby_atom_mask = docking_outputs.get("pocket_atom_mask")
    confidences = docking_outputs.get("confidence")
    ligand_mol = None

    if ligand_trajectory is not None:
        try:
            ligand_mol = load_reference_ligand(docking_outputs)
        except Exception as exc:
            logging.warning(
                "%s: failed to load ligand template for trajectory export due to %s",
                docking_outputs.get("name"),
                exc,
                exc_info=True,
            )
    reference_ligand_pos = reference_ligand_coords(ligand_mol)
    reference_atom_coords = None
    reference_protein_path = None
    backbone_atom_mask = None
    if atom_trajectory is not None:
        try:
            (
                reference_atom_coords,
                reference_protein_path,
                backbone_atom_mask,
            ) = load_reference_protein_coords(
                docking_outputs=docking_outputs,
                atom_mask=atom_mask,
            )
        except Exception as exc:
            logging.warning(
                "%s: failed to load reference protein for trajectory RMSD due to %s",
                docking_outputs.get("name"),
                exc,
                exc_info=True,
            )
    sidechain_mask = None
    if reference_atom_coords is not None:
        sidechain_mask = sidechain_rmsd_mask(
            n_atoms=reference_atom_coords.shape[0],
            nearby_atom_mask=nearby_atom_mask,
            ca_mask=ca_mask,
            c_mask=c_mask,
            n_mask=n_mask,
            backbone_atom_mask=backbone_atom_mask,
        )

    rows = []
    summary_rows = []
    ligand_warning_logged = False
    protein_warning_logged = False

    for sample_rank in range(n_ranks):
        sample_dir = trajectory_dir / f"rank_{sample_rank:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        n_steps = 0
        if ligand_trajectory is not None:
            n_steps = ligand_trajectory.shape[1]
        elif atom_trajectory is not None:
            n_steps = atom_trajectory.shape[1]

        final_ligand_rmsd = None
        final_raw_ligand_rmsd = None
        final_sidechain_rmsd = None
        final_raw_sidechain_rmsd = None
        final_rmsd_alignment = "raw"
        if ligand_trajectory is not None and n_steps:
            final_ligand_coords = denormalize_coords(
                ligand_trajectory[sample_rank, n_steps - 1],
                original_center,
            )
            final_raw_ligand_rmsd = compute_ligand_rmsd_to_reference(
                ligand_mol,
                final_ligand_coords,
                reference_ligand_pos,
                filter_hs,
            )
            final_aligned_atom_coords = None
            if atom_trajectory is not None:
                final_atom_coords = denormalize_coords(
                    atom_trajectory[sample_rank, n_steps - 1],
                    original_center,
                )
                final_raw_sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                    final_atom_coords,
                    reference_atom_coords,
                    sidechain_mask,
                )
                (
                    final_aligned_atom_coords,
                    final_ligand_coords,
                    final_rmsd_alignment,
                ) = align_prediction_to_reference_protein(
                    pred_atom_coords=final_atom_coords,
                    pred_ligand_coords=final_ligand_coords,
                    reference_atom_coords=reference_atom_coords,
                    ca_mask=ca_mask,
                    nearby_atom_mask=nearby_atom_mask,
                )
                final_sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                    final_aligned_atom_coords,
                    reference_atom_coords,
                    sidechain_mask,
                )
            final_ligand_rmsd = compute_ligand_rmsd_to_reference(
                ligand_mol,
                final_ligand_coords,
                reference_ligand_pos,
                filter_hs,
            )

        confidence = (
            None
            if confidences is None
            else float(np.asarray(confidences).reshape(-1)[sample_rank])
        )
        summary_rows.append(
            {
                "complex_id": docking_outputs.get("name"),
                "sample_rank": sample_rank,
                "confidence": confidence,
                "final_ligand_rmsd_to_reference": final_ligand_rmsd,
                "final_raw_ligand_rmsd_to_reference": final_raw_ligand_rmsd,
                "final_sidechain_rmsd_to_reference": final_sidechain_rmsd,
                "final_raw_sidechain_rmsd_to_reference": final_raw_sidechain_rmsd,
                "rmsd_alignment": final_rmsd_alignment,
                "rmsd_reference_protein_path": reference_protein_path or "",
            }
        )

        for step_idx in range(n_steps):
            ligand_sdf_path = None
            protein_pdb_path = None
            ligand_rmsd = None
            raw_ligand_rmsd = None
            sidechain_rmsd = None
            raw_sidechain_rmsd = None
            rmsd_alignment = "raw"
            atom_coords = None
            aligned_atom_coords = None

            if ligand_trajectory is not None and ligand_mol is not None:
                try:
                    ligand_coords = denormalize_coords(
                        ligand_trajectory[sample_rank, step_idx],
                        original_center,
                    )
                    raw_ligand_rmsd = compute_ligand_rmsd_to_reference(
                        ligand_mol,
                        ligand_coords,
                        reference_ligand_pos,
                        filter_hs,
                    )
                    ligand_rmsd_coords = ligand_coords
                    if atom_trajectory is not None:
                        atom_coords_for_alignment = denormalize_coords(
                            atom_trajectory[sample_rank, step_idx],
                            original_center,
                        )
                        raw_sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                            atom_coords_for_alignment,
                            reference_atom_coords,
                            sidechain_mask,
                        )
                        (
                            aligned_atom_coords,
                            aligned_ligand_coords,
                            rmsd_alignment,
                        ) = align_prediction_to_reference_protein(
                            pred_atom_coords=atom_coords_for_alignment,
                            pred_ligand_coords=ligand_coords,
                            reference_atom_coords=reference_atom_coords,
                            ca_mask=ca_mask,
                            nearby_atom_mask=nearby_atom_mask,
                        )
                        ligand_rmsd_coords = aligned_ligand_coords
                        sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                            aligned_atom_coords,
                            reference_atom_coords,
                            sidechain_mask,
                        )
                    ligand_rmsd = compute_ligand_rmsd_to_reference(
                        ligand_mol,
                        ligand_rmsd_coords,
                        reference_ligand_pos,
                        filter_hs,
                    )
                    ligand_sdf_path = sample_dir / f"step_{step_idx:04d}_ligand.sdf"
                    write_ligand_frame(
                        ligand_mol,
                        ligand_coords,
                        filter_hs,
                        ligand_sdf_path,
                    )
                except Exception as exc:
                    if not ligand_warning_logged:
                        logging.warning(
                            "%s: failed to write ligand trajectory frames due to %s",
                            docking_outputs.get("name"),
                            exc,
                            exc_info=True,
                        )
                        ligand_warning_logged = True
                    ligand_sdf_path = None

            if atom_trajectory is not None:
                try:
                    if atom_coords is None:
                        atom_coords = denormalize_coords(
                            atom_trajectory[sample_rank, step_idx],
                            original_center,
                        )
                    if raw_sidechain_rmsd is None:
                        raw_sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                            atom_coords,
                            reference_atom_coords,
                            sidechain_mask,
                        )
                    if sidechain_rmsd is None:
                        (
                            aligned_atom_coords,
                            _,
                            atom_rmsd_alignment,
                        ) = align_prediction_to_reference_protein(
                            pred_atom_coords=atom_coords,
                            pred_ligand_coords=np.zeros((0, 3), dtype=np.float32),
                            reference_atom_coords=reference_atom_coords,
                            ca_mask=ca_mask,
                            nearby_atom_mask=nearby_atom_mask,
                        )
                        sidechain_rmsd = compute_sidechain_rmsd_to_reference(
                            aligned_atom_coords,
                            reference_atom_coords,
                            sidechain_mask,
                        )
                        if rmsd_alignment == "raw":
                            rmsd_alignment = atom_rmsd_alignment
                    protein_pdb_path = sample_dir / f"step_{step_idx:04d}_protein.pdb"
                    write_protein_frame(
                        apo_rec_path,
                        atom_mask,
                        atom_coords,
                        protein_pdb_path,
                    )
                except Exception as exc:
                    if not protein_warning_logged:
                        logging.warning(
                            "%s: failed to write protein trajectory frames due to %s",
                            docking_outputs.get("name"),
                            exc,
                            exc_info=True,
                        )
                        protein_warning_logged = True
                    protein_pdb_path = None

            rows.append(
                {
                    "complex_id": docking_outputs.get("name"),
                    "sample_rank": sample_rank,
                    "step_idx": step_idx,
                    "confidence": confidence,
                    "ligand_rmsd_to_reference": ligand_rmsd,
                    "raw_ligand_rmsd_to_reference": raw_ligand_rmsd,
                    "final_ligand_rmsd_to_reference": final_ligand_rmsd,
                    "final_raw_ligand_rmsd_to_reference": final_raw_ligand_rmsd,
                    "sidechain_rmsd_to_reference": sidechain_rmsd,
                    "raw_sidechain_rmsd_to_reference": raw_sidechain_rmsd,
                    "final_sidechain_rmsd_to_reference": final_sidechain_rmsd,
                    "final_raw_sidechain_rmsd_to_reference": final_raw_sidechain_rmsd,
                    "rmsd_alignment": rmsd_alignment,
                    "rmsd_reference_protein_path": reference_protein_path or "",
                    "ligand_sdf_path": ""
                    if ligand_sdf_path is None
                    else str(ligand_sdf_path),
                    "protein_pdb_path": ""
                    if protein_pdb_path is None
                    else str(protein_pdb_path),
                }
            )

    assign_metric_ranks(
        summary_rows,
        "final_ligand_rmsd_to_reference",
        "ligand_rmsd_rank",
    )
    assign_metric_ranks(
        summary_rows,
        "final_sidechain_rmsd_to_reference",
        "sidechain_rmsd_rank",
    )
    assign_combined_ranks(summary_rows)

    summary_by_sample_rank = {
        int(summary_row["sample_rank"]): summary_row for summary_row in summary_rows
    }
    for row in rows:
        summary_row = summary_by_sample_rank[int(row["sample_rank"])]
        row["ligand_rmsd_rank"] = summary_row.get("ligand_rmsd_rank")
        row["sidechain_rmsd_rank"] = summary_row.get("sidechain_rmsd_rank")
        row["combined_rmsd_rank"] = summary_row.get("combined_rmsd_rank")

    write_rank_summary_csv(
        trajectory_dir / "rank_summary.csv",
        summary_rows,
        ("final_ligand_rmsd_to_reference", "final_sidechain_rmsd_to_reference"),
    )
    write_rank_summary_csv(
        trajectory_dir / "rank_summary_by_ligand_rmsd.csv",
        summary_rows,
        ("final_ligand_rmsd_to_reference", "final_sidechain_rmsd_to_reference"),
    )
    write_rank_summary_csv(
        trajectory_dir / "rank_summary_by_sidechain_rmsd.csv",
        summary_rows,
        ("final_sidechain_rmsd_to_reference", "final_ligand_rmsd_to_reference"),
    )

    manifest_path = trajectory_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "complex_id",
                "sample_rank",
                "step_idx",
                "confidence",
                "ligand_rmsd_to_reference",
                "raw_ligand_rmsd_to_reference",
                "final_ligand_rmsd_to_reference",
                "final_raw_ligand_rmsd_to_reference",
                "sidechain_rmsd_to_reference",
                "raw_sidechain_rmsd_to_reference",
                "final_sidechain_rmsd_to_reference",
                "final_raw_sidechain_rmsd_to_reference",
                "ligand_rmsd_rank",
                "sidechain_rmsd_rank",
                "combined_rmsd_rank",
                "rmsd_alignment",
                "rmsd_reference_protein_path",
                "ligand_sdf_path",
                "protein_pdb_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return manifest_path
