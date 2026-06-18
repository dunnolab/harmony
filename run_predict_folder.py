from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Geometry import Point3D

from docking_model.data.write.trajectory import denormalize_coords, write_protein_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run single-complex inference and export the confidence-selected prediction."
    )
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--protein", type=Path, required=True)
    parser.add_argument("--ligand", type=Path, required=True)
    parser.add_argument("--model-parameters", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("inference_outputs/folder_prediction"))
    parser.add_argument("--esm-embeddings-path", type=Path, default=None)
    parser.add_argument("--reference-ligand", type=Path, default=None)
    parser.add_argument("--reference-protein", type=Path, default=None)
    parser.add_argument("--pocket-residues", default=None)
    parser.add_argument("--pocket-ligand", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-trajectory", action="store_true")
    parser.add_argument("--export-trajectory-files", action="store_true")
    parser.add_argument("--trajectory-max-ranks", type=int, default=1)
    parser.add_argument("--posebusters", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    protein_path = input_path(input_dir, args.protein)
    ligand_path = input_path(input_dir, args.ligand)
    model_parameters = args.model_parameters.expanduser().resolve()
    checkpoint_path = (
        args.weights.expanduser().resolve()
        if args.weights is not None
        else model_parameters.parent / "best_model.pt"
    )

    if not model_parameters.exists():
        raise FileNotFoundError(model_parameters)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    input_csv = output_dir / f"{args.name}_input.csv"
    write_input_csv(
        input_csv=input_csv,
        name=args.name,
        input_dir=input_dir,
        protein_path=protein_path,
        ligand_path=ligand_path,
        reference_ligand_path=input_path(input_dir, args.reference_ligand) if args.reference_ligand else None,
        reference_protein_path=input_path(input_dir, args.reference_protein) if args.reference_protein else None,
        pocket_ligand_path=input_path(input_dir, args.pocket_ligand) if args.pocket_ligand else None,
        pocket_residues=args.pocket_residues,
    )

    overrides = build_overrides(
        input_csv=input_csv,
        output_dir=output_dir,
        checkpoint_path=checkpoint_path,
        esm_embeddings_path=args.esm_embeddings_path,
        samples=args.samples,
        steps=args.steps,
        device=args.device,
        save_trajectory=args.save_trajectory,
        export_trajectory_files=args.export_trajectory_files,
        trajectory_max_ranks=args.trajectory_max_ranks,
        posebusters=args.posebusters,
        wandb=args.wandb,
    )

    from docking_model.workflows.infer import run_inference

    run_inference(str(model_parameters), overrides=overrides, show_progress=True)

    prediction_dir = output_dir / args.name
    predictions_path = prediction_dir / "docking_predictions.pkl"
    with predictions_path.open("rb") as handle:
        docking_outputs = pickle.load(handle)

    prediction_index = confidence_selected_prediction_index(docking_outputs)
    protein_pdb, ligand_pdb, ligand_sdf, complex_pdb = write_prediction_structures(
        docking_outputs=docking_outputs,
        ligand_path=ligand_path,
        output_dir=prediction_dir,
        prediction_index=prediction_index,
    )

    print(f"predictions={predictions_path}")
    print(f"prediction_index={prediction_index}")
    print(f"prediction_number={prediction_index + 1}")
    print(f"protein_pdb={protein_pdb}")
    print(f"ligand_pdb={ligand_pdb}")
    print(f"ligand_sdf={ligand_sdf}")
    print(f"complex_pdb={complex_pdb}")
    if args.posebusters:
        print(f"posebusters_csv={prediction_dir / 'posebusters_metrics.csv'}")
        print(f"posebusters_summary_csv={output_dir / 'posebusters_metrics.csv'}")


def input_path(input_dir: Path, path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (input_dir / path).resolve()


def write_input_csv(
    input_csv: Path,
    name: str,
    input_dir: Path,
    protein_path: Path,
    ligand_path: Path,
    reference_ligand_path: Path | None,
    reference_protein_path: Path | None,
    pocket_ligand_path: Path | None,
    pocket_residues: str | None,
) -> None:
    row = {
        "pdbid": name,
        "base_dir": str(input_dir),
        "apo_protein_file": str(protein_path),
        "holo_protein_file": str(reference_protein_path or ""),
        "ligand_input": str(ligand_path),
        "ligand_true_file": str(reference_ligand_path or ""),
        "ligand_description": "filename",
        "pocket_residues": pocket_residues or "",
        "pocket_ligand_file": str(pocket_ligand_path or ""),
    }
    input_csv.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


def build_overrides(
    input_csv: Path,
    output_dir: Path,
    checkpoint_path: Path,
    esm_embeddings_path: Path | None,
    samples: int | None,
    steps: int | None,
    device: str | None,
    save_trajectory: bool,
    export_trajectory_files: bool,
    trajectory_max_ranks: int,
    posebusters: bool,
    wandb: bool,
) -> dict:
    overrides: dict = {
        "logger": {"wandb": bool(wandb)},
        "inference": {
            "input_csv": str(input_csv),
            "output_dir": str(output_dir),
            "limit_complexes": 1,
            "checkpoint": str(checkpoint_path),
            "save_trajectory": bool(save_trajectory or export_trajectory_files),
            "export_trajectory_files": bool(export_trajectory_files),
            "trajectory_max_ranks": int(trajectory_max_ranks),
            "posebusters_metrics": bool(posebusters),
        },
    }
    if esm_embeddings_path is not None:
        esm_path = str(esm_embeddings_path.expanduser().resolve())
        overrides["inference"]["esm_embeddings_path"] = esm_path
        overrides.setdefault("model", {})["esm_embeddings_path"] = esm_path
    if samples is not None:
        overrides.setdefault("sampler", {})["samples_per_complex"] = int(samples)
    if steps is not None:
        overrides.setdefault("sampler", {})["inference_steps"] = int(steps)
    if device is not None:
        overrides.setdefault("training", {})["device"] = device
    return overrides


def confidence_selected_prediction_index(docking_outputs: dict) -> int:
    confidences = np.asarray(docking_outputs["confidence"], dtype=float)
    if confidences.ndim > 1:
        confidences = confidences.reshape(confidences.shape[0], -1)[:, 0]
    if not np.isfinite(confidences).any():
        raise ValueError("docking_predictions.pkl has no finite confidence scores.")
    return int(np.nanargmax(confidences))


def write_prediction_structures(
    docking_outputs: dict,
    ligand_path: Path,
    output_dir: Path,
    prediction_index: int,
) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    original_center = docking_outputs.get("original_center")
    atom_coords = denormalize_coords(docking_outputs["atom_pos"][prediction_index], original_center)
    ligand_coords = denormalize_coords(docking_outputs["ligand_pos"][prediction_index], original_center)

    prediction_number = prediction_index + 1
    protein_pdb = output_dir / f"prediction{prediction_number}_protein.pdb"
    ligand_pdb = output_dir / f"prediction{prediction_number}_ligand.pdb"
    ligand_sdf = output_dir / f"prediction{prediction_number}_ligand.sdf"
    complex_pdb = output_dir / f"prediction{prediction_number}_complex.pdb"

    write_protein_frame(
        apo_rec_path=docking_outputs["apo_rec_path"],
        atom_mask=docking_outputs["atom_mask"],
        atom_coords=atom_coords,
        output_path=protein_pdb,
    )

    ligand_template, ligand_coords = ligand_template_for_coords(
        ligand_path=ligand_path,
        coords=ligand_coords,
        filter_hs=docking_outputs.get("filterHs"),
    )
    ligand_mol = mol_with_coordinates(ligand_template, ligand_coords)
    Chem.MolToPDBFile(ligand_mol, str(ligand_pdb))
    writer = Chem.SDWriter(str(ligand_sdf))
    writer.write(ligand_mol)
    writer.close()

    write_complex_pdb(protein_pdb=protein_pdb, ligand_pdb=ligand_pdb, output_path=complex_pdb)
    return protein_pdb, ligand_pdb, ligand_sdf, complex_pdb


def ligand_template_for_coords(ligand_path: Path, coords: np.ndarray, filter_hs) -> tuple[Chem.Mol, np.ndarray]:
    from docking_model.data.parse.molecule import read_single_mol

    mol = read_single_mol(str(ligand_path), remove_hs=False)
    if mol is None:
        raise ValueError(f"Could not load ligand template from {ligand_path}")
    if mol.GetNumAtoms() == coords.shape[0]:
        return mol, coords

    heavy_mol = Chem.RemoveAllHs(Chem.Mol(mol))
    if heavy_mol.GetNumAtoms() == coords.shape[0]:
        return heavy_mol, coords

    if filter_hs is not None:
        mask = np.asarray(filter_hs, dtype=bool).reshape(-1)
        if mask.shape[0] == coords.shape[0] and heavy_mol.GetNumAtoms() == int(mask.sum()):
            return heavy_mol, coords[mask]

    raise ValueError(
        "Predicted ligand atom count does not match the ligand template: "
        f"coords={coords.shape[0]}, template_atoms={mol.GetNumAtoms()}, heavy_atoms={heavy_mol.GetNumAtoms()}."
    )


def mol_with_coordinates(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()
    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coord in enumerate(coords):
        conformer.SetAtomPosition(atom_idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))
    mol.AddConformer(conformer, assignId=True)
    return mol


def write_complex_pdb(protein_pdb: Path, ligand_pdb: Path, output_path: Path) -> Path:
    protein_lines = [
        line.rstrip("\n")
        for line in protein_pdb.read_text(errors="ignore").splitlines()
        if not line.startswith(("END", "ENDMDL", "MODEL"))
    ]
    ligand_lines = [
        line.rstrip("\n")
        for line in ligand_pdb.read_text(errors="ignore").splitlines()
        if line.startswith(("ATOM", "HETATM", "CONECT"))
    ]
    output_path.write_text("\n".join(protein_lines + ligand_lines + ["END", ""]))
    return output_path


if __name__ == "__main__":
    main()
