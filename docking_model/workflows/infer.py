from __future__ import annotations

import csv
import json
import logging
import os
import pickle
import re
import tempfile
import time
from pathlib import Path
from typing import Iterable

_runtime_cache = Path(tempfile.gettempdir()) / "docking-model-runtime-cache"
os.environ.setdefault("MPLCONFIGDIR", str(_runtime_cache / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_runtime_cache / "xdg"))

import numpy as np
import torch
from rdkit import Chem
from rdkit.Geometry import Point3D
from tqdm.auto import tqdm

from docking_model.data.datasets import CachedComplexDataset
from docking_model.data.feature.featurizer import FeaturizerConfig
from docking_model.data.modules.inference import PredictionDataset
from docking_model.data.parse.molecule import read_single_mol
from docking_model.data.transforms.docking import SetInitTimeTransformInference
from docking_model.data.write.trajectory import (
    denormalize_coords,
    write_protein_frame,
)
from docking_model.data.write.writer import write_docking_outputs
from docking_model.metrics.docking import compute_inference_sample_metrics
from docking_model.runtime.checkpoint import load_model_state
from docking_model.runtime.factory import build_sampler, build_score_model, build_transform, resolve_inference_checkpoint, select_device
from docking_model.runtime.inference import load_inference_config
from docking_model.runtime.loggers import build_experiment_logger
from docking_model.runtime.seeding import seed_everything
from docking_model.sampling.engine import SamplingResult, rank_sampling_result_by_confidence


def run_inference(
    model_parameters_path: str,
    model=None,
    sampler=None,
    inference_transform=None,
    overrides: Iterable[str] | None = None,
    show_progress: bool = False,
):
    cfg = load_inference_config(model_parameters_path, overrides=overrides)
    seed_everything(cfg.seed, workers=False)
    set_logger_job_name(cfg, "inference")
    logger = build_experiment_logger(cfg, job_type="inference")

    try:
        model_was_built = model is None
        model = model or build_score_model(cfg)
        if model_was_built:
            checkpoint = resolve_inference_checkpoint(cfg)
            if checkpoint is not None:
                load_model_state(model, checkpoint, strict=True, prefer_ema=cfg.inference.use_ema_weights)

        device = select_device(cfg.training.device)
        model.to(device)
        sampler = sampler or build_sampler(cfg)
        output_dir = Path(cfg.inference.output_dir).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)

        graph_iter, total_complexes, source = inference_graph_source(cfg, inference_transform)
        if show_progress:
            print(f"inference_source={source}")
            print(f"output_dir={output_dir}")
            print(f"complexes={total_complexes}")
            print(f"samples_per_complex={cfg.sampler.samples_per_complex}")
            print(f"inference_steps={cfg.sampler.inference_steps}")
            print(f"posebusters_metrics={cfg.inference.posebusters_metrics}")

        output_paths: list[Path] = []
        complex_visualization_rows: list[dict[str, object]] = []
        posebusters_rows: list[dict[str, object]] = []
        posebusters_csv_paths: list[Path] = []
        progress = (
            tqdm(graph_iter, total=total_complexes, desc="Inference complexes", unit="complex")
            if show_progress
            else graph_iter
        )

        for graph in progress:
            if not is_successful_graph(graph):
                logging.warning("Skipping failed inference graph %s", graph_name(graph))
                continue
            name = graph_name(graph)
            if show_progress:
                progress.set_postfix_str(name)

            start_time = time.perf_counter()
            result = rank_sampling_result_by_confidence(
                sampler.generate(data_list=[graph.to("cpu")], model=model, device=device)
            )
            docking_outputs = pack_docking_outputs(cfg, name, graph, result)
            docking_outputs["run_time"] = time.perf_counter() - start_time
            output_path = write_docking_outputs(
                docking_outputs=docking_outputs,
                output_dir=output_dir / name,
                export_trajectory_files=cfg.inference.save_trajectory and cfg.inference.export_trajectory_files,
                trajectory_max_ranks=cfg.inference.trajectory_max_ranks,
            )
            output_paths.append(output_path)
            if show_progress:
                tqdm.write(f"{name}: wrote {output_path}")

            if not cfg.inference.posebusters_metrics:
                complex_visualization_rows.append(complex_visualization_row_from_outputs(name, docking_outputs))

            complex_posebusters_rows = posebusters_csv_rows(name, docking_outputs)
            if complex_posebusters_rows:
                complex_csv_path = write_csv(output_dir / name / "posebusters_metrics.csv", complex_posebusters_rows)
                posebusters_rows.extend(complex_posebusters_rows)
                posebusters_csv_paths.append(complex_csv_path)
                if show_progress:
                    tqdm.write(f"{name}: wrote {complex_csv_path}")

        posebusters_summary = summarize_posebusters_rows(posebusters_rows)
        if posebusters_rows:
            aggregate_csv_path = write_csv(output_dir / "posebusters_metrics.csv", posebusters_rows)
            posebusters_csv_paths.append(aggregate_csv_path)
        elif cfg.inference.posebusters_metrics:
            raise RuntimeError(
                "PoseBusters metrics were requested, but no PoseBusters rows were produced. "
                f"Expected at least one per-complex PoseBusters result under {output_dir}."
            )

        logger.log({
            "inference_num_complexes": float(len(output_paths)),
            "inference_num_samples_total": float(len(output_paths) * int(cfg.sampler.samples_per_complex)),
            "inference_samples_mean": float(cfg.sampler.samples_per_complex),
            **posebusters_summary,
        })
        log_inference_wandb_artifacts(logger, cfg, output_dir, posebusters_rows, complex_visualization_rows)

        if show_progress:
            print(f"inference_batches={len(output_paths)}")
            if output_paths:
                print("prediction_files:")
                for output_path in output_paths:
                    print(f"  {output_path}")
            if posebusters_csv_paths:
                print("posebusters_csv:")
                for csv_path in posebusters_csv_paths:
                    print(f"  {csv_path}")
                for key in sorted(posebusters_summary):
                    print(f"{key}={posebusters_summary[key]}")
            elif cfg.inference.posebusters_metrics:
                print("posebusters_metrics=none")
        return output_paths
    finally:
        if cfg.logger.finish_on_exit:
            logger.finish()


def evaluate_saved_inference(
    model_parameters_path: str,
    overrides: Iterable[str] | None = None,
    show_progress: bool = False,
) -> dict[str, float]:
    cfg = load_inference_config(model_parameters_path, overrides=overrides)
    seed_everything(cfg.seed, workers=False)
    set_logger_job_name(cfg, "inference-eval")
    logger = build_experiment_logger(cfg, job_type="inference-eval")

    try:
        output_dir = Path(cfg.inference.output_dir).expanduser()
        graph_iter, total_complexes, source = inference_graph_source(cfg, inference_transform=None)
        metric_csv_path = (
            Path(cfg.inference.results_table_csv).expanduser()
            if cfg.inference.results_table_csv is not None
            else output_dir / "inference_metrics.csv"
        )
        metric_rows: list[dict[str, float]] = []
        metric_csv_rows: list[dict[str, object]] = []
        complex_visualization_rows: list[dict[str, object]] = []
        progress = (
            tqdm(graph_iter, total=total_complexes, desc="Inference complexes", unit="complex")
            if show_progress
            else graph_iter
        )

        if show_progress:
            print(f"inference_source={source}")
            print(f"output_dir={output_dir}")
            print(f"complexes={total_complexes}")
            print(f"posebusters_metrics={cfg.inference.posebusters_metrics}")

        for graph in progress:
            if not is_successful_graph(graph):
                continue

            name = graph_name(graph)
            if show_progress:
                progress.set_postfix_str(name)
            output_path = output_dir / name / "docking_predictions.pkl"
            if not output_path.exists():
                logging.warning("Skipping %s because %s is missing", name, output_path)
                continue

            with output_path.open("rb") as handle:
                docking_outputs = pickle.load(handle)

            predictions = prediction_records_from_outputs(docking_outputs)
            result = rank_sampling_result_by_confidence(
                SamplingResult(
                    predictions=predictions,
                    confidences=docking_outputs["confidence"],
                    details={},
                )
            )
            if cfg.inference.posebusters_metrics:
                row_metrics = {}
            else:
                row_metrics = inference_metrics(name, graph, result)
            if row_metrics:
                print_ligand_rmsd(name, row_metrics)
                metric_rows.append(row_metrics)
                metric_csv_rows.append({"complex_name": name, **row_metrics})
                complex_visualization_rows.append(
                    complex_visualization_row_from_outputs(name, docking_outputs, row_metrics)
                )

        if metric_csv_rows:
            write_csv(metric_csv_path, metric_csv_rows)

        summary_metrics = summarize_inference_metrics(metric_rows)
        posebusters_csv_path = output_dir / "posebusters_metrics.csv"
        posebusters_rows = read_posebusters_rows(output_dir)
        posebusters_rows = add_saved_posebusters_row_values(output_dir, posebusters_rows)
        if cfg.inference.posebusters_metrics and posebusters_rows:
            write_csv(posebusters_csv_path, posebusters_rows)
        posebusters_summary = summarize_posebusters_rows(posebusters_rows)
        if cfg.inference.posebusters_metrics and not posebusters_summary:
            raise RuntimeError(
                "PoseBusters metrics were requested, but no saved PoseBusters CSV rows were found. "
                f"Expected {posebusters_csv_path} or per-complex posebusters_metrics.csv files under {output_dir}."
            )
        summary_metrics.update(posebusters_summary)

        if summary_metrics:
            logger.log(summary_metrics)
        log_inference_wandb_artifacts(logger, cfg, output_dir, posebusters_rows, complex_visualization_rows)
        if show_progress:
            print(f"inference_metrics_csv={metric_csv_path}")
            if posebusters_summary:
                print(f"posebusters_metrics_csv={posebusters_csv_path}")
            for key in sorted(summary_metrics):
                print(f"{key}={summary_metrics[key]}")
        return summary_metrics
    finally:
        if cfg.logger.finish_on_exit:
            logger.finish()


def inference_graph_source(cfg, inference_transform):
    if cfg.inference.input_csv is not None:
        dataset = PredictionDataset(
            input_csv=cfg.inference.input_csv,
            featurizer_cfg=FeaturizerConfig(
                matching=False,
                remove_hs=True,
                max_lig_size=cfg.data.max_lig_size,
                all_atoms=cfg.pocket.all_atoms,
                flexible_backbone=cfg.protein.flexible_backbone,
                flexible_sidechains=cfg.protein.flexible_sidechains,
            ),
            esm_embeddings_path=cfg.inference.esm_embeddings_path or cfg.model.esm_embeddings_path,
            limit_complexes=cfg.inference.limit_complexes,
            complex_id=cfg.inference.complex_id,
            pocket_reduction=cfg.inference.pocket_reduction,
            pocket_radius=cfg.inference.pocket_radius,
            pocket_buffer=cfg.inference.pocket_buffer,
            pocket_min_size=cfg.inference.pocket_min_size,
            only_nearby_residues_atomic=cfg.inference.only_nearby_residues_atomic,
            nearby_residues_selection_mode=cfg.inference.nearby_residues_selection_mode,
            nearby_residues_atomic_radius=cfg.inference.nearby_residues_atomic_radius,
            nearby_residues_atomic_min=cfg.inference.nearby_residues_atomic_min,
            all_atoms=cfg.inference.all_atoms,
            flexible_backbone=cfg.inference.flexible_backbone,
            flexible_sidechains=cfg.inference.flexible_sidechains,
            load_reference_metrics=not cfg.inference.posebusters_metrics,
        )
        set_time = SetInitTimeTransformInference(
            flexible_backbone=cfg.protein.flexible_backbone,
            flexible_sidechains=cfg.protein.flexible_sidechains,
            all_atoms=cfg.pocket.all_atoms,
        )

        def graphs():
            for idx in range(len(dataset)):
                graph = dataset.get(idx)
                if not is_successful_graph(graph):
                    yield graph
                    continue
                yield set_time(graph)

        return graphs(), len(dataset), f"input_csv={cfg.inference.input_csv}"

    transform = inference_transform or build_transform(cfg, mode="inference")
    cache_path = cfg.inference.cache_path or cfg.data.cache_path
    split_path = cfg.inference.split_path or cfg.data.split_val
    if cache_path is None or split_path is None:
        raise ValueError("Inference requires either inference.input_csv or cache_path/split_path.")
    dataset = CachedComplexDataset(
        cache_path=cache_path,
        split_path=split_path,
        transform=transform,
        affinity_csv=cfg.data.affinity_csv,
        esm_embeddings_path=cfg.inference.esm_embeddings_path or cfg.model.esm_embeddings_path,
        limit_complexes=cfg.inference.limit_complexes,
        multiplicity=1,
    )

    def graphs():
        for idx in range(len(dataset)):
            yield dataset[idx]

    return graphs(), len(dataset), f"cache_path={cache_path} split_path={split_path}"


def pack_docking_outputs(cfg, name: str, graph, result) -> dict:
    predictions = result.predictions
    confidence = to_numpy(result.confidences)
    if confidence is None:
        raise ValueError(f"{name}: inference result has no confidence scores.")

    atom_pos = []
    ligand_pos = []
    for prediction in predictions:
        atom_coords = to_numpy(prediction["atom"].pos)
        ligand_coords = to_numpy(prediction["ligand"].pos)
        if not np.isfinite(atom_coords).all():
            raise ValueError(f"{name}: non-finite values in predicted protein atom positions.")
        if not np.isfinite(ligand_coords).all():
            raise ValueError(f"{name}: non-finite values in predicted ligand positions.")
        atom_pos.append(atom_coords)
        ligand_pos.append(ligand_coords)

    details = result.details or {}
    atom_store = graph["atom"]
    outputs = {
        "name": name,
        "atom_pos": atom_pos,
        "ligand_pos": ligand_pos,
        "confidence": confidence,
        "ligand_trajectory": details.get("ligand_trajectory"),
        "atom_trajectory": details.get("atom_trajectory"),
        "original_center": to_numpy(graph._global_store["original_center"]) if "original_center" in graph._global_store else None,
        "filterHs": to_numpy(graph["ligand"].x[:, 0] != 0),
        "atom_mask": to_numpy(atom_store.atom_mask) if "atom_mask" in atom_store else None,
        "ca_mask": to_numpy(atom_store.ca_mask) if "ca_mask" in atom_store else None,
        "c_mask": to_numpy(atom_store.c_mask) if "c_mask" in atom_store else None,
        "n_mask": to_numpy(atom_store.n_mask) if "n_mask" in atom_store else None,
        "pocket_atom_mask": to_numpy(atom_store.nearby_atoms) if "nearby_atoms" in atom_store else None,
    }
    for key in [
        "base_dir",
        "apo_rec_path",
        "holo_rec_path",
        "holo_rec_path_for_metrics",
        "ligand_input",
        "ligand_true_file",
        "pocket_ligand_file",
        "pocket_residues",
        "ligand_description",
    ]:
        outputs[key] = metadata_value(graph, key)
    if cfg.inference.posebusters_metrics:
        outputs["posebusters_metrics"] = run_posebusters(outputs)
    return outputs


def run_posebusters(docking_outputs: dict) -> dict[str, np.ndarray]:
    name = docking_outputs["name"]
    from posebusters import PoseBusters

    predicted_mols = posebusters_prediction_mols(docking_outputs)
    reference_ligand_path = posebusters_path(docking_outputs, "ligand_true_file")
    buster = PoseBusters(
        config="redock" if reference_ligand_path is not None else "dock",
        top_n=None,
    )
    if reference_ligand_path is not None:
        buster.config["loading"]["mol_true"]["load_all"] = False

    report = buster.bust(
        predicted_mols,
        mol_true=reference_ligand_path,
        mol_cond=posebusters_path(docking_outputs, "holo_rec_path_for_metrics")
        or posebusters_path(docking_outputs, "holo_rec_path")
        or posebusters_path(docking_outputs, "apo_rec_path"),
        full_report=False,
    )

    if report.empty:
        raise RuntimeError(f"{name}: PoseBusters returned an empty report.")

    if reference_ligand_path is None:
        report = report.drop(columns=reference_posebusters_columns(report), errors="ignore")

    details = {
        (column_name if column_name.startswith("posebusters_") else f"posebusters_{column_name}"): report[column].to_numpy()
        for column in report.columns
        for column_name in [str(column).lower().replace(" ", "_")]
    }
    bool_columns = [
        column
        for column in report.columns
        if not report[column].dropna().empty and report[column].dropna().isin([True, False]).all()
    ]

    if bool_columns:
        passed = report[bool_columns].fillna(False).astype(bool)
        details["posebusters_pass_count"] = passed.sum(axis=1).to_numpy()
        details["posebusters_all_pass"] = passed.all(axis=1).to_numpy()

    if reference_ligand_path is not None:
        details["posebusters_ligand_rmsd"] = posebusters_ligand_rmsds(
            predicted_mols=predicted_mols,
            reference_ligand_path=reference_ligand_path,
        )

    return details


def posebusters_ligand_rmsds(
    predicted_mols: list[Chem.Mol],
    reference_ligand_path: Path,
) -> np.ndarray:
    reference_mol = read_single_mol(str(reference_ligand_path), remove_hs=False)
    if reference_mol is None:
        return np.full(len(predicted_mols), np.nan, dtype=float)

    reference_mol = Chem.RemoveAllHs(Chem.Mol(reference_mol))
    if reference_mol.GetNumConformers() == 0:
        return np.full(len(predicted_mols), np.nan, dtype=float)
    reference_coords = np.asarray(reference_mol.GetConformer().GetPositions(), dtype=np.float32)

    rmsds = []
    for predicted_mol in predicted_mols:
        predicted_mol = Chem.RemoveAllHs(Chem.Mol(predicted_mol))
        if predicted_mol.GetNumConformers() == 0 or predicted_mol.GetNumAtoms() != reference_mol.GetNumAtoms():
            rmsds.append(np.nan)
            continue

        predicted_coords = np.asarray(predicted_mol.GetConformer().GetPositions(), dtype=np.float32)
        rmsds.append(symmetry_ligand_rmsd(reference_mol, predicted_mol, reference_coords, predicted_coords))

    return np.asarray(rmsds, dtype=float)


def symmetry_ligand_rmsd(
    reference_mol: Chem.Mol,
    predicted_mol: Chem.Mol,
    reference_coords: np.ndarray,
    predicted_coords: np.ndarray,
) -> float:
    try:
        from docking_model.data.conformers.molecule import get_symmetry_rmsd

        return float(
            get_symmetry_rmsd(
                reference_mol,
                reference_coords,
                [predicted_coords],
                mol2=predicted_mol,
            )[0]
        )
    except Exception:
        return float(np.sqrt(np.mean(np.sum((predicted_coords - reference_coords) ** 2, axis=-1))))


def reference_posebusters_columns(report) -> list[str]:
    reference_columns = []
    reference_terms = (
        "mol_true",
        "molecular_formula",
        "molecular_bonds",
        "double_bond_stereochemistry",
        "tetrahedral_chirality",
        "rmsd",
    )
    for column in report.columns:
        normalized = str(column).lower()
        if any(term in normalized for term in reference_terms):
            reference_columns.append(column)
    return reference_columns


def posebusters_prediction_mols(docking_outputs: dict) -> list[Chem.Mol]:
    template_path = posebusters_path(docking_outputs, "ligand_input") or posebusters_path(docking_outputs, "ligand_true_file")
    if template_path is None:
        raise ValueError(f"{docking_outputs['name']}: PoseBusters requires ligand_input or ligand_true_file.")

    template = read_single_mol(str(template_path), remove_hs=False)
    if template is None:
        raise ValueError(f"{docking_outputs['name']}: failed to load ligand template {template_path}.")

    mols = []
    original_center = to_numpy(docking_outputs["original_center"])
    filter_hs = docking_outputs["filterHs"]

    for coords in docking_outputs["ligand_pos"]:
        coords = np.asarray(coords, dtype=np.float64)
        if original_center is not None:
            center = np.asarray(original_center, dtype=np.float64).reshape(-1)
            if center.size == 3:
                coords = coords + center.reshape(1, 3)
        mols.append(mol_with_prediction_coords(template, coords, filter_hs))

    return mols


def posebusters_path(docking_outputs: dict, key: str) -> Path | None:
    value = docking_outputs.get(key)
    if value is None:
        return None

    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return path

    base_dir = docking_outputs["base_dir"]
    return Path(str(base_dir)).expanduser() / path if base_dir is not None else path


def mol_with_prediction_coords(template: Chem.Mol, coords: np.ndarray, filter_hs) -> Chem.Mol:
    mol = Chem.Mol(template)
    heavy_mol = Chem.RemoveAllHs(Chem.Mol(template))

    if mol.GetNumAtoms() == coords.shape[0]:
        return make_mol_with_coordinates(mol, coords)

    if heavy_mol.GetNumAtoms() == coords.shape[0]:
        return make_mol_with_coordinates(heavy_mol, coords)

    if filter_hs is not None:
        mask = np.asarray(filter_hs, dtype=bool).reshape(-1)
        if mask.shape[0] == coords.shape[0] and heavy_mol.GetNumAtoms() == int(mask.sum()):
            return make_mol_with_coordinates(heavy_mol, coords[mask])

    raise ValueError(
        "Predicted ligand atom count does not match the input ligand template: "
        f"coords={coords.shape[0]}, template_atoms={mol.GetNumAtoms()}, heavy_atoms={heavy_mol.GetNumAtoms()}."
    )


def make_mol_with_coordinates(mol: Chem.Mol, coords: np.ndarray) -> Chem.Mol:
    mol = Chem.Mol(mol)
    mol.RemoveAllConformers()

    conformer = Chem.Conformer(mol.GetNumAtoms())
    for atom_idx, coord in enumerate(coords):
        conformer.SetAtomPosition(atom_idx, Point3D(float(coord[0]), float(coord[1]), float(coord[2])))

    mol.AddConformer(conformer, assignId=True)
    return mol


def posebusters_csv_rows(name: str, docking_outputs: dict) -> list[dict[str, object]]:
    if "posebusters_metrics" not in docking_outputs:
        return []
    metrics = docking_outputs["posebusters_metrics"]
    if not metrics:
        return []

    values_by_key = {key: np.asarray(to_numpy(value), dtype=object).reshape(-1) for key, value in metrics.items()}
    num_rows = max((len(values) for values in values_by_key.values()), default=0)
    if num_rows == 0:
        return []

    confidence_values = np.asarray(to_numpy(docking_outputs["confidence"]), dtype=object)
    if confidence_values.ndim > 1:
        confidence_values = confidence_values.reshape(confidence_values.shape[0], -1)[:, 0]
    else:
        confidence_values = confidence_values.reshape(-1)

    rows = []
    for index in range(num_rows):
        row: dict[str, object] = {"complex_name": name, "rank": index + 1}
        if index < len(confidence_values):
            row["confidence"] = confidence_values[index]

        for key, values in values_by_key.items():
            row[key] = values[index] if index < len(values) else ""
        rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def read_csv(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []

    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def read_posebusters_rows(output_dir: Path) -> list[dict[str, object]]:
    rows = read_csv(output_dir / "posebusters_metrics.csv")
    if rows:
        return rows

    all_rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*/posebusters_metrics.csv")):
        all_rows.extend(read_csv(path))
    return all_rows


def summarize_posebusters_rows(rows: list[dict[str, object]]) -> dict[str, float]:
    if not rows:
        return {}

    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["complex_name"]), []).append(row)
    summary = {
        "posebusters_num_complexes": float(len(grouped_rows)),
        "posebusters_num_samples": float(len(rows)),
    }

    posebench_columns = [column for column in _POSEBUSTERS_POSEBENCH_COLUMNS if column in rows[0]]
    for column in posebench_columns:
        summary[posebusters_summary_name(column)] = float(
            np.mean([posebusters_flag(row, column) for row in rows])
        )

    pb_valid_values = [posebusters_pb_valid(row) for row in rows]
    summary["posebusters_pb_valid"] = float(np.mean(pb_valid_values))
    top1_rows = [min(group, key=lambda row: int(row["rank"])) for group in grouped_rows.values()]
    top1_pb_valid = [posebusters_pb_valid(row) for row in top1_rows]
    summary["posebusters_top1_conf_pb_valid"] = float(np.mean(top1_pb_valid))
    summary["posebusters_oracle_pb_valid"] = float(
        np.mean([any(posebusters_pb_valid(row) for row in group) for group in grouped_rows.values()])
    )

    if any(int(row["rank"]) >= 5 for row in rows):
        top5_groups = [
            [row for row in group if int(row["rank"]) <= 5]
            for group in grouped_rows.values()
        ]
        summary["posebusters_top5_conf_pb_valid"] = float(
            np.mean([any(posebusters_pb_valid(row) for row in group) for group in top5_groups])
        )

    if _POSEBUSTERS_RMSD_LE_2_COLUMN in rows[0]:
        rmsd_le_2_values = [posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) for row in rows]
        summary["posebusters_rmsd_le_2_and_pb_valid"] = float(
            np.mean([
                rmsd and pb_valid
                for rmsd, pb_valid in zip(rmsd_le_2_values, pb_valid_values)
            ])
        )

        top1_rmsd = [posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) for row in top1_rows]
        summary["posebusters_top1_conf_rmsd_le_2"] = float(np.mean(top1_rmsd))
        summary["posebusters_top1_conf_rmsd_le_2_and_pb_valid"] = float(
            np.mean([
                rmsd and pb_valid
                for rmsd, pb_valid in zip(top1_rmsd, top1_pb_valid)
            ])
        )

        summary["posebusters_oracle_rmsd_le_2"] = float(
            np.mean([
                any(posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) for row in group)
                for group in grouped_rows.values()
            ])
        )
        summary["posebusters_oracle_rmsd_le_2_and_pb_valid"] = float(
            np.mean([
                any(
                    posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) and posebusters_pb_valid(row)
                    for row in group
                )
                for group in grouped_rows.values()
            ])
        )

    if _POSEBUSTERS_RMSD_LE_2_COLUMN in rows[0] and any(int(row["rank"]) >= 5 for row in rows):
        top5_groups = [
            [row for row in group if int(row["rank"]) <= 5]
            for group in grouped_rows.values()
        ]
        summary["posebusters_top5_conf_rmsd_le_2"] = float(
            np.mean([
                any(posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) for row in group)
                for group in top5_groups
            ])
        )
        summary["posebusters_top5_conf_rmsd_le_2_and_pb_valid"] = float(
            np.mean([
                any(
                    posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN) and posebusters_pb_valid(row)
                    for row in group
                )
                for group in top5_groups
            ])
        )

    return summary


def add_saved_posebusters_row_values(output_dir: Path, rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        return rows

    enriched_rows = [dict(row) for row in rows]
    rows_by_name: dict[str, list[dict[str, object]]] = {}
    for row in enriched_rows:
        rows_by_name.setdefault(str(row["complex_name"]), []).append(row)

    for name, complex_rows in rows_by_name.items():
        needs_confidence = any(numeric_value(row.get("confidence")) is None for row in complex_rows)
        output_path = output_dir / name / "docking_predictions.pkl"
        if not output_path.exists():
            continue

        try:
            with output_path.open("rb") as handle:
                docking_outputs = pickle.load(handle)

            confidence = to_numpy(docking_outputs.get("confidence"))
            confidence_values = (
                np.asarray(confidence, dtype=float).reshape(-1)
                if confidence is not None
                else np.asarray([], dtype=float)
            )
            ligand_rmsds = None
            reference_ligand_path = posebusters_path(docking_outputs, "ligand_true_file")
            if reference_ligand_path is not None:
                ligand_rmsds = posebusters_ligand_rmsds(
                    predicted_mols=posebusters_prediction_mols(docking_outputs),
                    reference_ligand_path=reference_ligand_path,
                )
        except Exception as exc:
            logging.warning("%s: failed to enrich saved PoseBusters rows due to %s", name, exc)
            continue

        for row in complex_rows:
            rank_index = int(row["rank"]) - 1
            if needs_confidence and rank_index < len(confidence_values):
                row["confidence"] = float(confidence_values[rank_index])
            if ligand_rmsds is not None and rank_index < len(ligand_rmsds):
                row["posebusters_ligand_rmsd"] = float(ligand_rmsds[rank_index])

    return enriched_rows


def log_inference_wandb_artifacts(
    logger,
    cfg,
    output_dir: Path,
    posebusters_rows: list[dict[str, object]],
    complex_rows: list[dict[str, object]],
) -> None:
    wandb = getattr(logger, "wandb", None)
    if wandb is None:
        return

    artifacts = {}
    if cfg.inference.posebusters_metrics and posebusters_rows:
        artifacts.update(posebusters_confidence_rmsd_artifacts(wandb, posebusters_rows))
        complex_rows = posebusters_rows

    max_examples = int(cfg.inference.wandb_max_complex_examples or 0)
    if max_examples > 0 and complex_rows:
        molecule_table = complex_visualization_table(wandb, output_dir, complex_rows, max_examples)
        if molecule_table is not None:
            artifacts["predicted_complexes"] = molecule_table

    logger.log_artifacts(artifacts)


def posebusters_confidence_rmsd_artifacts(wandb, rows: list[dict[str, object]]) -> dict[str, object]:
    table = wandb.Table(
        columns=[
            "complex_name",
            "rank",
            "confidence",
            "posebusters_ligand_rmsd",
            "posebusters_rmsd_le_2",
            "posebusters_pb_valid",
        ]
    )

    row_count = 0
    for row in rows:
        confidence = numeric_value(row.get("confidence"))
        ligand_rmsd = posebusters_ligand_rmsd_value(row)
        if confidence is None or ligand_rmsd is None:
            continue

        table.add_data(
            str(row["complex_name"]),
            int(row["rank"]),
            confidence,
            ligand_rmsd,
            posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN)
            if _POSEBUSTERS_RMSD_LE_2_COLUMN in row
            else None,
            posebusters_pb_valid(row),
        )
        row_count += 1

    if row_count == 0:
        return {}

    return {
        "posebusters_confidence_rmsd_table": table,
        "posebusters_confidence_vs_ligand_rmsd": wandb.plot.scatter(
            table,
            "confidence",
            "posebusters_ligand_rmsd",
            title="PoseBusters confidence vs ligand RMSD",
        ),
    }


def complex_visualization_table(wandb, output_dir: Path, rows: list[dict[str, object]], max_examples: int):
    selected_rows = select_complex_visualization_rows(rows, max_examples)
    if not selected_rows:
        return None

    table = wandb.Table(
        columns=[
            "complex_name",
            "rank",
            "confidence",
            "ligand_rmsd",
            "rmsd_le_2",
            "pb_valid",
            "legend",
            "pdb",
            "html",
        ]
    )

    row_count = 0
    for row in selected_rows:
        pdb_path, html_path = write_wandb_complex_files(output_dir, row)
        if pdb_path is None:
            continue

        name = str(row["complex_name"])
        rank = int(row["rank"])
        ligand_rmsd = complex_ligand_rmsd_value(row)
        table.add_data(
            name,
            rank,
            numeric_value(row.get("confidence")),
            ligand_rmsd,
            posebusters_flag(row, _POSEBUSTERS_RMSD_LE_2_COLUMN)
            if _POSEBUSTERS_RMSD_LE_2_COLUMN in row
            else threshold_bool(ligand_rmsd, 2.0),
            posebusters_pb_valid(row) if _POSEBUSTERS_POSEBENCH_COLUMNS[1] in row else None,
            "A predicted protein; B predicted ligand yellow; C true protein; D true ligand blue",
            wandb.Molecule(str(pdb_path), caption=f"{name} rank {rank}"),
            wandb.Html(html_path.read_text(), inject=False) if html_path is not None else None,
        )
        row_count += 1

    return table if row_count else None


def select_complex_visualization_rows(rows: list[dict[str, object]], max_examples: int) -> list[dict[str, object]]:
    grouped_rows: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["complex_name"]), []).append(row)

    top_rows = [
        min(group, key=lambda row: int(row["rank"]))
        for _, group in sorted(grouped_rows.items())
    ]
    if len(top_rows) <= max_examples:
        return top_rows

    selected_indices = np.linspace(0, len(top_rows) - 1, max_examples, dtype=int)
    return [top_rows[index] for index in selected_indices]


def write_wandb_complex_files(output_dir: Path, row: dict[str, object]) -> tuple[Path | None, Path | None]:
    name = str(row["complex_name"])
    rank_index = int(row["rank"]) - 1
    output_path = output_dir / name / "docking_predictions.pkl"
    if not output_path.exists():
        return None, None

    with output_path.open("rb") as handle:
        docking_outputs = pickle.load(handle)

    atom_positions = docking_outputs.get("atom_pos") or []
    if rank_index >= len(atom_positions):
        return None, None

    try:
        ligand_mol = posebusters_prediction_mols(docking_outputs)[rank_index]
        true_ligand_mol = None
        true_ligand_path = posebusters_path(docking_outputs, "ligand_true_file") or posebusters_path(
            docking_outputs,
            "pocket_ligand_file",
        )
        if true_ligand_path is not None:
            true_ligand_mol = read_single_mol(str(true_ligand_path), remove_hs=False)

        true_protein_path = (
            posebusters_path(docking_outputs, "holo_rec_path_for_metrics")
            or posebusters_path(docking_outputs, "holo_rec_path")
        )
        original_center = to_numpy(docking_outputs.get("original_center"))
        atom_coords = denormalize_coords(atom_positions[rank_index], original_center)

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
        complex_dir = output_dir / "wandb_complexes"
        protein_path = complex_dir / f"{safe_name}_rank{rank_index + 1}_protein.pdb"
        complex_path = complex_dir / f"{safe_name}_rank{rank_index + 1}.pdb"
        html_path = complex_dir / f"{safe_name}_rank{rank_index + 1}.html"
        write_protein_frame(
            docking_outputs.get("apo_rec_path"),
            docking_outputs.get("atom_mask"),
            atom_coords,
            protein_path,
        )

        serial = 1
        protein_lines, serial = pdb_atom_lines_with_chain(
            protein_path.read_text().splitlines(),
            chain_id="A",
            start_serial=serial,
        )
        ligand_lines, ligand_conect_lines, serial = pdb_block_lines_with_chain(
            Chem.MolToPDBBlock(Chem.RemoveAllHs(Chem.Mol(ligand_mol))).splitlines(),
            chain_id="B",
            start_serial=serial,
        )

        true_protein_lines = []
        if true_protein_path is not None and true_protein_path.exists():
            true_protein_lines, serial = pdb_atom_lines_with_chain(
                [line for line in true_protein_path.read_text().splitlines() if line.startswith("ATOM")],
                chain_id="C",
                start_serial=serial,
            )

        true_ligand_lines = []
        true_ligand_conect_lines = []
        if true_ligand_mol is not None:
            true_ligand_lines, true_ligand_conect_lines, serial = pdb_block_lines_with_chain(
                Chem.MolToPDBBlock(Chem.RemoveAllHs(Chem.Mol(true_ligand_mol))).splitlines(),
                chain_id="D",
                start_serial=serial,
            )

        legend_lines = [
            "REMARK WandB overlay legend:",
            "REMARK Chain A: predicted protein",
            "REMARK Chain B: predicted ligand, yellow in HTML viewer",
            "REMARK Chain C: true/reference protein",
            "REMARK Chain D: true/reference ligand, blue in HTML viewer",
        ]
        complex_path.write_text(
            "\n".join(
                legend_lines
                + protein_lines
                + ligand_lines
                + true_protein_lines
                + true_ligand_lines
                + ligand_conect_lines
                + true_ligand_conect_lines
                + ["END", ""]
            )
        )
        html_path.write_text(colored_complex_html(complex_path.read_text(), name, rank_index + 1))
        return complex_path, html_path
    except Exception as exc:
        logging.warning("%s: failed to write WandB complex visualization due to %s", name, exc)
        return None, None


def complex_visualization_row_from_outputs(
    name: str,
    docking_outputs: dict,
    metrics: dict[str, float] | None = None,
) -> dict[str, object]:
    confidence = to_numpy(docking_outputs.get("confidence"))
    confidence_values = (
        np.asarray(confidence, dtype=float).reshape(-1)
        if confidence is not None
        else np.asarray([], dtype=float)
    )
    row: dict[str, object] = {"complex_name": name, "rank": 1}
    if confidence_values.size:
        row["confidence"] = float(confidence_values[0])
    if metrics:
        row.update(metrics)
    return row


def complex_ligand_rmsd_value(row: dict[str, object]) -> float | None:
    for key in ["posebusters_ligand_rmsd", "top1_conf_mean_ligand_rmsd", "ligand_rmsd"]:
        if key in row:
            value = numeric_value(row.get(key))
            if value is not None:
                return value
    return None


def threshold_bool(value: float | None, cutoff: float) -> bool | None:
    return None if value is None else bool(value < cutoff)


def colored_complex_html(pdb_text: str, name: str, rank: int) -> str:
    pdb_json = json.dumps(pdb_text)
    title = json.dumps(f"{name} rank {rank}")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #ffffff; }}
    #viewer {{ width: 100%; height: 520px; position: relative; }}
    #legend {{
      position: absolute;
      top: 10px;
      left: 10px;
      z-index: 10;
      background: rgba(255, 255, 255, 0.92);
      border: 1px solid #d0d0d0;
      border-radius: 6px;
      padding: 8px 10px;
      font-size: 12px;
      line-height: 1.4;
      color: #222;
    }}
    .item {{ display: flex; align-items: center; gap: 6px; }}
    .swatch {{ width: 12px; height: 12px; border-radius: 2px; display: inline-block; }}
  </style>
</head>
<body>
  <div id="viewer">
    <div id="legend">
      <div><b id="title"></b></div>
      <div class="item"><span class="swatch" style="background:#f2c200"></span>predicted ligand</div>
      <div class="item"><span class="swatch" style="background:#1f77ff"></span>true ligand</div>
      <div class="item"><span class="swatch" style="background:#b8b8b8"></span>predicted protein</div>
      <div class="item"><span class="swatch" style="background:#8fd3ff"></span>true protein</div>
    </div>
  </div>
  <script>
    const pdb = {pdb_json};
    document.getElementById("title").textContent = {title};
    const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
    viewer.addModel(pdb, "pdb");
    viewer.setStyle({{ chain: "A" }}, {{ cartoon: {{ color: "#b8b8b8", opacity: 0.55 }} }});
    viewer.setStyle({{ chain: "C" }}, {{ cartoon: {{ color: "#8fd3ff", opacity: 0.45 }} }});
    viewer.setStyle({{ chain: "B" }}, {{ stick: {{ color: "#f2c200", radius: 0.22 }} }});
    viewer.setStyle({{ chain: "D" }}, {{ stick: {{ color: "#1f77ff", radius: 0.22 }} }});
    viewer.zoomTo({{ chain: "B" }});
    viewer.zoom(0.75);
    viewer.render();
  </script>
</body>
</html>
"""


def pdb_block_lines_with_chain(lines: list[str], chain_id: str, start_serial: int) -> tuple[list[str], list[str], int]:
    atom_lines, serial_map, next_serial = pdb_atom_lines_with_chain(
        lines,
        chain_id=chain_id,
        start_serial=start_serial,
        return_serial_map=True,
    )
    conect_lines = pdb_conect_lines(lines, serial_map)
    return atom_lines, conect_lines, next_serial


def pdb_atom_lines_with_chain(
    lines: list[str],
    chain_id: str,
    start_serial: int,
    return_serial_map: bool = False,
):
    atom_lines = []
    serial_map = {}
    next_serial = start_serial

    for line in lines:
        if not line.startswith(("ATOM", "HETATM")):
            continue

        old_serial = pdb_atom_serial(line)
        serial_map[old_serial] = next_serial
        atom_lines.append(pdb_chain_line(line, chain_id, next_serial))
        next_serial += 1

    if return_serial_map:
        return atom_lines, serial_map, next_serial
    return atom_lines, next_serial


def pdb_conect_lines(lines: list[str], serial_map: dict[int | None, int]) -> list[str]:
    conect_lines = []
    for line in lines:
        if not line.startswith("CONECT"):
            continue

        old_serials = [
            int(line[index:index + 5])
            for index in range(6, len(line), 5)
            if line[index:index + 5].strip()
        ]
        new_serials = [serial_map[serial] for serial in old_serials if serial in serial_map]
        if len(new_serials) >= 2:
            conect_lines.append("CONECT" + "".join(f"{serial:5d}" for serial in new_serials))

    return conect_lines


def pdb_atom_serial(line: str) -> int | None:
    try:
        return int(line[6:11])
    except ValueError:
        return None


def pdb_chain_line(line: str, chain_id: str, serial: int) -> str:
    line = line.rstrip("\n")
    if len(line) < 80:
        line = line.ljust(80)

    line = f"{line[:6]}{serial:5d}{line[11:]}"
    return f"{line[:21]}{chain_id[:1]}{line[22:]}"


def posebusters_ligand_rmsd_value(row: dict[str, object]) -> float | None:
    for key in ["posebusters_ligand_rmsd", "posebusters_rmsd"]:
        if key in row:
            value = numeric_value(row.get(key))
            if value is not None:
                return value
    return None


def numeric_value(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value == "":
        return None
    if isinstance(value, np.generic):
        value = value.item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def posebusters_pb_valid(row: dict[str, object]) -> bool:
    available_columns = [column for column in _POSEBUSTERS_POSEBENCH_COLUMNS[1:] if column in row]
    return bool(available_columns) and all(posebusters_flag(row, column) for column in available_columns)


def posebusters_flag(row: dict[str, object], column: str) -> bool:
    value = row[column]
    if value == "":
        return False
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value) == "True"


def posebusters_summary_name(column: str) -> str:
    name = str(column)
    if name.startswith("posebusters_"):
        name = name[len("posebusters_"):]
    name = name.lower()
    name = name.replace("≤", "_le_")
    name = name.replace("≥", "_ge_")
    name = name.replace("å", "a")
    name = name.replace("-", "_")
    name = name.replace(" ", "_")
    name = re.sub(r"[^a-z0-9_]+", "", name)
    name = re.sub(r"_+", "_", name)
    return f"posebusters_{name.strip('_')}"


def prediction_records_from_outputs(docking_outputs: dict) -> list:
    atom_positions = [np.asarray(pos) for pos in docking_outputs["atom_pos"]]
    ligand_positions = [np.asarray(pos) for pos in docking_outputs["ligand_pos"]]
    if len(atom_positions) != len(ligand_positions):
        raise ValueError(
            f"atom_pos count {len(atom_positions)} does not match ligand_pos count {len(ligand_positions)}."
        )
    return [
        {"atom_pos": atom_pos, "ligand_pos": ligand_pos}
        for atom_pos, ligand_pos in zip(atom_positions, ligand_positions)
    ]


def set_logger_job_name(cfg, suffix: str) -> None:
    base_name = cfg.logger.name or cfg.run_name
    if base_name and not str(base_name).endswith(f"-{suffix}"):
        cfg.logger.name = f"{base_name}-{suffix}"


def inference_metrics(
    name: str,
    graph,
    result,
) -> dict[str, float]:
    target_metrics = compute_inference_sample_metrics(graph, result.predictions)
    if not target_metrics:
        return {}
    sample_metrics = dict(target_metrics)
    rmsds = np.asarray(to_numpy(sample_metrics["rmsds"]), dtype=float).reshape(-1)
    confidences = np.asarray(to_numpy(result.confidences), dtype=float)
    if confidences.ndim > 1:
        confidences = confidences.reshape(confidences.shape[0], -1)[:, 0]
    else:
        confidences = confidences.reshape(-1)
    if len(confidences) != len(rmsds):
        raise ValueError(f"{name}: confidence count {len(confidences)} does not match RMSD count {len(rmsds)}.")
    confidences = np.where(np.isfinite(confidences), confidences, -np.inf)
    if np.all(np.isneginf(confidences)):
        raise ValueError(f"{name}: inference metrics require at least one finite confidence score.")
    top1_idx = int(np.argsort(confidences)[::-1][0])

    metrics: dict[str, float] = {}

    top1_ligand_rmsd = float(rmsds[top1_idx]) if np.isfinite(rmsds[top1_idx]) else float("nan")
    finite_ligand_rmsds = rmsds[np.isfinite(rmsds)]
    oracle_ligand_rmsd = float(finite_ligand_rmsds.min()) if finite_ligand_rmsds.size else None
    metrics["top1_conf_ligand_rmsd_1a"] = threshold(top1_ligand_rmsd, 1.0)
    metrics["top1_conf_ligand_rmsd_2a"] = threshold(top1_ligand_rmsd, 2.0)
    metrics["top1_conf_ligand_rmsd_5a"] = threshold(top1_ligand_rmsd, 5.0)
    metrics["top1_conf_mean_ligand_rmsd"] = top1_ligand_rmsd
    if oracle_ligand_rmsd is not None:
        metrics["oracle_ligand_rmsd_1a"] = threshold(oracle_ligand_rmsd, 1.0)
        metrics["oracle_ligand_rmsd_2a"] = threshold(oracle_ligand_rmsd, 2.0)
        metrics["oracle_ligand_rmsd_5a"] = threshold(oracle_ligand_rmsd, 5.0)
        metrics["oracle_mean_ligand_rmsd"] = oracle_ligand_rmsd

    pli_lddt = metric_vector(sample_metrics, "pli_lddt")
    if pli_lddt is not None and top1_idx < len(pli_lddt):
        metrics["pli_lddt"] = float(pli_lddt[top1_idx]) if np.isfinite(pli_lddt[top1_idx]) else float("nan")

    aa_rmsds = metric_vector(sample_metrics, "aa_rmsds")
    if aa_rmsds is not None:
        top1_aa_rmsd = (
            float(aa_rmsds[top1_idx])
            if top1_idx < len(aa_rmsds) and np.isfinite(aa_rmsds[top1_idx])
            else float("nan")
        )
        finite_aa_rmsds = aa_rmsds[np.isfinite(aa_rmsds)]
        oracle_aa_rmsd = float(finite_aa_rmsds.min()) if finite_aa_rmsds.size else None
        metrics["top1_conf_aa_rmsd_05a"] = threshold(top1_aa_rmsd, 0.5)
        metrics["top1_conf_aa_rmsd_1a"] = threshold(top1_aa_rmsd, 1.0)
        metrics["top1_conf_aa_rmsd_2a"] = threshold(top1_aa_rmsd, 2.0)
        metrics["top1_conf_mean_aa_rmsd"] = top1_aa_rmsd
        if oracle_aa_rmsd is not None:
            metrics["oracle_aa_rmsd_05a"] = threshold(oracle_aa_rmsd, 0.5)
            metrics["oracle_aa_rmsd_1a"] = threshold(oracle_aa_rmsd, 1.0)
            metrics["oracle_aa_rmsd_2a"] = threshold(oracle_aa_rmsd, 2.0)
            metrics["oracle_mean_aa_rmsd"] = oracle_aa_rmsd

    bb_rmsds = metric_vector(sample_metrics, "bb_rmsds")
    if bb_rmsds is not None:
        top1_bb_rmsd = (
            float(bb_rmsds[top1_idx])
            if top1_idx < len(bb_rmsds) and np.isfinite(bb_rmsds[top1_idx])
            else float("nan")
        )
        finite_bb_rmsds = bb_rmsds[np.isfinite(bb_rmsds)]
        oracle_bb_rmsd = float(finite_bb_rmsds.min()) if finite_bb_rmsds.size else None
        metrics["top1_conf_bb_rmsd_05a"] = threshold(top1_bb_rmsd, 0.5)
        metrics["top1_conf_bb_rmsd_1a"] = threshold(top1_bb_rmsd, 1.0)
        metrics["top1_conf_bb_rmsd_2a"] = threshold(top1_bb_rmsd, 2.0)
        if oracle_bb_rmsd is not None:
            metrics["oracle_bb_rmsd_05a"] = threshold(oracle_bb_rmsd, 0.5)
            metrics["oracle_bb_rmsd_1a"] = threshold(oracle_bb_rmsd, 1.0)
            metrics["oracle_bb_rmsd_2a"] = threshold(oracle_bb_rmsd, 2.0)

    return metrics


def summarize_inference_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in _INFERENCE_METRIC_KEYS:
        values = [row[key] for row in rows if key in row and np.isfinite(row[key])]
        if values:
            summary[key] = float(np.mean(values))
    return summary


def print_ligand_rmsd(name: str, metrics: dict[str, float]) -> None:
    top1_rmsd = metrics["top1_conf_mean_ligand_rmsd"]
    message = f"{name}: ligand_rmsd={top1_rmsd:.3f}"
    oracle_rmsd = metrics.get("oracle_mean_ligand_rmsd")
    if oracle_rmsd is not None:
        message += f" oracle_ligand_rmsd={oracle_rmsd:.3f}"
    print(message, flush=True)


_INFERENCE_METRIC_KEYS = [
    "top1_conf_ligand_rmsd_1a",
    "top1_conf_ligand_rmsd_2a",
    "top1_conf_ligand_rmsd_5a",
    "oracle_ligand_rmsd_1a",
    "oracle_ligand_rmsd_2a",
    "oracle_ligand_rmsd_5a",
    "top1_conf_aa_rmsd_05a",
    "top1_conf_aa_rmsd_1a",
    "top1_conf_aa_rmsd_2a",
    "oracle_aa_rmsd_05a",
    "oracle_aa_rmsd_1a",
    "oracle_aa_rmsd_2a",
    "top1_conf_bb_rmsd_05a",
    "top1_conf_bb_rmsd_1a",
    "top1_conf_bb_rmsd_2a",
    "oracle_bb_rmsd_05a",
    "oracle_bb_rmsd_1a",
    "oracle_bb_rmsd_2a",
    "pli_lddt",
    "top1_conf_mean_ligand_rmsd",
    "oracle_mean_ligand_rmsd",
    "top1_conf_mean_aa_rmsd",
    "oracle_mean_aa_rmsd",
]


_POSEBUSTERS_RMSD_LE_2_COLUMN = "posebusters_rmsd_≤_2å"
_POSEBUSTERS_POSEBENCH_COLUMNS = [
    _POSEBUSTERS_RMSD_LE_2_COLUMN,
    "posebusters_mol_pred_loaded",
    "posebusters_mol_true_loaded",
    "posebusters_mol_cond_loaded",
    "posebusters_sanitization",
    "posebusters_bond_lengths",
    "posebusters_bond_angles",
    "posebusters_internal_steric_clash",
    "posebusters_aromatic_ring_flatness",
    "posebusters_double_bond_flatness",
    "posebusters_internal_energy",
    "posebusters_minimum_distance_to_protein",
    "posebusters_minimum_distance_to_organic_cofactors",
    "posebusters_minimum_distance_to_inorganic_cofactors",
    "posebusters_volume_overlap_with_protein",
    "posebusters_volume_overlap_with_organic_cofactors",
    "posebusters_volume_overlap_with_inorganic_cofactors",
    "posebusters_molecular_formula",
    "posebusters_molecular_bonds",
    "posebusters_tetrahedral_chirality",
    "posebusters_double_bond_stereochemistry",
]


def threshold(value: float, threshold: float) -> float:
    return float("nan") if not np.isfinite(value) else 100.0 * float(value < threshold)


def metric_vector(row: dict[str, np.ndarray | float], key: str) -> np.ndarray | None:
    if key not in row or row[key] is None:
        return None
    values = np.asarray(to_numpy(row[key]), dtype=float).reshape(-1)
    return values if values.size else None


def to_numpy(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def metadata_value(graph, key: str):
    if key not in graph._global_store:
        return None

    value = graph._global_store[key]
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if torch.is_tensor(value):
        value = (
            value.detach().cpu().item()
            if value.numel() == 1
            else value.detach().cpu().numpy()
        )
    if value is None:
        return None
    return str(value)


def graph_name(graph) -> str:
    value = metadata_value(graph, "name")
    return value or "complex"


def is_successful_graph(graph) -> bool:
    if "success" not in graph._global_store:
        return True
    value = graph._global_store["success"]
    if torch.is_tensor(value):
        return bool(value.detach().cpu().view(-1)[0].item())
    return bool(value)
