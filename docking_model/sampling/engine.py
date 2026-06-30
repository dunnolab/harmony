from __future__ import annotations

import contextlib
import copy
import logging
from dataclasses import dataclass

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from torch_geometric.data import Batch, HeteroData
from torch_geometric.loader import DataLoader

from docking_model.data.conformers.modify import (
    modify_conformer_fast_batch,
    modify_conformer_torsion_angles,
)
from docking_model.data.feature.helpers import rotate_backbone_torch, to_atom_grid_torch
from docking_model.geometry.ops import axis_angle_to_matrix
from docking_model.sampling.schedules import get_schedules, set_time, t_to_sigma as schedule_t_to_sigma


@dataclass
class SamplingResult:
    predictions: list
    confidences: torch.Tensor | None = None
    details: dict | None = None


@dataclass(frozen=True)
class ModelSamplingArgs:
    sigma: object
    no_torsion: bool = False
    flexible_sidechains: bool = False
    flexible_backbone: bool = False
    sidechain_tor_bridge: bool = False
    use_bb_orientation_feats: bool = False
    all_atoms: bool = True


class FastSamplingBackend:
    """Fast docking sampler."""

    def __init__(
        self,
        sigma_cfg=None,
        protein_cfg=None,
        *,
        all_atoms: bool = True,
        prior=None,
        confidence_model=None,
        filtering_data_list: list | None = None,
        filtering_model_args=None,
    ):
        self.sigma_cfg = sigma_cfg
        self.protein_cfg = protein_cfg
        self.all_atoms = all_atoms
        self.prior = prior
        self.confidence_model = confidence_model
        self.filtering_data_list = filtering_data_list
        self.filtering_model_args = filtering_model_args

    def configure(self, sigma_cfg=None, protein_cfg=None) -> None:
        if sigma_cfg is not None:
            self.sigma_cfg = sigma_cfg
        if protein_cfg is not None:
            self.protein_cfg = protein_cfg

    def randomize(self, data_list: list, model, cfg, device: torch.device) -> list:
        data_list = normalize_data_list(data_list)
        data_list = expand_samples(data_list, int(getattr(cfg, "samples_per_complex", 1)))
        model_args = self.model_args(model)
        randomize_position_inf(
            data_list=data_list,
            no_torsion=model_args.no_torsion,
            no_random=bool(getattr(cfg, "no_random", False)),
            tr_sigma_max=float(model_args.sigma.tr_sigma_max),
            flexible_sidechains=model_args.flexible_sidechains,
            flexible_backbone=model_args.flexible_backbone,
            sidechain_tor_bridge=model_args.sidechain_tor_bridge,
            use_bb_orientation_feats=model_args.use_bb_orientation_feats,
            prior=self.prior,
            initial_noise_std_proportion=float(getattr(cfg, "initial_noise_std_proportion", 1.0)),
            all_atoms=model_args.all_atoms,
            reset_sidechain_ve_to_apo_before_randomize=bool(
                getattr(cfg, "reset_sidechain_ve_to_apo_before_randomize", False)
            ),
        )
        return data_list

    def sample(self, data_list: list, model, schedules: dict, sigma_fn, cfg, device: torch.device):
        model_args = self.model_args(model)
        confidence_model = self.confidence_model if self.confidence_model is not None else model
        return sampling(
            data_list=normalize_data_list(data_list),
            model=model,
            inference_steps=int(getattr(cfg, "inference_steps", len(schedules["tr"]))),
            schedules=schedules,
            sidechain_tor_bridge=model_args.sidechain_tor_bridge,
            device=device,
            t_to_sigma=sigma_fn,
            model_args=model_args,
            no_random=bool(getattr(cfg, "no_random", False)),
            ode=bool(getattr(cfg, "ode", False)),
            confidence_model=confidence_model,
            filtering_data_list=self.filtering_data_list,
            filtering_model_args=self.filtering_model_args,
            batch_size=int(getattr(cfg, "batch_size", 32)),
            no_final_step_noise=bool(getattr(cfg, "no_final_step_noise", False)),
            return_full_trajectory=bool(getattr(cfg, "return_full_trajectory", False)),
            use_bb_orientation_feats=model_args.use_bb_orientation_feats,
            diff_temp_sampling=getattr(cfg, "diff_temp_sampling", None),
            diff_temp_psi=getattr(cfg, "diff_temp_psi", None),
            diff_temp_sigma_data=getattr(cfg, "diff_temp_sigma_data", None),
            flow_temp_scale_0=getattr(cfg, "flow_temp_scale_0", None),
            flow_temp_scale_1=getattr(cfg, "flow_temp_scale_1", None),
            precision=getattr(cfg, "precision", None),
            run_confidence=bool(getattr(cfg, "run_confidence", True)),
        )

    def model_args(self, model) -> ModelSamplingArgs:
        sigma_cfg = self.sigma_cfg
        if sigma_cfg is None:
            sigma_cfg = getattr(model, "sigma", None) or getattr(getattr(model, "args", None), "sigma", None)
        if sigma_cfg is None:
            raise ValueError("FastSamplingBackend requires sigma_cfg.")

        protein_cfg = self.protein_cfg
        return ModelSamplingArgs(
            sigma=sigma_cfg,
            no_torsion=bool(getattr(model, "no_torsion", getattr(getattr(model, "args", None), "no_torsion", False))),
            flexible_sidechains=bool(
                getattr(
                    model,
                    "flexible_sidechains",
                    getattr(protein_cfg, "flexible_sidechains", getattr(getattr(model, "args", None), "flexible_sidechains", False)),
                )
            ),
            flexible_backbone=bool(
                getattr(
                    model,
                    "flexible_backbone",
                    getattr(protein_cfg, "flexible_backbone", getattr(getattr(model, "args", None), "flexible_backbone", False)),
                )
            ),
            sidechain_tor_bridge=bool(
                getattr(
                    model,
                    "sidechain_tor_bridge",
                    getattr(protein_cfg, "sidechain_tor_bridge", getattr(getattr(model, "args", None), "sidechain_tor_bridge", False)),
                )
            ),
            use_bb_orientation_feats=bool(
                getattr(
                    model,
                    "use_bb_orientation_feats",
                    getattr(protein_cfg, "use_bb_orientation_feats", getattr(getattr(model, "args", None), "use_bb_orientation_feats", False)),
                )
            ),
            all_atoms=bool(getattr(getattr(model, "args", None), "all_atoms", self.all_atoms)),
        )


class SamplingEngine:
    """Shared validation and standalone inference sampling path."""

    def __init__(self, cfg, sigma_cfg, time_cfg, protein_cfg, backend: FastSamplingBackend):
        self.cfg = cfg
        self.sigma_cfg = sigma_cfg
        self.time_cfg = time_cfg
        self.protein_cfg = protein_cfg
        self.backend = backend
        self.backend.configure(sigma_cfg=sigma_cfg, protein_cfg=protein_cfg)

    @torch.no_grad()
    def generate(
        self,
        data_list: list,
        model: torch.nn.Module,
        device: torch.device,
        overrides: dict | None = None,
    ) -> SamplingResult:
        model.eval()
        cfg = copy.copy(self.cfg)
        for key, value in (overrides or {}).items():
            setattr(cfg, key, value)
        schedules = get_schedules(
            inference_steps=cfg.inference_steps,
            sampling_alpha=getattr(cfg, "sampling_alpha", self.time_cfg.sampling_alpha),
            sampling_beta=getattr(cfg, "sampling_beta", self.time_cfg.sampling_beta),
            bb_tr_bridge_alpha=getattr(cfg, "bb_tr_bridge_alpha", self.time_cfg.bb_tr_bridge_alpha)
            if self.protein_cfg.flexible_backbone
            else None,
            bb_rot_bridge_alpha=getattr(cfg, "bb_rot_bridge_alpha", self.time_cfg.bb_rot_bridge_alpha)
            if self.protein_cfg.flexible_backbone
            else None,
            sc_tor_bridge_alpha=getattr(cfg, "sc_tor_bridge_alpha", self.time_cfg.sc_tor_bridge_alpha)
            if self.protein_cfg.flexible_sidechains
            else None,
            sidechain_tor_bridge=self.protein_cfg.sidechain_tor_bridge,
        )
        randomized = self.backend.randomize(
            data_list=normalize_data_list(data_list),
            model=model,
            cfg=cfg,
            device=device,
        )
        sampled = self.backend.sample(
            data_list=randomized,
            model=model,
            schedules=schedules,
            sigma_fn=lambda t: schedule_t_to_sigma(t, self.sigma_cfg),
            cfg=cfg,
            device=device,
        )
        rank_by_confidence = bool(getattr(cfg, "rank_by_confidence", True))
        if isinstance(sampled, SamplingResult):
            result = sampled
        elif len(sampled) == 4:
            predictions, confidences, ligand_trajectory, atom_trajectory = sampled
            result = SamplingResult(
                predictions=predictions,
                confidences=confidences,
                details={"ligand_trajectory": ligand_trajectory, "atom_trajectory": atom_trajectory},
            )
        else:
            predictions, confidences = sampled
            result = SamplingResult(predictions=predictions, confidences=confidences)

        if rank_by_confidence:
            return rank_sampling_result_by_confidence(result)
        if not result.predictions:
            raise ValueError("Sampling produced no predictions.")
        return result


def rank_sampling_result_by_confidence(result: SamplingResult) -> SamplingResult:
    predictions = list(result.predictions or [])
    if not predictions:
        raise ValueError("Sampling produced no predictions.")

    scores = confidence_scores(result.confidences, expected_len=len(predictions))
    rank_values = torch.where(
        torch.isfinite(scores),
        scores,
        torch.full_like(scores, -torch.inf),
    )
    if not bool(torch.isfinite(rank_values).any()):
        raise ValueError("Inference confidence scores are all non-finite.")

    order = torch.argsort(rank_values, descending=True).detach().cpu().tolist()
    return SamplingResult(
        predictions=[predictions[idx] for idx in order],
        confidences=index_first_dim(result.confidences, order),
        details=rank_details(result.details, order),
    )


def confidence_scores(confidences, expected_len: int) -> torch.Tensor:
    if confidences is None:
        raise ValueError("Inference requires confidence scores, but the sampler returned None.")

    if torch.is_tensor(confidences):
        values = confidences.detach()
    else:
        values = torch.as_tensor(np.asarray(confidences, dtype=float))

    if values.ndim == 0:
        values = values.reshape(1)
    elif values.ndim > 1:
        values = values.reshape(values.shape[0], -1)[:, 0]
    else:
        values = values.reshape(-1)

    if values.shape[0] != expected_len:
        raise ValueError(
            f"Confidence count {values.shape[0]} does not match prediction count {expected_len}."
        )
    return values.float()


def index_first_dim(value, order: list[int]):
    if value is None:
        return None
    if torch.is_tensor(value):
        index = torch.as_tensor(order, dtype=torch.long, device=value.device)
        return value.index_select(0, index)
    if isinstance(value, np.ndarray):
        return value[order]
    if isinstance(value, list):
        return [value[idx] for idx in order]
    if isinstance(value, tuple):
        return tuple(value[idx] for idx in order)
    array = np.asarray(value)
    if array.ndim > 0 and array.shape[0] == len(order):
        return array[order]
    return value


def rank_details(details: dict | None, order: list[int]) -> dict | None:
    if details is None:
        return None
    return {key: rank_detail_value(value, order) for key, value in details.items()}


def rank_detail_value(value, order: list[int]):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.ndim > 0 and value.shape[0] == len(order):
            index = torch.as_tensor(order, dtype=torch.long, device=value.device)
            return value.index_select(0, index)
        return value
    if isinstance(value, np.ndarray):
        return value[order] if value.ndim > 0 and value.shape[0] == len(order) else value
    if isinstance(value, list) and len(value) == len(order):
        return [value[idx] for idx in order]
    if isinstance(value, tuple) and len(value) == len(order):
        return tuple(value[idx] for idx in order)
    return value


def wrap_torus(theta: torch.Tensor) -> torch.Tensor:
    return (theta + torch.pi) % (2 * torch.pi) - torch.pi


def normalize_data_list(data_list: list) -> list:
    normalized = []
    for item in data_list:
        if isinstance(item, Batch):
            normalized.extend(Batch.to_data_list(item))
        else:
            normalized.append(item)
    return normalized


def expand_samples(data_list: list, samples_per_complex: int) -> list:
    if samples_per_complex <= 1:
        return data_list
    expanded = []
    for data in data_list:
        expanded.extend(copy.deepcopy(data) for _ in range(samples_per_complex))
    return expanded


def get_sidechain_tor_sigma_bounds(model_args):
    sigma_cfg = getattr(model_args, "sigma", None)
    sigma_min = getattr(model_args, "sidechain_tor_sigma_min", None)
    sigma_max = getattr(model_args, "sidechain_tor_sigma_max", None)
    if sigma_cfg is not None:
        sigma_min = getattr(sigma_cfg, "sidechain_tor_sigma_min", sigma_min)
        sigma_max = getattr(sigma_cfg, "sidechain_tor_sigma_max", sigma_max)
    if sigma_min is None or sigma_max is None:
        raise ValueError("VE sidechain torsion sampling requires sidechain_tor_sigma_min and sidechain_tor_sigma_max.")
    if sigma_min <= 0 or sigma_max <= 0 or sigma_max < sigma_min:
        raise ValueError(f"Invalid VE sidechain sigma bounds: sigma_min={sigma_min}, sigma_max={sigma_max}.")
    return float(sigma_min), float(sigma_max)


def append_trajectory_snapshot(ligand_trajectory, atom_trajectory, sample_offset: int, complex_graph_batch) -> None:
    batch_cpu = copy.deepcopy(complex_graph_batch).to("cpu")
    for local_idx, graph in enumerate(Batch.to_data_list(batch_cpu)):
        sample_idx = sample_offset + local_idx
        ligand_trajectory[sample_idx].append(graph["ligand"].pos.detach().cpu().numpy())
        atom_trajectory[sample_idx].append(graph["atom"].pos.detach().cpu().numpy())


def randomize_position_inf(
    data_list: list[HeteroData],
    no_torsion: bool,
    no_random: bool,
    tr_sigma_max: float,
    flexible_sidechains: bool = False,
    flexible_backbone: bool = False,
    sidechain_tor_bridge: bool = False,
    use_bb_orientation_feats: bool = False,
    prior=None,
    initial_noise_std_proportion: float = 1.0,
    all_atoms: bool = True,
    reset_sidechain_ve_to_apo_before_randomize: bool = False,
):
    if not no_torsion:
        for complex_graph in data_list:
            num_torsions = int(complex_graph["ligand"].edge_mask.sum().item())
            torsion_updates = np.random.uniform(low=-np.pi, high=np.pi, size=num_torsions)
            edge_index = complex_graph["ligand", "lig_bond", "ligand"].edge_index
            torsion_updates_tensor = torch.tensor(torsion_updates, device=edge_index.device).float()
            complex_graph["ligand"].pos = modify_conformer_torsion_angles(
                pos=complex_graph["ligand"].pos,
                edge_index=edge_index,
                mask_rotate=complex_graph["ligand"].edge_mask,
                torsion_updates=torsion_updates_tensor,
                fragment_index=complex_graph["ligand"].lig_fragment_index,
            )
            complex_graph["ligand"].tor_theta = wrap_torus(
                torsion_updates_tensor.to(device=complex_graph["ligand"].pos.device, dtype=complex_graph["ligand"].pos.dtype)
            )
    else:
        for complex_graph in data_list:
            complex_graph["ligand"].tor_theta = torch.empty(
                0,
                device=complex_graph["ligand"].pos.device,
                dtype=complex_graph["ligand"].pos.dtype,
            )

    if flexible_backbone and all_atoms:
        for complex_graph in data_list:
            complex_graph["atom"].pos = complex_graph["atom"].orig_aligned_apo_pos.float()
            complex_graph["receptor"].pos = complex_graph["atom"].pos[complex_graph["atom"].ca_mask]

            if prior is not None:
                atom_grid, x, y = to_atom_grid_torch(
                    complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors
                )
                calpha_delta_random = prior.sample_for_inference(complex_graph)
                complex_graph["receptor"].pos = complex_graph["receptor"].pos + calpha_delta_random
                atom_grid = atom_grid + calpha_delta_random.unsqueeze(1)
                complex_graph["atom"].pos = atom_grid[x, y]

            if flexible_sidechains:
                edge_mask = complex_graph["atom", "atom_bond", "atom"].edge_mask
                num_rotatable = int(edge_mask.sum().item())
                if not sidechain_tor_bridge and num_rotatable > 0:
                    sidechain_torsion_updates = torch.tensor(
                        np.random.uniform(low=-np.pi, high=np.pi, size=num_rotatable),
                        dtype=complex_graph["atom"].pos.dtype,
                        device=complex_graph["atom"].pos.device,
                    )
                    complex_graph["atom"].pos = modify_conformer_torsion_angles(
                        pos=complex_graph["atom"].pos,
                        edge_index=complex_graph["atom", "atom_bond", "atom"].edge_index,
                        mask_rotate=edge_mask,
                        fragment_index=complex_graph["atom_bond", "atom"].atom_fragment_index,
                        torsion_updates=sidechain_torsion_updates,
                        sidechains=True,
                    )
                    complex_graph.sidechain_tor_theta = wrap_torus(sidechain_torsion_updates)
                else:
                    complex_graph.sidechain_tor_theta = torch.zeros(
                        num_rotatable,
                        dtype=complex_graph["atom"].pos.dtype,
                        device=complex_graph["atom"].pos.device,
                    )

            if use_bb_orientation_feats:
                atom_grid, _, _ = to_atom_grid_torch(complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors)
                complex_graph["receptor"].bb_orientation = torch.cat(
                    [atom_grid[:, 0] - atom_grid[:, 1], atom_grid[:, 2] - atom_grid[:, 1]],
                    dim=1,
                )
    elif all_atoms:
        if flexible_sidechains:
            for complex_graph in data_list:
                complex_graph.sidechain_tor_theta = torch.empty(
                    0,
                    dtype=complex_graph["atom"].pos.dtype,
                    device=complex_graph["atom"].pos.device,
                )

        if flexible_sidechains and not sidechain_tor_bridge:
            for complex_graph in data_list:
                edge_mask = complex_graph["atom", "atom_bond", "atom"].edge_mask
                num_rotatable = int(edge_mask.sum().item())
                if num_rotatable == 0:
                    continue
                if reset_sidechain_ve_to_apo_before_randomize:
                    complex_graph["atom"].pos = complex_graph["atom"].orig_apo_pos.float()
                    complex_graph["receptor"].pos = complex_graph["atom"].pos[complex_graph["atom"].ca_mask]

                sidechain_torsion_updates = torch.tensor(
                    np.random.uniform(low=-np.pi, high=np.pi, size=num_rotatable),
                    dtype=complex_graph["atom"].pos.dtype,
                    device=complex_graph["atom"].pos.device,
                )
                complex_graph["atom"].pos = modify_conformer_torsion_angles(
                    pos=complex_graph["atom"].pos,
                    edge_index=complex_graph["atom", "atom_bond", "atom"].edge_index,
                    mask_rotate=edge_mask,
                    fragment_index=complex_graph["atom_bond", "atom"].atom_fragment_index,
                    torsion_updates=sidechain_torsion_updates,
                    sidechains=True,
                )
                complex_graph.sidechain_tor_theta = wrap_torus(sidechain_torsion_updates)

        elif flexible_sidechains and sidechain_tor_bridge:
            for complex_graph in data_list:
                edge_mask = complex_graph["atom", "atom_bond", "atom"].edge_mask
                if edge_mask.sum() == 0:
                    continue
                atom_edge_store = complex_graph["atom", "atom_bond", "atom"]
                if hasattr(atom_edge_store, "sc_conformer_match_rotations"):
                    sc_tor_delta_holo_all = atom_edge_store.sc_conformer_match_rotations
                else:
                    sc_tor_delta_holo_all = torch.zeros(
                        edge_mask.shape[0],
                        dtype=complex_graph["atom"].pos.dtype,
                        device=complex_graph["atom"].pos.device,
                    )
                sidechain_torsion_updates = -sc_tor_delta_holo_all[edge_mask]
                complex_graph["atom"].pos = modify_conformer_torsion_angles(
                    pos=complex_graph["atom"].pos,
                    edge_index=complex_graph["atom", "atom_bond", "atom"].edge_index,
                    mask_rotate=edge_mask,
                    fragment_index=complex_graph["atom_bond", "atom"].atom_fragment_index,
                    torsion_updates=sidechain_torsion_updates,
                    sidechains=True,
                )
                complex_graph.sidechain_tor_theta = torch.zeros_like(sidechain_torsion_updates)

    for complex_graph in data_list:
        if use_bb_orientation_feats and "bb_orientation" not in complex_graph["receptor"]:
            if not all_atoms:
                raise NotImplementedError("Blind mode is not supported.")
            atom_grid, _, _ = to_atom_grid_torch(complex_graph["atom"].pos, complex_graph["receptor"].lens_receptors)
            complex_graph["receptor"].bb_orientation = torch.cat(
                [atom_grid[:, 0] - atom_grid[:, 1], atom_grid[:, 2] - atom_grid[:, 1]],
                dim=1,
            )

        molecule_center = torch.mean(complex_graph["ligand"].pos, dim=0, keepdim=True)
        random_rotation = torch.from_numpy(R.random().as_matrix()).float().to(complex_graph["ligand"].pos.device)
        complex_graph["ligand"].pos = (complex_graph["ligand"].pos - molecule_center) @ random_rotation.T

        if not no_random:
            tr_update = torch.normal(
                mean=0,
                std=tr_sigma_max * initial_noise_std_proportion,
                size=(1, 3),
            ).to(complex_graph["ligand"].pos.device)
            complex_graph["ligand"].pos += tr_update

        if all_atoms:
            assert torch.isnan(complex_graph["atom"].pos).sum().item() == 0, "NaNs are encountered in atom positions at the end of randomize_position_inf"
        assert torch.isnan(complex_graph["ligand"].pos).sum().item() == 0, "NaNs are encountered in ligand positions at the end of randomize_position_inf"
        assert torch.isnan(complex_graph["receptor"].pos).sum().item() == 0, "NaNs are encountered in receptor positions at the end of randomize_position_inf"


def sampling(
    data_list: list[HeteroData],
    model: torch.nn.Module,
    inference_steps: int,
    schedules,
    sidechain_tor_bridge,
    device: str | torch.device,
    t_to_sigma: callable,
    model_args,
    no_random: bool = False,
    ode: bool = False,
    visualization_list=None,
    sidechain_visualization_list=None,
    confidence_model=None,
    filtering_data_list=None,
    filtering_model_args=None,
    batch_size: int = 32,
    no_final_step_noise: bool = False,
    return_full_trajectory: bool = False,
    debug_backbone: bool = False,
    debug_sidechain: bool = False,
    use_bb_orientation_feats: bool = False,
    diff_temp_sampling: tuple | None = None,
    diff_temp_psi: tuple | None = None,
    diff_temp_sigma_data: tuple | None = None,
    flow_temp_scale_0: tuple | None = None,
    flow_temp_scale_1: tuple | None = None,
    precision: str | None = None,
    run_confidence: bool = True,
):
    del visualization_list, sidechain_visualization_list, debug_backbone, debug_sidechain
    num_samples = len(data_list)
    ligand_trajectory = [[] for _ in range(num_samples)] if return_full_trajectory else None
    atom_trajectory = [[] for _ in range(num_samples)] if return_full_trajectory else None

    tr_schedule = schedules["tr"]
    rot_schedule = schedules["rot"]
    tor_schedule = schedules["tor"]
    bb_tr_schedule = schedules["bb_tr"]
    bb_rot_schedule = schedules["bb_rot"]
    sc_tor_schedule = schedules["sc_tor"]
    t_schedule = schedules["t"]

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    all_atoms = bool(getattr(model_args, "all_atoms", False))
    final_data_list = []
    sample_offset = 0

    for complex_graph_batch in loader:
        complex_graph_batch = complex_graph_batch.to(device)
        batch_graphs = complex_graph_batch.num_graphs

        if return_full_trajectory:
            append_trajectory_snapshot(ligand_trajectory, atom_trajectory, sample_offset, complex_graph_batch)

        for t_idx in range(inference_steps):
            inputs = complex_graph_batch.clone()
            t_tr = tr_schedule[t_idx]
            t_rot = rot_schedule[t_idx]
            t_tor = tor_schedule[t_idx]
            t_sidechain_tor = sc_tor_schedule[t_idx]
            t_bb_tr = bb_tr_schedule[t_idx] if bb_tr_schedule is not None else None
            t_bb_rot = bb_rot_schedule[t_idx] if bb_rot_schedule is not None else None
            sigma_dict = t_to_sigma(
                {
                    "tr": t_tr,
                    "rot": t_rot,
                    "tor": t_tor,
                    "sc_tor": t_sidechain_tor,
                    "bb_tr": t_bb_tr,
                    "bb_rot": t_bb_rot,
                }
            )
            tr_sigma = sigma_dict["tr_sigma"]
            rot_sigma = sigma_dict["rot_sigma"]
            tor_sigma = sigma_dict["tor_sigma"]
            sidechain_tor_sigma = sigma_dict["sc_tor_sigma"]
            bb_tr_sigma = sigma_dict["bb_tr_sigma"]
            bb_rot_sigma = sigma_dict["bb_rot_sigma"]

            set_time(
                inputs,
                t_schedule[t_idx] if t_schedule is not None else None,
                t_tr,
                t_rot,
                t_tor,
                t_sidechain_tor,
                t_bb_tr,
                t_bb_rot,
                batch_graphs,
                all_atoms=all_atoms,
                device=device,
            )
            if not run_confidence:
                inputs.skip_confidence_head = True

            dt_tr = tr_schedule[t_idx] - tr_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tr_schedule[t_idx]
            dt_rot = rot_schedule[t_idx] - rot_schedule[t_idx + 1] if t_idx < inference_steps - 1 else rot_schedule[t_idx]
            dt_tor = tor_schedule[t_idx] - tor_schedule[t_idx + 1] if t_idx < inference_steps - 1 else tor_schedule[t_idx]

            with torch.no_grad(), autocast_context(device, precision):
                outputs = model(inputs)
            tr_score = outputs["tr_pred"]
            rot_score = outputs["rot_pred"]
            tor_score = outputs["tor_pred"]
            bb_tr_drift = outputs["bb_tr_pred"]
            bb_rot_drift = outputs["bb_rot_pred"]
            sidechain_tor_score = outputs["sc_tor_pred"]

            tr_g = tr_sigma * torch.sqrt(
                torch.tensor(
                    2 * np.log(model_args.sigma.tr_sigma_max / model_args.sigma.tr_sigma_min),
                    device=tr_score.device,
                    dtype=tr_score.dtype,
                )
            )
            rot_g = rot_sigma * torch.sqrt(
                torch.tensor(
                    2 * np.log(model_args.sigma.rot_sigma_max / model_args.sigma.rot_sigma_min),
                    device=rot_score.device,
                    dtype=rot_score.dtype,
                )
            )

            tor_perturb = None
            tor_z = None
            tor_g = None
            if ode:
                tr_perturb = 0.5 * tr_g**2 * dt_tr * tr_score
                rot_perturb = 0.5 * rot_score * dt_rot * rot_g**2
            else:
                tr_z, rot_z = diffusion_noise(
                    tr_score,
                    rot_score,
                    batch_graphs,
                    no_random or (no_final_step_noise and t_idx == inference_steps - 1),
                )
                tr_perturb = tr_g**2 * dt_tr * tr_score + tr_g * np.sqrt(dt_tr) * tr_z
                rot_perturb = rot_score * dt_rot * rot_g**2 + rot_g * np.sqrt(dt_rot) * rot_z

            if not model_args.no_torsion:
                tor_g = tor_sigma * torch.sqrt(
                    torch.tensor(
                        2 * np.log(model_args.sigma.tor_sigma_max / model_args.sigma.tor_sigma_min),
                        device=tor_score.device,
                        dtype=tor_score.dtype,
                    )
                )
                if ode:
                    tor_perturb = 0.5 * tor_g**2 * dt_tor * tor_score
                else:
                    if no_random or (no_final_step_noise and t_idx == inference_steps - 1):
                        tor_z = tor_score.new_zeros(tor_score.shape)
                    else:
                        tor_z = torch.normal(mean=0, std=1, size=tor_score.shape, device=tor_score.device)
                    tor_perturb = tor_g**2 * dt_tor * tor_score + tor_g * np.sqrt(dt_tor) * tor_z

            if (not ode) and diff_temp_sampling is not None:
                tr_perturb, rot_perturb, tor_perturb = apply_diffusion_temperatures(
                    tr_perturb,
                    rot_perturb,
                    tor_perturb,
                    tr_score,
                    rot_score,
                    tor_score,
                    tr_z,
                    rot_z,
                    tor_z,
                    tr_g,
                    rot_g,
                    tor_g,
                    tr_sigma,
                    rot_sigma,
                    tor_sigma,
                    dt_tr,
                    dt_rot,
                    dt_tor,
                    model_args,
                    diff_temp_sampling,
                    diff_temp_psi,
                    diff_temp_sigma_data,
                )

            if flow_temp_scale_0 is not None:
                bb_tr_drift, bb_rot_drift, sidechain_tor_score = apply_flow_temperatures(
                    bb_tr_drift,
                    bb_rot_drift,
                    sidechain_tor_score,
                    bb_tr_schedule,
                    bb_rot_schedule,
                    sc_tor_schedule,
                    t_idx,
                    flow_temp_scale_0,
                    flow_temp_scale_1,
                )

            bb_tr_perturb = None
            bb_rot_perturb = None
            sidechain_tor_update = None
            if model_args.flexible_sidechains:
                sidechain_tor_update = sidechain_tor_perturb(
                    sidechain_tor_score,
                    sidechain_tor_sigma,
                    sc_tor_schedule,
                    t_idx,
                    inference_steps,
                    sidechain_tor_bridge,
                    model_args,
                    no_random,
                    no_final_step_noise,
                    ode,
                )

            if model_args.flexible_backbone:
                dt_bb_tr = bb_tr_schedule[t_idx + 1] - bb_tr_schedule[t_idx] if t_idx < inference_steps - 1 else 1 - bb_tr_schedule[t_idx]
                dt_bb_rot = bb_rot_schedule[t_idx + 1] - bb_rot_schedule[t_idx] if t_idx < inference_steps - 1 else 1 - bb_rot_schedule[t_idx]
                if ode:
                    bb_tr_perturb = bb_tr_drift * dt_bb_tr
                    bb_rot_perturb = bb_rot_drift * dt_bb_rot
                else:
                    zero_noise = no_random or (no_final_step_noise and t_idx == inference_steps - 1)
                    bb_tr_z = bb_tr_drift.new_zeros(bb_tr_drift.shape) if zero_noise else torch.normal(mean=0, std=1, size=bb_tr_drift.shape, device=bb_tr_drift.device)
                    bb_rot_z = bb_rot_drift.new_zeros(bb_rot_drift.shape) if zero_noise else torch.normal(mean=0, std=1, size=bb_rot_drift.shape, device=bb_rot_drift.device)
                    bb_tr_perturb = bb_tr_drift * dt_bb_tr + bb_tr_z * np.sqrt(dt_bb_tr) * bb_tr_sigma
                    bb_rot_perturb = bb_rot_drift * dt_bb_rot + bb_rot_z * np.sqrt(dt_bb_rot) * bb_rot_sigma

            if model_args.flexible_sidechains and sidechain_tor_update is not None:
                if (
                    not hasattr(complex_graph_batch, "sidechain_tor_theta")
                    or complex_graph_batch.sidechain_tor_theta.numel() != sidechain_tor_update.numel()
                ):
                    complex_graph_batch.sidechain_tor_theta = sidechain_tor_update.new_zeros(sidechain_tor_update.shape)
                complex_graph_batch.sidechain_tor_theta = wrap_torus(
                    complex_graph_batch.sidechain_tor_theta.to(sidechain_tor_update.device) + sidechain_tor_update
                )
                complex_graph_batch["atom"].pos = modify_conformer_torsion_angles(
                    pos=complex_graph_batch["atom"].pos,
                    edge_index=complex_graph_batch["atom", "atom_bond", "atom"].edge_index,
                    mask_rotate=complex_graph_batch["atom", "atom_bond", "atom"].edge_mask,
                    fragment_index=complex_graph_batch["atom_bond", "atom"].atom_fragment_index,
                    torsion_updates=sidechain_tor_update,
                    sidechains=True,
                )

            if model_args.flexible_backbone:
                new_pos, _ = rotate_backbone_torch(
                    atoms=complex_graph_batch["atom"].pos,
                    t_vec=bb_tr_perturb,
                    rot_mat=axis_angle_to_matrix(bb_rot_perturb),
                    lens_receptors=complex_graph_batch["receptor"].lens_receptors,
                    total_rot=None,
                    detach=False,
                )
                complex_graph_batch["atom"].pos = new_pos
                complex_graph_batch["receptor"].pos = complex_graph_batch["atom"].pos[complex_graph_batch["atom"].ca_mask]
                if use_bb_orientation_feats:
                    atom_grid, _, _ = to_atom_grid_torch(
                        complex_graph_batch["atom"].pos,
                        complex_graph_batch["receptor"].lens_receptors,
                    )
                    complex_graph_batch["receptor"].bb_orientation = torch.cat(
                        [atom_grid[:, 0] - atom_grid[:, 1], atom_grid[:, 2] - atom_grid[:, 1]],
                        dim=1,
                    )

            if (not model_args.no_torsion) and tor_perturb is not None:
                if (
                    "tor_theta" not in complex_graph_batch["ligand"]
                    or complex_graph_batch["ligand"].tor_theta.numel() != tor_perturb.numel()
                ):
                    complex_graph_batch["ligand"].tor_theta = tor_perturb.new_zeros(tor_perturb.shape)
                complex_graph_batch["ligand"].tor_theta = wrap_torus(
                    complex_graph_batch["ligand"].tor_theta.to(tor_perturb.device) + tor_perturb
                )

            complex_graph_batch = modify_conformer_fast_batch(
                data=complex_graph_batch,
                tr_update=tr_perturb,
                rot_update=rot_perturb,
                torsion_updates=tor_perturb,
            )

            if return_full_trajectory:
                append_trajectory_snapshot(ligand_trajectory, atom_trajectory, sample_offset, complex_graph_batch)

        final_data_list.extend(Batch.to_data_list(complex_graph_batch))
        sample_offset += batch_graphs

    if return_full_trajectory:
        for sample_idx, graph in enumerate(final_data_list):
            ligand_trajectory[sample_idx].append(graph["ligand"].pos.detach().cpu().numpy())
            atom_trajectory[sample_idx].append(graph["atom"].pos.detach().cpu().numpy())

    confidence = None
    if run_confidence:
        confidence = run_confidence_model(
            final_data_list=final_data_list,
            confidence_model=confidence_model,
            filtering_data_list=filtering_data_list,
            filtering_model_args=filtering_model_args,
            model_args=model_args,
            sidechain_tor_bridge=sidechain_tor_bridge,
            batch_size=batch_size,
            all_atoms=all_atoms,
            device=device,
        )
    if return_full_trajectory:
        return final_data_list, confidence, np.asarray(ligand_trajectory), np.asarray(atom_trajectory)
    return final_data_list, confidence


def diffusion_noise(tr_score, rot_score, batch_graphs: int, zero_noise: bool):
    if zero_noise:
        return tr_score.new_zeros((batch_graphs, 3)), rot_score.new_zeros((batch_graphs, 3))
    return (
        torch.normal(mean=0, std=1, size=(batch_graphs, 3), device=tr_score.device),
        torch.normal(mean=0, std=1, size=(batch_graphs, 3), device=rot_score.device),
    )


def apply_diffusion_temperatures(
    tr_perturb,
    rot_perturb,
    tor_perturb,
    tr_score,
    rot_score,
    tor_score,
    tr_z,
    rot_z,
    tor_z,
    tr_g,
    rot_g,
    tor_g,
    tr_sigma,
    rot_sigma,
    tor_sigma,
    dt_tr,
    dt_rot,
    dt_tor,
    model_args,
    diff_temp_sampling,
    diff_temp_psi,
    diff_temp_sigma_data,
):
    if len(diff_temp_sampling) != 3 or len(diff_temp_psi) != 3 or len(diff_temp_sigma_data) != 3:
        raise ValueError("diff_temp_sampling, diff_temp_psi, and diff_temp_sigma_data must all have length 3.")
    if diff_temp_sampling[0] != 1.0:
        tr_sigma_data = np.exp(
            diff_temp_sigma_data[0] * np.log(model_args.sigma.tr_sigma_max)
            + (1 - diff_temp_sigma_data[0]) * np.log(model_args.sigma.tr_sigma_min)
        )
        lambda_tr = (tr_sigma_data + tr_sigma) / (tr_sigma_data + tr_sigma / diff_temp_sampling[0])
        tr_perturb = (
            tr_g**2 * dt_tr * (lambda_tr + diff_temp_sampling[0] * diff_temp_psi[0] / 2) * tr_score
            + tr_g * np.sqrt(dt_tr * (1 + diff_temp_psi[0])) * tr_z
        )
    if diff_temp_sampling[1] != 1.0:
        rot_sigma_data = np.exp(
            diff_temp_sigma_data[1] * np.log(model_args.sigma.rot_sigma_max)
            + (1 - diff_temp_sigma_data[1]) * np.log(model_args.sigma.rot_sigma_min)
        )
        lambda_rot = (rot_sigma_data + rot_sigma) / (rot_sigma_data + rot_sigma / diff_temp_sampling[1])
        rot_perturb = (
            rot_g**2 * dt_rot * (lambda_rot + diff_temp_sampling[1] * diff_temp_psi[1] / 2) * rot_score
            + rot_g * np.sqrt(dt_rot * (1 + diff_temp_psi[1])) * rot_z
        )
    if tor_perturb is not None and diff_temp_sampling[2] != 1.0:
        tor_sigma_data = np.exp(
            diff_temp_sigma_data[2] * np.log(model_args.sigma.tor_sigma_max)
            + (1 - diff_temp_sigma_data[2]) * np.log(model_args.sigma.tor_sigma_min)
        )
        lambda_tor = (tor_sigma_data + tor_sigma) / (tor_sigma_data + tor_sigma / diff_temp_sampling[2])
        tor_perturb = (
            tor_g**2 * dt_tor * (lambda_tor + diff_temp_sampling[2] * diff_temp_psi[2] / 2) * tor_score
            + tor_g * np.sqrt(dt_tor * (1 + diff_temp_psi[2])) * tor_z
        )
    return tr_perturb, rot_perturb, tor_perturb


def apply_flow_temperatures(
    bb_tr_drift,
    bb_rot_drift,
    sidechain_tor_score,
    bb_tr_schedule,
    bb_rot_schedule,
    sc_tor_schedule,
    t_idx: int,
    flow_temp_scale_0,
    flow_temp_scale_1,
):
    if len(flow_temp_scale_0) != 3 or len(flow_temp_scale_1) != 3:
        raise ValueError("flow_temp_scale_0 and flow_temp_scale_1 must both have length 3.")
    if bb_tr_schedule is not None and torch.is_tensor(bb_tr_drift) and bb_tr_drift.numel() > 0:
        bb_tr_drift = bb_tr_drift * (bb_tr_schedule[t_idx] * flow_temp_scale_0[0] + (1 - bb_tr_schedule[t_idx]) * flow_temp_scale_1[0])
    if bb_rot_schedule is not None and torch.is_tensor(bb_rot_drift) and bb_rot_drift.numel() > 0:
        bb_rot_drift = bb_rot_drift * (bb_rot_schedule[t_idx] * flow_temp_scale_0[1] + (1 - bb_rot_schedule[t_idx]) * flow_temp_scale_1[1])
    if sc_tor_schedule is not None and torch.is_tensor(sidechain_tor_score) and sidechain_tor_score.numel() > 0:
        sidechain_tor_score = sidechain_tor_score * (sc_tor_schedule[t_idx] * flow_temp_scale_0[2] + (1 - sc_tor_schedule[t_idx]) * flow_temp_scale_1[2])
    return bb_tr_drift, bb_rot_drift, sidechain_tor_score


def sidechain_tor_perturb(
    sidechain_tor_score,
    sidechain_tor_sigma,
    sc_tor_schedule,
    t_idx: int,
    inference_steps: int,
    sidechain_tor_bridge: bool,
    model_args,
    no_random: bool,
    no_final_step_noise: bool,
    ode: bool,
):
    if sidechain_tor_score.numel() == 0:
        return sidechain_tor_score
    if sidechain_tor_bridge:
        dt_sidechain_tor = (
            sc_tor_schedule[t_idx + 1] - sc_tor_schedule[t_idx]
            if t_idx < inference_steps - 1
            else 1 - sc_tor_schedule[t_idx]
        )
        if ode:
            return dt_sidechain_tor * sidechain_tor_score
        zero_noise = no_random or (no_final_step_noise and t_idx == inference_steps - 1)
        sidechain_tor_z = sidechain_tor_score.new_zeros(sidechain_tor_score.shape) if zero_noise else torch.normal(
            mean=0,
            std=1,
            size=sidechain_tor_score.shape,
            device=sidechain_tor_score.device,
        )
        return dt_sidechain_tor * sidechain_tor_score + np.sqrt(dt_sidechain_tor) * sidechain_tor_sigma * sidechain_tor_z

    dt_sidechain_tor = (
        sc_tor_schedule[t_idx] - sc_tor_schedule[t_idx + 1]
        if t_idx < inference_steps - 1
        else sc_tor_schedule[t_idx]
    )
    sc_tor_sigma_min, sc_tor_sigma_max = get_sidechain_tor_sigma_bounds(model_args)
    sidechain_tor_g = sidechain_tor_sigma * torch.sqrt(
        torch.tensor(
            2 * np.log(sc_tor_sigma_max / sc_tor_sigma_min),
            device=sidechain_tor_score.device,
            dtype=sidechain_tor_score.dtype,
        )
    )
    if ode:
        return 0.5 * sidechain_tor_g**2 * dt_sidechain_tor * sidechain_tor_score
    zero_noise = no_random or (no_final_step_noise and t_idx == inference_steps - 1)
    sidechain_tor_z = sidechain_tor_score.new_zeros(sidechain_tor_score.shape) if zero_noise else torch.normal(
        mean=0,
        std=1,
        size=sidechain_tor_score.shape,
        device=sidechain_tor_score.device,
    )
    return sidechain_tor_g**2 * dt_sidechain_tor * sidechain_tor_score + sidechain_tor_g * np.sqrt(dt_sidechain_tor) * sidechain_tor_z


def run_confidence_model(
    final_data_list,
    confidence_model,
    filtering_data_list,
    filtering_model_args,
    model_args,
    sidechain_tor_bridge,
    batch_size,
    all_atoms,
    device,
):
    if confidence_model is None:
        raise ValueError("Inference requires a confidence model.")

    confidence = []
    loader = DataLoader(final_data_list, batch_size=batch_size)
    filtering_loader = iter(DataLoader(filtering_data_list, batch_size=batch_size)) if filtering_data_list is not None else None
    confidence_args = filtering_model_args if filtering_model_args is not None else model_args
    confidence_all_atoms = bool(getattr(confidence_args, "all_atoms", all_atoms))

    for complex_graph_batch in loader:
        complex_graph_batch = complex_graph_batch.to(device)
        if filtering_loader is not None:
            inference_batch = next(filtering_loader).to(device)
            inference_batch["ligand"].pos = complex_graph_batch["ligand"].pos
            inference_batch["atom"].pos = complex_graph_batch["atom"].pos
            inference_batch["receptor"].pos = complex_graph_batch["receptor"].pos
        else:
            inference_batch = complex_graph_batch

        set_time(
            inference_batch,
            t=0,
            t_tr=0,
            t_rot=0,
            t_tor=0,
            t_sidechain_tor=1 if sidechain_tor_bridge else 0,
            t_bb_tr=1,
            t_bb_rot=1,
            batch_size=inference_batch.num_graphs,
            all_atoms=confidence_all_atoms,
            device=device,
        )
        with torch.no_grad():
            pred = confidence_model(inference_batch)
        if "filtering_pred" not in pred:
            raise ValueError("Confidence model output missing 'filtering_pred'.")
        confidence.append(pred["filtering_pred"])

    if not confidence:
        raise ValueError("Confidence model produced no scores.")
    return torch.cat(confidence, dim=0)


def autocast_context(device, precision: str | None):
    torch_device = torch.device(device)
    dtype = autocast_dtype(precision)
    if torch_device.type != "cuda" or dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=torch_device.type, dtype=dtype)


def autocast_dtype(precision: str | None) -> torch.dtype | None:
    value = str(precision or "fp32").lower()
    if value in {"amp", "fp16", "float16", "16", "16-mixed", "fp16-mixed"}:
        return torch.float16
    if value in {"bf16", "bfloat16", "bf16-mixed"}:
        return torch.bfloat16
    return None
