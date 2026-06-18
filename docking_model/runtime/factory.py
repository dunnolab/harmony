from __future__ import annotations

from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import torch
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from docking_model.config.schema import DockingConfig
from docking_model.data.datasets import CachedComplexDataset, ListDataset
from docking_model.data.transforms.docking import construct_transform
from docking_model.models.score_model import DockingModel
from docking_model.runtime.checkpoint import load_model_state
from docking_model.runtime.distributed import get_local_rank, get_rank, get_world_size, is_distributed
from docking_model.runtime.seeding import make_generator, seed_worker
from docking_model.sampling.engine import FastSamplingBackend, SamplingEngine
from docking_model.sampling.schedules import get_timestep_embedding, t_to_sigma as schedule_t_to_sigma


def select_device(requested: str) -> torch.device:
    if is_distributed() and torch.cuda.is_available() and (requested == "auto" or requested.startswith("cuda")):
        return torch.device("cuda", get_local_rank())
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def build_optimizer(model: torch.nn.Module, cfg: DockingConfig) -> torch.optim.Optimizer:
    optimizer_cfg = cfg.training.optimizer
    optimizer_cls = torch.optim.AdamW if optimizer_cfg.name == "adamw" else torch.optim.Adam
    return optimizer_cls(
        (param for param in model.parameters() if param.requires_grad),
        lr=optimizer_cfg.lr,
        weight_decay=optimizer_cfg.weight_decay,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: DockingConfig):
    scheduler_name = str(cfg.training.scheduler or "none").lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if cfg.training.inference_earlystop_goal == "max" else "min",
            patience=cfg.training.scheduler_patience,
            factor=cfg.training.scheduler_gamma,
        )
    if scheduler_name == "cosineannealing":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(cfg.training.epochs, 1))
    if scheduler_name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.training.scheduler_gamma)
    raise ValueError(f"Unsupported scheduler: {cfg.training.scheduler}")


def build_t_to_sigma(cfg: DockingConfig):
    return partial(schedule_t_to_sigma, sigma_cfg=cfg.sigma)


def build_timestep_embedding(cfg: DockingConfig):
    return get_timestep_embedding(
        embedding_type=cfg.model.sigma_embed_type,
        embedding_dim=cfg.model.sigma_embed_dim,
        embedding_scale=cfg.model.embedding_scale,
    )


def build_score_model(cfg: DockingConfig) -> DockingModel:
    if not cfg.pocket.all_atoms:
        raise NotImplementedError("Score model construction requires pocket.all_atoms=true.")

    num_confidence_outputs = cfg.model.num_confidence_outputs
    if cfg.model.rmsd_classification_cutoff:
        num_confidence_outputs = max(num_confidence_outputs, len(cfg.model.rmsd_classification_cutoff) + 1)

    lm_embedding_type = "precomputed" if cfg.model.esm_embeddings_path is not None else cfg.model.esm_embeddings_model
    model = DockingModel(
        t_to_sigma=build_t_to_sigma(cfg),
        timestep_emb_func=build_timestep_embedding(cfg),
        in_lig_edge_features=cfg.model.in_lig_edge_features,
        sigma_embed_dim=cfg.model.sigma_embed_dim,
        sh_lmax=cfg.model.sh_lmax,
        ns=cfg.model.ns,
        nv=cfg.model.nv,
        num_conv_layers=cfg.model.num_conv_layers,
        lig_max_radius=cfg.model.ligand_max_radius,
        rec_max_radius=cfg.model.receptor_radius,
        cross_max_distance=cfg.model.cross_max_distance,
        distance_embed_dim=cfg.model.distance_embed_dim,
        cross_distance_embed_dim=cfg.model.cross_distance_embed_dim,
        no_torsion=cfg.ligand.no_torsion,
        scale_by_sigma=cfg.model.scale_by_sigma,
        use_second_order_repr=cfg.model.use_second_order_repr,
        batch_norm=not cfg.model.no_batch_norm,
        norm_type=None if cfg.model.norm_type == "none" else cfg.model.norm_type,
        dynamic_max_cross=cfg.model.dynamic_max_cross,
        dropout=cfg.model.dropout,
        smooth_edges=cfg.model.smooth_edges,
        odd_parity=cfg.model.odd_parity,
        lm_embedding_type=lm_embedding_type,
        confidence_mode=cfg.model.confidence_mode,
        confidence_dropout=cfg.model.confidence_dropout,
        confidence_no_batchnorm=cfg.model.confidence_no_batchnorm,
        num_confidence_outputs=num_confidence_outputs,
        fixed_center_conv=not cfg.model.not_fixed_center_conv,
        no_aminoacid_identities=cfg.model.no_aminoacid_identities,
        flexible_sidechains=cfg.protein.flexible_sidechains,
        flexible_backbone=cfg.protein.flexible_backbone,
        differentiate_convolutions=cfg.model.differentiate_convolutions,
        tp_weights_layers=cfg.model.tp_weights_layers,
        use_oeq_kernels=cfg.model.use_oeq_kernels,
        reduce_pseudoscalars=cfg.model.reduce_pseudoscalars,
        c_alpha_radius=cfg.model.receptor_radius,
        c_alpha_max_neighbors=cfg.model.c_alpha_max_neighbors,
        atom_radius=cfg.model.atom_radius,
        atom_max_neighbors=cfg.model.atom_max_neighbors,
        sidechain_tor_bridge=cfg.protein.sidechain_tor_bridge,
        use_bb_orientation_feats=cfg.protein.use_bb_orientation_feats,
        only_nearby_residues_atomic=cfg.nearby_atoms.restrict_to_nearby,
        atom_lig_confidence=cfg.model.atom_lig_confidence,
        confidence_head_type=cfg.model.confidence_head_type,
        confidence_contact_cutoff=cfg.model.confidence_contact_cutoff,
        confidence_use_time_features=cfg.model.confidence_use_time_features,
        activation_func=cfg.model.activation_func,
        norm_affine=cfg.model.norm_affine,
        clamped_norm_min=cfg.model.clamped_norm_min,
        lig_transform_type=cfg.model.lig_transform_type,
        tor_fourier_enabled=cfg.model.tor_fourier_enabled,
        tor_fourier_num_freqs=cfg.model.tor_fourier_num_freqs,
        sc_tor_fourier_enabled=cfg.model.sc_tor_fourier_enabled,
        sc_tor_fourier_num_freqs=cfg.model.sc_tor_fourier_num_freqs,
        sc_tor_fourier_sigma_conditioning=cfg.model.sc_tor_fourier_sigma_conditioning,
        sc_tor_fourier_gate_type=cfg.model.sc_tor_fourier_gate_type,
        sc_tor_fourier_gate_hidden=cfg.model.sc_tor_fourier_gate_hidden,
        sc_tor_fourier_poly_degree=cfg.model.sc_tor_fourier_poly_degree,
        sc_tor_fourier_joint_time=cfg.model.sc_tor_fourier_joint_time,
        tor_sc_coupling_enabled=cfg.model.tor_sc_coupling_enabled,
        tor_sc_coupling_unary_enabled=cfg.model.tor_sc_coupling_unary_enabled,
        tor_sc_coupling_pairwise_enabled=cfg.model.tor_sc_coupling_pairwise_enabled,
        tor_sc_coupling_radius=cfg.model.tor_sc_coupling_radius,
        tor_sc_coupling_max_neighbors=cfg.model.tor_sc_coupling_max_neighbors,
    )
    model.sigma = cfg.sigma
    model.args = SimpleNamespace(
        all_atoms=cfg.pocket.all_atoms,
        no_torsion=cfg.ligand.no_torsion,
        flexible_backbone=cfg.protein.flexible_backbone,
        flexible_sidechains=cfg.protein.flexible_sidechains,
        sidechain_tor_bridge=cfg.protein.sidechain_tor_bridge,
        use_bb_orientation_feats=cfg.protein.use_bb_orientation_feats,
        sigma=cfg.sigma,
    )
    if cfg.model.checkpoint is not None:
        load_model_state(model, cfg.model.checkpoint, strict=True)
    return model


def resolve_inference_checkpoint(cfg: DockingConfig) -> str | None:
    if cfg.inference.checkpoint is not None:
        return cfg.inference.checkpoint
    if cfg.inference.docking_model_dir is not None and cfg.inference.docking_ckpt is not None:
        return str(Path(cfg.inference.docking_model_dir).expanduser() / cfg.inference.docking_ckpt)
    if cfg.inference.docking_ckpt is not None and cfg.source_path is not None:
        candidate = Path(cfg.source_path).expanduser().parent / cfg.inference.docking_ckpt
        if candidate.exists():
            return str(candidate)
    return None


def build_sampler(cfg: DockingConfig) -> SamplingEngine:
    return SamplingEngine(
        cfg=cfg.sampler,
        sigma_cfg=cfg.sigma,
        time_cfg=cfg.time,
        protein_cfg=cfg.protein,
        backend=FastSamplingBackend(
            sigma_cfg=cfg.sigma,
            protein_cfg=cfg.protein,
            all_atoms=cfg.pocket.all_atoms,
        ),
    )


def build_transform(cfg: DockingConfig, mode: str = "train"):
    return construct_transform(cfg, mode=mode)


def build_cached_loader(
    cfg: DockingConfig,
    split_path: str,
    transform: Callable | None,
    batch_size: int | None = None,
    shuffle: bool = False,
    multiplicity: int | None = None,
    distributed: bool | None = None,
):
    if cfg.data.cache_path is None:
        raise ValueError("data.cache_path is required for cached training loaders.")

    dataset = CachedComplexDataset(
        cache_path=cfg.data.cache_path,
        split_path=split_path,
        transform=transform,
        affinity_csv=cfg.data.affinity_csv,
        esm_embeddings_path=cfg.model.esm_embeddings_path,
        limit_complexes=cfg.data.limit_complexes or None,
        multiplicity=multiplicity if multiplicity is not None else cfg.data.multiplicity,
    )
    use_distributed = is_distributed() if distributed is None else distributed
    sampler = None
    if use_distributed:
        sampler = DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=int(cfg.seed),
            drop_last=bool(cfg.data.drop_last),
        )
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size or cfg.data.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
        worker_init_fn=seed_worker if cfg.data.num_workers > 0 else None,
        generator=make_generator(cfg.seed),
        pin_memory=cfg.data.pin_memory,
        drop_last=cfg.data.drop_last,
    )


def build_validation_inference_loader(
    cfg: DockingConfig,
    validation_dataset,
    transform: Callable | None,
):
    count = min(cfg.data.num_inference_complexes, len(validation_dataset))
    indices = list(range(count))
    if len(indices) == 1:
        indices = indices * 20
    if is_distributed():
        indices = indices[get_rank() :: get_world_size()]
    items = [validation_dataset.get(idx) for idx in indices]
    return DataLoader(
        ListDataset(items, transform=transform),
        batch_size=1,
        shuffle=False,
        generator=make_generator(cfg.seed),
        pin_memory=cfg.data.pin_memory,
    )
