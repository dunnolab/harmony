from __future__ import annotations

from functools import partial

import numpy as np
import torch
from torch_geometric.transforms import BaseTransform, Compose

from docking_model.data.transforms.docking.bb_priors import construct_bb_prior
from docking_model.data.transforms.docking.molecule import LigandTransform
from docking_model.data.transforms.docking.pocket import PocketTransform, UnbalancedTransform
from docking_model.data.transforms.docking.protein import NearbyAtomsTransform, ProteinTransform
from docking_model.sampling.schedules import bridge_transform_t, set_time_t_dict, t_to_sigma as t_to_sigma_compl


class DockingTransform(BaseTransform):
    """Training noiser for ligand diffusion plus optional protein flexibility targets."""

    def __init__(
        self,
        time_config,
        sigma_config,
        lig_transform: LigandTransform,
        prot_transform: ProteinTransform,
        all_atoms: bool,
        include_miscellaneous_atoms: bool = False,
        lig_transform_type: str = "diffusion",
        tor_fourier_enabled: bool = False,
        sidechain_tor_bridge: bool = True,
    ):
        self.all_atoms = all_atoms
        self.include_miscellaneous_atoms = include_miscellaneous_atoms
        self.sidechain_tor_bridge = sidechain_tor_bridge
        if prot_transform.flexible_sidechains:
            if sidechain_tor_bridge and time_config.sc_tor_bridge_alpha is None:
                raise ValueError("Bridge sidechain torsion transform requires time.sc_tor_bridge_alpha.")
            if sidechain_tor_bridge and sigma_config.sidechain_tor_sigma is None:
                raise ValueError("Bridge sidechain torsion transform requires sigma.sidechain_tor_sigma.")
            if not sidechain_tor_bridge and (
                sigma_config.sidechain_tor_sigma_min is None or sigma_config.sidechain_tor_sigma_max is None
            ):
                raise ValueError("VE sidechain torsion transform requires sidechain torsion sigma bounds.")
        if time_config.bb_rot_bridge_alpha is None:
            if sigma_config.bb_rot_sigma is not None or sigma_config.bb_tr_sigma is not None:
                raise ValueError("Backbone bridge sigmas must be null when backbone bridge alphas are null.")
        if tor_fourier_enabled and lig_transform_type != "diffusion":
            raise ValueError("Fourier torsion score model currently supports only lig_transform_type='diffusion'.")

        self.t_to_sigma = partial(t_to_sigma_compl, sigma_cfg=sigma_config)
        self.time_config = time_config
        self.lig_transform = lig_transform
        self.prot_transform = prot_transform
        self.lig_transform_type = lig_transform_type
        self.tor_fourier_enabled = tor_fourier_enabled

    def sample_t(self, data):
        t_lig = np.random.beta(self.time_config.sampling_alpha, self.time_config.sampling_beta)
        t_dict = {"tr": t_lig, "rot": t_lig, "tor": t_lig, "t": t_lig}

        if self.prot_transform.flexible_sidechains and self.sidechain_tor_bridge:
            t_dict["sc_tor"] = bridge_transform_t(t_lig, self.time_config.sc_tor_bridge_alpha)
        elif self.prot_transform.flexible_sidechains:
            t_dict["sc_tor"] = t_lig
        else:
            t_dict["sc_tor"] = None

        if self.time_config.bb_rot_bridge_alpha is not None:
            t_dict["bb_tr"] = bridge_transform_t(t_lig, self.time_config.bb_tr_bridge_alpha)
            t_dict["bb_rot"] = bridge_transform_t(t_lig, self.time_config.bb_rot_bridge_alpha)
        else:
            t_dict["bb_tr"], t_dict["bb_rot"] = None, None

        set_time_t_dict(
            data,
            t_dict,
            batch_size=1,
            all_atoms=self.all_atoms,
            device=None,
            include_miscellaneous_atoms=self.include_miscellaneous_atoms,
        )
        return t_dict

    def forward(self, data):
        t_dict = self.sample_t(data)
        sigma_dict = self.t_to_sigma(t_dict)
        data = self.lig_transform(data, t_dict, sigma_dict)
        return self.prot_transform(data, t_dict, sigma_dict)


class SetInitTimeTransformInference(BaseTransform):
    def __init__(self, flexible_backbone: bool, flexible_sidechains: bool, all_atoms: bool = True):
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains
        self.all_atoms = all_atoms

    def forward(self, data):
        t0 = torch.tensor([0.0])
        t_dict = {"t": t0, "tr": t0, "rot": t0, "tor": t0}
        if self.flexible_sidechains:
            t_dict["sc_tor"] = t0
        if self.flexible_backbone:
            t_dict["bb_tr"] = t0
            t_dict["bb_rot"] = t0
        set_time_t_dict(data, t_dict, batch_size=1, all_atoms=self.all_atoms, device=None)
        return data


def construct_transform(cfg, mode: str = "train"):
    stages = [
        PocketTransform(
            pocket_reduction=cfg.pocket.enabled,
            pocket_buffer=cfg.pocket.buffer,
            pocket_radius=cfg.pocket.radius,
            pocket_min_size=cfg.pocket.min_size,
            all_atoms=cfg.pocket.all_atoms,
            flexible_backbone=cfg.protein.flexible_backbone,
            flexible_sidechains=cfg.protein.flexible_sidechains,
        ),
        NearbyAtomsTransform(
            only_nearby_residues_atomic=cfg.nearby_atoms.restrict_to_nearby,
            nearby_residues_atomic_radius=cfg.nearby_atoms.radius,
            nearby_residues_atomic_min=cfg.nearby_atoms.min_atoms,
        ),
    ]

    if mode in {"train", "val"}:
        if cfg.transforms.unbalanced.match_max_rmsd is not None and mode == "train":
            stages.append(UnbalancedTransform(match_max_rmsd=cfg.transforms.unbalanced.match_max_rmsd))
        stages.append(
            DockingTransform(
                sigma_config=cfg.sigma,
                time_config=cfg.time,
                lig_transform=LigandTransform(no_torsion=cfg.ligand.no_torsion),
                prot_transform=ProteinTransform(
                    all_atoms=cfg.pocket.all_atoms,
                    flexible_backbone=cfg.protein.flexible_backbone,
                    flexible_sidechains=cfg.protein.flexible_sidechains,
                    sidechain_tor_bridge=cfg.protein.sidechain_tor_bridge,
                    use_bb_orientation_feats=cfg.protein.use_bb_orientation_feats,
                    bb_prior=construct_bb_prior(cfg.transforms.bb_prior),
                ),
                all_atoms=cfg.pocket.all_atoms,
                include_miscellaneous_atoms=False,
                lig_transform_type=cfg.model.lig_transform_type,
                tor_fourier_enabled=cfg.model.tor_fourier_enabled,
                sidechain_tor_bridge=cfg.protein.sidechain_tor_bridge,
            )
        )
    else:
        stages.append(
            SetInitTimeTransformInference(
                flexible_backbone=cfg.protein.flexible_backbone,
                flexible_sidechains=cfg.protein.flexible_sidechains,
                all_atoms=cfg.pocket.all_atoms,
            )
        )
    return Compose(transforms=stages)
