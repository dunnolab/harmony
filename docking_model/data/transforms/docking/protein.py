from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch_geometric.transforms import BaseTransform

from docking_model.data.conformers.modify import modify_conformer_torsion_angles
from docking_model.data.feature.helpers import (
    compute_nearby_atom_mask,
    rotate_backbone_torch,
    to_atom_grid_torch,
)
from docking_model.geometry.manifolds import so3, torus
from docking_model.geometry.ops import axis_angle_to_matrix


class NearbyAtomsTransform(BaseTransform):
    def __init__(
        self,
        only_nearby_residues_atomic: bool,
        nearby_residues_atomic_radius: float,
        nearby_residues_atomic_min: int,
    ):
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.nearby_residues_atomic_radius = nearby_residues_atomic_radius
        self.nearby_residues_atomic_min = nearby_residues_atomic_min

    def compute_nearby_atoms(self, data):
        if data["receptor"].lens_receptors.numel() == 0:
            raise ValueError(f"Empty receptor pocket for complex {getattr(data, 'name', '<unknown>')}")
        nearby_atoms, nearby_residues = compute_nearby_atom_mask(
            atom_pos=data["atom"].orig_holo_pos,
            lens_receptors=data["receptor"].lens_receptors,
            ligand_atoms=data["ligand"].orig_pos,
            nearby_residues_atomic_radius=self.nearby_residues_atomic_radius,
            nearby_residues_atomic_min=self.nearby_residues_atomic_min,
        )
        data["receptor"].nearby_residues = nearby_residues
        return nearby_atoms

    def forward(self, data):
        if not self.only_nearby_residues_atomic:
            data["atom"].nearby_atoms = torch.ones(data["atom"].pos.shape[0], dtype=torch.bool, device=data["atom"].pos.device)
            data["receptor"].nearby_residues = torch.arange(data["receptor"].pos.shape[0], device=data["receptor"].pos.device)
            return data

        if "nearby_atoms" not in data["atom"]:
            data["atom"].nearby_atoms = self.compute_nearby_atoms(data)

        atom_edge_index = data["atom", "atom_bond", "atom"].edge_index
        nearby_atom_edges = data["atom"].nearby_atoms[atom_edge_index[0]] & data["atom"].nearby_atoms[atom_edge_index[1]]
        data["atom", "atom_bond", "atom"].edge_mask[~nearby_atom_edges] = False
        return data


class ProteinTransform:
    def __init__(
        self,
        all_atoms: bool = True,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
        sidechain_tor_bridge: bool = False,
        use_bb_orientation_feats: bool = False,
        bb_prior=None,
    ):
        self.all_atoms = all_atoms
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains
        self.sidechain_tor_bridge = sidechain_tor_bridge
        self.use_bb_orientation_feats = use_bb_orientation_feats
        self.bb_prior = bb_prior

    def __call__(self, data, t_dict, sigma_dict):
        if self.flexible_backbone:
            data = self.apply_backbone_transform(data, t_dict, sigma_dict)
        if self.flexible_sidechains and self.all_atoms:
            data = self.apply_sidechain_transform(data, t_dict, sigma_dict)
        if self.use_bb_orientation_feats and "bb_orientation" not in data["receptor"]:
            self.set_bb_orientation(data)
        return data

    def apply_backbone_transform(self, data, t_dict, sigma_dict):
        t_bb_tr, t_bb_rot = t_dict["bb_tr"], t_dict["bb_rot"]
        bb_tr_sigma = sigma_dict["bb_tr_sigma"]
        bb_rot_sigma = sigma_dict["bb_rot_sigma"]
        calpha_mask = data["atom"].ca_mask
        calpha_apo = data["atom"].orig_aligned_apo_pos[calpha_mask]
        calpha_holo = data["atom"].orig_holo_pos[calpha_mask]
        if self.bb_prior is not None:
            calpha_apo = self.bb_prior(calpha_apo, calpha_holo)

        bb_rot_delta_holo = data["receptor"].rot_vec
        calpha_atoms_mu_t = calpha_apo * (1 - t_bb_tr) + calpha_holo * t_bb_tr
        sigma_t = bb_tr_sigma * np.sqrt(t_bb_tr * (1 - t_bb_tr))
        calpha_atoms_t = calpha_atoms_mu_t + sigma_t * torch.randn_like(calpha_atoms_mu_t)
        data.bb_tr_drift = (calpha_holo - calpha_atoms_t) / (1 - t_bb_tr)

        bb_rot_delta_mu_t = so3.exp_map_at_point(
            tangent_vec=so3.log_map_at_point(
                point=t_bb_rot * bb_rot_delta_holo,
                base_point=torch.zeros_like(bb_rot_delta_holo),
            ),
            base_point=torch.zeros_like(bb_rot_delta_holo),
        )
        sigma_t = bb_rot_sigma * np.sqrt(t_bb_rot * (1 - t_bb_rot))
        bb_rot_delta_t = so3.sample_from_igso3(mu=bb_rot_delta_mu_t, sigma=sigma_t)
        data.bb_rot_drift = so3.log_map_at_point(point=bb_rot_delta_holo, base_point=bb_rot_delta_t) / (1 - t_bb_rot)
        if not torch.is_tensor(data.bb_rot_drift):
            data.bb_rot_drift = torch.tensor(data.bb_rot_drift)

        rot_holo_to_t = Rotation.from_rotvec(-bb_rot_delta_holo.detach().cpu().numpy()) * Rotation.from_rotvec(
            bb_rot_delta_t.detach().cpu().numpy()
        )
        rot_holo_to_t = torch.tensor(rot_holo_to_t.as_rotvec()).float()

        new_pos, _ = rotate_backbone_torch(
            atoms=data["atom"].pos,
            t_vec=calpha_atoms_t - calpha_holo,
            rot_mat=axis_angle_to_matrix(rot_holo_to_t),
            lens_receptors=data["receptor"].lens_receptors,
            detach=False,
            total_rot=None,
        )
        data["atom"].pos = new_pos.float()
        data["receptor"].pos = data["atom"].pos[calpha_mask]
        if self.use_bb_orientation_feats:
            self.set_bb_orientation(data)
        return data

    def apply_sidechain_transform(self, data, t_dict, sigma_dict):
        if self.sidechain_tor_bridge:
            return self.apply_sidechain_bridge_transform(data, t_dict, sigma_dict)
        return self.apply_sidechain_diffusion_transform(data, sigma_dict)

    def apply_sidechain_bridge_transform(self, data, t_dict, sigma_dict):
        sc_tor_sigma = sigma_dict["sc_tor_sigma"]
        t_sc_tor = t_dict["sc_tor"]
        if sc_tor_sigma is None:
            raise ValueError("sc_tor_sigma cannot be None when flexible_sidechains=True.")

        sc_tor_delta_holo = self.get_sidechain_tor_clean_delta(data)
        sigma_t = torch.as_tensor(
            sc_tor_sigma * np.sqrt(t_sc_tor * (1 - t_sc_tor)),
            dtype=sc_tor_delta_holo.dtype,
            device=sc_tor_delta_holo.device,
        )
        sc_tor_delta_mu_t = torus.exp_map_at_point(
            tangent_vec=t_sc_tor
            * torus.log_map_at_point(point=sc_tor_delta_holo, base_point=torch.zeros_like(sc_tor_delta_holo)),
            base_point=torch.zeros_like(sc_tor_delta_holo),
        )
        sc_tor_delta_t = torus.sample_from_wrapped_normal(mu=sc_tor_delta_mu_t, sigma=sigma_t)
        data.sidechain_tor_theta = (sc_tor_delta_t + torch.pi) % (2 * torch.pi) - torch.pi
        data.sidechain_tor_score = torus.log_map_at_point(point=sc_tor_delta_holo, base_point=sc_tor_delta_t) / (1 - t_sc_tor)
        data.sidechain_tor_sigma_edge = np.ones(len(data.sidechain_tor_score))
        return self.apply_sidechain_torsion_updates(data, sc_tor_delta_t - sc_tor_delta_holo)

    def apply_sidechain_diffusion_transform(self, data, sigma_dict):
        sc_tor_sigma = sigma_dict["sc_tor_sigma"]
        if sc_tor_sigma is None:
            raise ValueError("sc_tor_sigma cannot be None when flexible_sidechains=True.")

        num_torsions = int(data["atom", "atom_bond", "atom"].edge_mask.sum().item())
        device = data["atom"].pos.device
        dtype = data["atom"].pos.dtype
        if num_torsions == 0:
            data.sidechain_tor_theta = torch.empty(0, device=device, dtype=dtype)
            data.sidechain_tor_score = torch.empty(0, device=device, dtype=dtype)
            data.sidechain_tor_sigma_edge = np.empty(0, dtype=np.float32)
            return data

        sc_tor_sigma = torch.as_tensor(sc_tor_sigma, dtype=dtype, device=device)
        sc_tor_noise = torch.randn(num_torsions, device=device, dtype=dtype) * sc_tor_sigma
        data.sidechain_tor_theta = (sc_tor_noise + torch.pi) % (2 * torch.pi) - torch.pi
        sc_tor_sigma_edge = np.ones(num_torsions, dtype=np.float32) * float(sc_tor_sigma.detach().cpu().item())
        data.sidechain_tor_score = torch.from_numpy(
            torus.score(sc_tor_noise.detach().cpu().numpy(), sc_tor_sigma_edge)
        ).to(device=device, dtype=dtype)
        data.sidechain_tor_sigma_edge = sc_tor_sigma_edge
        return self.apply_sidechain_torsion_updates(data, sc_tor_noise)

    def get_sidechain_tor_clean_delta(self, data):
        edge_store = data["atom", "atom_bond", "atom"]
        return edge_store.sc_conformer_match_rotations[edge_store.edge_mask]

    def apply_sidechain_torsion_updates(self, data, torsion_updates):
        data["atom"].pos = modify_conformer_torsion_angles(
            pos=data["atom"].pos,
            edge_index=data["atom", "atom_bond", "atom"].edge_index,
            mask_rotate=data["atom", "atom_bond", "atom"].edge_mask,
            fragment_index=data["atom_bond", "atom"].atom_fragment_index,
            torsion_updates=torsion_updates,
            sidechains=True,
        )
        return data

    @staticmethod
    def set_bb_orientation(data) -> None:
        atom_grid, _, _ = to_atom_grid_torch(data["atom"].pos, data["receptor"].lens_receptors)
        data["receptor"].bb_orientation = torch.cat(
            [atom_grid[:, 0] - atom_grid[:, 1], atom_grid[:, 2] - atom_grid[:, 1]],
            dim=1,
        )
