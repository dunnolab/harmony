from __future__ import annotations

import numpy as np
import torch

from docking_model.data.conformers.modify import modify_conformer_fast
from docking_model.geometry.manifolds import so3, torus


class LigandTransform:
    def __init__(self, no_torsion: bool = False):
        self.no_torsion = no_torsion

    def __call__(self, data, t_dict, sigma_dict):
        return self.apply_diffusion_transform(data, t_dict, sigma_dict)

    def apply_diffusion_transform(self, data, t_dict, sigma_dict):
        del t_dict
        tr_sigma = sigma_dict["tr_sigma"]
        rot_sigma = sigma_dict["rot_sigma"]
        tor_sigma = sigma_dict["tor_sigma"] if not self.no_torsion else None

        tr_update = torch.normal(mean=0, std=tr_sigma, size=(1, 3))
        rot_update = so3.sample_vec(eps=rot_sigma)
        torsion_updates = None
        if not self.no_torsion:
            torsion_updates = np.random.normal(
                loc=0.0,
                scale=tor_sigma,
                size=int(data["ligand"].edge_mask.sum().item()),
            )

        modify_conformer_fast(
            data=data,
            tr_update=tr_update,
            rot_update=torch.from_numpy(rot_update).float(),
            torsion_updates=torsion_updates,
        )

        data.tr_score = -tr_update / tr_sigma**2
        data.rot_score = torch.from_numpy(so3.score_vec(vec=rot_update, eps=rot_sigma)).float().unsqueeze(0)
        data.tor_score = None if self.no_torsion else torch.from_numpy(torus.score(torsion_updates, tor_sigma)).float()
        data.tor_sigma_edge = None if self.no_torsion else np.ones(int(data["ligand"].edge_mask.sum().item())) * tor_sigma
        if self.no_torsion:
            data["ligand"].tor_theta = torch.empty(0)
        else:
            tor_theta = torch.from_numpy(np.asarray(torsion_updates)).float()
            data["ligand"].tor_theta = (tor_theta + torch.pi) % (2 * torch.pi) - torch.pi
        return data
