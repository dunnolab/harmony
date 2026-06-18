from __future__ import annotations

import torch
from torch_geometric.transforms import BaseTransform

from docking_model.data.feature.protein import get_binding_pocket_masks
from docking_model.geometry.ops import rigid_transform_kabsch


class PocketTransform(BaseTransform):
    def __init__(
        self,
        pocket_reduction: bool = False,
        pocket_radius: float = 5.0,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        all_atoms: bool = False,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
    ):
        self.pocket_reduction = pocket_reduction
        self.pocket_radius = pocket_radius
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.all_atoms = all_atoms
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains

    def forward(self, data):
        data["ligand"].orig_pos = torch.as_tensor(
            data["ligand"].orig_pos,
            dtype=torch.float32,
            device=data["ligand"].pos.device,
        )
        data = self.align_apo_to_holo(data)
        if not self.pocket_reduction:
            ca_mask = data["atom"].ca_mask
            if "orig_holo_pos" in data["atom"] and data["atom"].orig_holo_pos is not None:
                center = data["atom"].orig_holo_pos[ca_mask].mean(dim=0, keepdim=True)
            else:
                center = data["atom"].orig_aligned_apo_pos[ca_mask].mean(dim=0, keepdim=True)
            data["atom"].atom_mask = torch.ones(data["atom"].pos.shape[0], dtype=torch.bool, device=data["atom"].pos.device)
            data["receptor"].nearby_residues = torch.arange(data["receptor"].pos.shape[0], device=data["receptor"].pos.device)
            return self.center_complex(data, center)

        pocket_info = self.compute_pocket(data)
        return self.select_pocket(data, pocket_info)

    def align_apo_to_holo(self, data):
        if "orig_apo_pos" not in data["atom"]:
            return data
        apo_pos = data["atom"].orig_apo_pos
        holo_pos = data["atom"].get("orig_holo_pos", None)
        if holo_pos is None:
            data["atom"].orig_aligned_apo_pos = apo_pos.clone()
        else:
            ca_mask = data["atom"].ca_mask
            r_mat, tr_vec = rigid_transform_kabsch(apo_pos[ca_mask], holo_pos[ca_mask])
            data["atom"].orig_aligned_apo_pos = (apo_pos @ r_mat.t()) + tr_vec
        data["receptor"].orig_aligned_apo_pos = data["atom"].orig_aligned_apo_pos[data["atom"].ca_mask]
        return data

    def compute_pocket(self, data):
        apo_pos = (
            data["atom"].orig_aligned_apo_pos
            if "orig_aligned_apo_pos" in data["atom"]
            else data["atom"].orig_apo_pos
        )
        holo_pos = data["atom"].get("orig_holo_pos", None)
        return get_binding_pocket_masks(
            atom_pos=apo_pos,
            ref_atom_pos=holo_pos if holo_pos is not None else apo_pos,
            lig_pos=data["ligand"].orig_pos,
            ca_mask=data["atom"].ca_mask,
            atom_rec_index=data["atom", "atom_rec_contact", "receptor"].edge_index[1],
            pocket_cutoff=self.pocket_radius,
            pocket_buffer=self.pocket_buffer,
            pocket_min_size=self.pocket_min_size,
        )

    def select_pocket(self, data, pocket_info):
        pocket_center, res_mask, atom_mask, nearby_residues = pocket_info
        data.pocket_mask = ":" + ",".join(str(idx + 1) for idx in torch.argwhere(res_mask).view(-1).cpu().tolist())
        data["atom"].atom_mask = atom_mask
        data["receptor"].nearby_residues = nearby_residues

        atom_old = torch.arange(data["atom"].pos.size(0), device=atom_mask.device)[atom_mask]
        atom_new = torch.arange(atom_old.numel(), device=atom_mask.device)
        atom_map = {int(old): int(new) for old, new in zip(atom_old.cpu(), atom_new.cpu())}
        res_old = torch.arange(data["receptor"].x.size(0), device=res_mask.device)[res_mask]
        res_new = torch.arange(res_old.numel(), device=res_mask.device)
        res_map = {int(old): int(new) for old, new in zip(res_old.cpu(), res_new.cpu())}

        self.filter_node_store(data["receptor"], res_mask)
        self.filter_node_store(data["atom"], atom_mask)

        edge_store = data["atom", "atom_bond", "atom"]
        atom_edge_index = edge_store.edge_index
        keep_edges = atom_mask[atom_edge_index[0]] & atom_mask[atom_edge_index[1]]
        kept_edge_old = torch.arange(atom_edge_index.size(1), device=atom_edge_index.device)[keep_edges]
        edge_map = {int(old): int(new) for new, old in enumerate(kept_edge_old.cpu())}
        filtered_edge_index = atom_edge_index[:, keep_edges].clone()
        filtered_edge_index[0].apply_(lambda x: atom_map[int(x)])
        filtered_edge_index[1].apply_(lambda x: atom_map[int(x)])
        edge_store.edge_index = filtered_edge_index
        self.filter_edge_store(edge_store, keep_edges)

        if "atom_fragment_index" in data["atom_bond", "atom"]:
            fragment = data["atom_bond", "atom"].atom_fragment_index
            edge_keep = keep_edges[fragment[0]]
            atom_keep = atom_mask[fragment[1]]
            keep_fragment = edge_keep & atom_keep
            fragment = fragment[:, keep_fragment].clone()
            fragment[0].apply_(lambda x: edge_map[int(x)])
            fragment[1].apply_(lambda x: atom_map[int(x)])
            data["atom_bond", "atom"].atom_fragment_index = fragment

        if ("atom", "atom_rec_contact", "receptor") in data.edge_types:
            ar_store = data["atom", "atom_rec_contact", "receptor"]
            atom_idx, res_idx = ar_store.edge_index
            keep_ar = atom_mask[atom_idx] & res_mask[res_idx]
            atom_idx = atom_idx[keep_ar].clone()
            res_idx = res_idx[keep_ar].clone()
            atom_idx.apply_(lambda x: atom_map[int(x)])
            res_idx.apply_(lambda x: res_map[int(x)])
            ar_store.edge_index = torch.stack([atom_idx, res_idx], dim=0)

        if "res_to_rotate" in edge_store:
            res_ids = edge_store.res_to_rotate[:, 0]
            keep_res = res_mask[res_ids]
            res_ids = res_ids[keep_res].clone()
            res_ids.apply_(lambda x: res_map[int(x)])
            edge_store.res_to_rotate = torch.stack([res_ids, torch.arange(len(res_ids), device=res_ids.device)], dim=1)

        if pocket_center.ndim == 1:
            pocket_center = pocket_center[None, :]
        return self.center_complex(data, pocket_center)

    def center_complex(self, data, pocket_center):
        for store_name, keys in {
            "receptor": ["pos", "orig_apo_pos", "orig_holo_pos", "orig_aligned_apo_pos"],
            "atom": ["pos", "orig_apo_pos", "orig_holo_pos", "orig_aligned_apo_pos", "pos_sc_matched"],
            "ligand": ["pos", "orig_pos", "orig_aligned_apo_pos"],
        }.items():
            store = data[store_name]
            for key in keys:
                if key in store and store[key] is not None:
                    store[key] = store[key] - pocket_center.to(store[key].device)
        data.original_center = pocket_center
        return data

    @staticmethod
    def filter_node_store(store, mask):
        node_count = int(mask.shape[0])
        for key in list(store.keys()):
            value = store[key]
            if torch.is_tensor(value) and value.shape[:1] == (node_count,):
                store[key] = value[mask]

    @staticmethod
    def filter_edge_store(store, mask):
        edge_count = int(mask.shape[0])
        for key in list(store.keys()):
            if key == "edge_index":
                continue
            value = store[key]
            if torch.is_tensor(value) and value.shape[:1] == (edge_count,):
                store[key] = value[mask]


class UnbalancedTransform(BaseTransform):
    def __init__(self, match_max_rmsd: float | None = None):
        self.match_max_rmsd = match_max_rmsd

    def forward(self, data):
        if self.match_max_rmsd is None:
            data.loss_weight = 1.0
            return data

        aligned_apo_pos = data["atom"].orig_aligned_apo_pos[data["atom"].ca_mask]
        holo_pos = data["atom"].orig_holo_pos[data["atom"].ca_mask]
        rmsd = torch.sqrt(torch.mean(torch.sum((holo_pos - aligned_apo_pos) ** 2, axis=1))).item()
        data.loss_weight = 0.0 if rmsd > self.match_max_rmsd else 1.0
        return data


PocketTransformFixed = PocketTransform
