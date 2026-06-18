from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from e3nn import o3
from torch import nn
from torch_cluster import knn_graph, radius, radius_graph

from docking_model.models.encoders import GaussianSmearing


@dataclass
class BuiltGraph:
    node_attr: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    edge_sh: torch.Tensor
    edge_weight: torch.Tensor | int | float


@dataclass
class CrossGraphs:
    ligand_receptor: BuiltGraph
    ligand_atom: BuiltGraph
    atom_receptor: BuiltGraph


@dataclass
class GraphBuilderOutput:
    ligand: BuiltGraph
    receptor: BuiltGraph
    atom: BuiltGraph
    cross: CrossGraphs
    sigma: dict[str, Any]
    t: dict[str, torch.Tensor]
    data: Any


class GraphBuilderStack(nn.Module):
    """Graph construction for Docking model message passing."""

    def __init__(
        self,
        t_to_sigma: Callable,
        timestep_emb_func: Callable,
        in_lig_edge_features: int = 5,
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        lig_max_radius: float = 5.0,
        rec_max_radius: float = 30.0,
        cross_max_distance: float = 250.0,
        center_max_distance: float = 30.0,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        dynamic_max_cross: bool = False,
        smooth_edges: bool = False,
        asyncronous_noise_schedule: bool = False,
        no_aminoacid_identities: bool = False,
        flexible_sidechains: bool = False,
        flexible_backbone: bool = False,
        c_alpha_radius: float = 20.0,
        c_alpha_max_neighbors: int | None = None,
        atom_radius: float = 5.0,
        atom_max_neighbors: int | None = None,
        only_nearby_residues_atomic: bool = False,
    ):
        super().__init__()
        self.t_to_sigma = t_to_sigma
        self.timestep_emb_func = timestep_emb_func
        self.in_lig_edge_features = in_lig_edge_features
        self.sigma_embed_dim = sigma_embed_dim
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.lig_max_radius = lig_max_radius
        self.rec_max_radius = rec_max_radius
        self.cross_max_distance = cross_max_distance
        self.center_max_distance = center_max_distance
        self.dynamic_max_cross = dynamic_max_cross
        self.smooth_edges = smooth_edges
        self.asyncronous_noise_schedule = asyncronous_noise_schedule
        self.no_aminoacid_identities = no_aminoacid_identities
        self.flexible_sidechains = flexible_sidechains
        self.flexible_backbone = flexible_backbone
        self.c_alpha_radius = c_alpha_radius
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.atom_radius = atom_radius
        self.atom_max_neighbors = atom_max_neighbors
        self.only_nearby_residues_atomic = only_nearby_residues_atomic

        self.lig_distance_expansion = GaussianSmearing(0.0, lig_max_radius, distance_embed_dim)
        self.rec_distance_expansion = GaussianSmearing(0.0, rec_max_radius, distance_embed_dim)
        self.cross_distance_expansion = GaussianSmearing(0.0, cross_max_distance, cross_distance_embed_dim)
        self.center_distance_expansion = GaussianSmearing(0.0, center_max_distance, distance_embed_dim)

    def forward(self, data) -> GraphBuilderOutput:
        self.prepare_data(data)
        t_dict = self.get_time_dict(data)
        sigma_dict = self.t_to_sigma(t_dict)
        cross_cutoff = (
            (sigma_dict["tr_sigma"] * 3 + 20).unsqueeze(1)
            if self.dynamic_max_cross
            else self.cross_max_distance
        )
        return GraphBuilderOutput(
            ligand=self.build_lig_conv_graph(data),
            receptor=self.build_rec_conv_graph(data),
            atom=self.build_atom_conv_graph(data),
            cross=self.build_cross_conv_graph(data, cross_cutoff),
            sigma=sigma_dict,
            t=t_dict,
            data=data,
        )

    def prepare_data(self, data) -> None:
        if self.no_aminoacid_identities:
            data["receptor"].x = data["receptor"].x * 0
        data["atom"].orig_batch = data["atom"].batch

        if not self.only_nearby_residues_atomic:
            return

        nearby_atoms = data["atom"].nearby_atoms
        data["atom"].ca_mask = data["atom"].ca_mask[nearby_atoms]
        data["atom"].n_mask = data["atom"].n_mask[nearby_atoms]
        data["atom"].c_mask = data["atom"].c_mask[nearby_atoms]

        atom_new_idx_map = torch.zeros(
            data["atom"].x.shape[0], dtype=torch.long, device=nearby_atoms.device
        ) + 1000000000
        atom_new_idx_map[nearby_atoms] = torch.arange(
            len(atom_new_idx_map[nearby_atoms]), device=nearby_atoms.device
        )
        data["atom"].atom_new_idx_map = atom_new_idx_map

        data["atom"].x = data["atom"].x[nearby_atoms]
        data["atom"].pos = data["atom"].pos[nearby_atoms]
        if hasattr(data["atom"], "pos_sc_matched"):
            data["atom"].pos_sc_matched = data["atom"].pos_sc_matched[nearby_atoms]
        data["atom"].batch = data["atom"].batch[nearby_atoms]
        for noise_type, node_t in list(data["atom"].node_t.items()):
            if torch.is_tensor(node_t) and node_t.numel() == nearby_atoms.numel():
                data["atom"].node_t[noise_type] = node_t[nearby_atoms]

        atom_to_res_mapping = self.atom_receptor_edge_index(data)[1][nearby_atoms]
        atom_res_edge_index = torch.stack(
            [
                torch.arange(len(data["atom"].x), device=nearby_atoms.device),
                atom_to_res_mapping,
            ]
        )
        data["atom", "receptor"].edge_index = atom_res_edge_index

    def get_time_dict(self, data) -> dict[str, torch.Tensor]:
        noise_types = ["tr", "rot", "tor"]
        if self.flexible_sidechains:
            noise_types.append("sc_tor")
        if self.flexible_backbone:
            noise_types.extend(["bb_tr", "bb_rot"])
        noise_types.append("t")
        return {noise_type: data.complex_t[noise_type] for noise_type in noise_types}

    def build_lig_conv_graph(self, data) -> BuiltGraph:
        data["ligand"].node_sigma_emb = self.node_sigma_emb(data["ligand"])
        radius_edges = radius_graph(
            data["ligand"].pos, self.lig_max_radius, data["ligand"].batch
        )
        edge_index = torch.cat(
            [data["ligand", "lig_bond", "ligand"].edge_index, radius_edges], 1
        ).long()
        edge_attr = torch.cat(
            [
                data["ligand", "lig_bond", "ligand"].edge_attr[:, : self.in_lig_edge_features],
                torch.zeros(
                    radius_edges.shape[-1],
                    self.in_lig_edge_features,
                    device=data["ligand"].x.device,
                ),
            ],
            0,
        )
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        node_attr = torch.cat([data["ligand"].x, data["ligand"].node_sigma_emb], 1)

        src, dst = edge_index
        edge_vec = data["ligand"].pos[dst.long()] - data["ligand"].pos[src.long()]
        edge_length_emb = self.lig_distance_expansion(edge_vec.norm(dim=-1))
        edge_attr = torch.cat([edge_attr, edge_length_emb], 1)
        edge_sh = self.spherical_harmonics(edge_vec)
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return BuiltGraph(node_attr, edge_index, edge_attr, edge_sh, edge_weight)

    def build_rec_conv_graph(self, data) -> BuiltGraph:
        data["receptor"].node_sigma_emb = self.node_sigma_emb(data["receptor"])
        node_attr = torch.cat([data["receptor"].x, data["receptor"].node_sigma_emb], 1)
        edge_index = knn_graph(
            data["receptor"].pos,
            k=self.c_alpha_max_neighbors if self.c_alpha_max_neighbors else 32,
            batch=data["receptor"].batch,
        )
        edge_vec = data["receptor"].pos[edge_index[1].long()] - data["receptor"].pos[edge_index[0].long()]
        edge_d = edge_vec.norm(dim=-1)
        if self.c_alpha_radius:
            to_keep = edge_d < self.c_alpha_radius
            edge_index = edge_index[:, to_keep]
            edge_vec = edge_vec[to_keep]
            edge_d = edge_d[to_keep]
        edge_length_emb = self.rec_distance_expansion(edge_d)
        edge_sigma_emb = data["receptor"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = self.spherical_harmonics(edge_vec)
        edge_weight = self.get_edge_weight(edge_vec, self.rec_max_radius)
        return BuiltGraph(node_attr, edge_index, edge_attr, edge_sh, edge_weight)

    def build_atom_conv_graph(self, data) -> BuiltGraph:
        data["atom"].node_sigma_emb = self.node_sigma_emb(data["atom"])
        node_attr = torch.cat([data["atom"].x, data["atom"].node_sigma_emb], 1)
        edge_index = knn_graph(
            data["atom"].pos,
            k=self.atom_max_neighbors if self.atom_max_neighbors else 32,
            batch=data["atom"].batch,
        )
        edge_vec = data["atom"].pos[edge_index[1].long()] - data["atom"].pos[edge_index[0].long()]
        edge_d = edge_vec.norm(dim=-1)
        if self.atom_radius:
            to_keep = edge_d < self.atom_radius
            edge_index = edge_index[:, to_keep]
            edge_vec = edge_vec[to_keep]
            edge_d = edge_d[to_keep]
        edge_length_emb = self.lig_distance_expansion(edge_d)
        edge_sigma_emb = data["atom"].node_sigma_emb[edge_index[0].long()]
        edge_attr = torch.cat([edge_sigma_emb, edge_length_emb], 1)
        edge_sh = self.spherical_harmonics(edge_vec)
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return BuiltGraph(node_attr, edge_index, edge_attr, edge_sh, edge_weight)

    def build_cross_conv_graph(self, data, lr_cross_distance_cutoff) -> CrossGraphs:
        if torch.is_tensor(lr_cross_distance_cutoff):
            lr_edge_index = radius(
                data["receptor"].pos / lr_cross_distance_cutoff[data["receptor"].batch],
                data["ligand"].pos / lr_cross_distance_cutoff[data["ligand"].batch],
                1,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        else:
            lr_edge_index = radius(
                data["receptor"].pos,
                data["ligand"].pos,
                lr_cross_distance_cutoff,
                data["receptor"].batch,
                data["ligand"].batch,
                max_num_neighbors=10000,
            )
        lr_edge_vec = data["receptor"].pos[lr_edge_index[1].long()] - data["ligand"].pos[lr_edge_index[0].long()]
        lr_edge_attr = torch.cat(
            [
                data["ligand"].node_sigma_emb[lr_edge_index[0].long()],
                self.cross_distance_expansion(lr_edge_vec.norm(dim=-1)),
            ],
            1,
        )
        lr_edge_sh = self.spherical_harmonics(lr_edge_vec)
        cutoff_d = (
            lr_cross_distance_cutoff[data["ligand"].batch[lr_edge_index[0]]].squeeze()
            if torch.is_tensor(lr_cross_distance_cutoff)
            else lr_cross_distance_cutoff
        )
        lr_edge_weight = self.get_edge_weight(lr_edge_vec, cutoff_d)

        la_edge_index = radius(
            data["atom"].pos,
            data["ligand"].pos,
            self.lig_max_radius,
            data["atom"].batch,
            data["ligand"].batch,
            max_num_neighbors=10000,
        )
        la_edge_vec = data["atom"].pos[la_edge_index[1].long()] - data["ligand"].pos[la_edge_index[0].long()]
        la_edge_attr = torch.cat(
            [
                data["ligand"].node_sigma_emb[la_edge_index[0].long()],
                self.cross_distance_expansion(la_edge_vec.norm(dim=-1)),
            ],
            1,
        )
        la_edge_sh = self.spherical_harmonics(la_edge_vec)
        la_edge_weight = self.get_edge_weight(la_edge_vec, self.lig_max_radius)

        ar_edge_index = self.atom_receptor_edge_index(data)
        ar_edge_vec = data["receptor"].pos[ar_edge_index[1].long()] - data["atom"].pos[ar_edge_index[0].long()]
        ar_edge_attr = torch.cat(
            [
                data["atom"].node_sigma_emb[ar_edge_index[0].long()],
                self.rec_distance_expansion(ar_edge_vec.norm(dim=-1)),
            ],
            1,
        )
        ar_edge_sh = self.spherical_harmonics(ar_edge_vec)
        return CrossGraphs(
            ligand_receptor=BuiltGraph(data["ligand"].x, lr_edge_index, lr_edge_attr, lr_edge_sh, lr_edge_weight),
            ligand_atom=BuiltGraph(data["ligand"].x, la_edge_index, la_edge_attr, la_edge_sh, la_edge_weight),
            atom_receptor=BuiltGraph(data["atom"].x, ar_edge_index, ar_edge_attr, ar_edge_sh, 1),
        )

    def build_center_conv_graph(self, data):
        edge_index = torch.cat(
            [
                data["ligand"].batch.unsqueeze(0),
                torch.arange(len(data["ligand"].batch), device=data["ligand"].x.device).unsqueeze(0),
            ],
            dim=0,
        )
        center_pos = torch.zeros((data.num_graphs, 3), device=data["ligand"].x.device)
        center_pos.index_add_(0, index=data["ligand"].batch, source=data["ligand"].pos)
        center_pos = center_pos / torch.bincount(data["ligand"].batch).unsqueeze(1)
        edge_vec = data["ligand"].pos[edge_index[1]] - center_pos[edge_index[0]]
        edge_attr = self.center_distance_expansion(edge_vec.norm(dim=-1))
        edge_sigma_emb = data["ligand"].node_sigma_emb[edge_index[1].long()]
        edge_attr = torch.cat([edge_attr, edge_sigma_emb], 1)
        edge_sh = self.spherical_harmonics(edge_vec)
        return edge_index, edge_attr, edge_sh

    def build_bond_conv_graph(self, data, edge_embedder: nn.Module):
        bonds = data["ligand", "lig_bond", "ligand"].edge_index[:, data["ligand"].edge_mask].long()
        bond_pos = (data["ligand"].pos[bonds[0]] + data["ligand"].pos[bonds[1]]) / 2
        bond_batch = data["ligand"].batch[bonds[0]]
        edge_index = radius(
            data["ligand"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["ligand"].batch,
            batch_y=bond_batch,
        )
        edge_vec = data["ligand"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = edge_embedder(self.lig_distance_expansion(edge_vec.norm(dim=-1)))
        edge_sh = self.spherical_harmonics(edge_vec)
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return bonds, edge_index, edge_attr, edge_sh, edge_weight

    def build_sidechain_conv_graph(self, data, edge_embedder: nn.Module):
        edge_mask_rotatable = data["atom", "atom_bond", "atom"].edge_mask
        bonds = data["atom", "atom_bond", "atom"].edge_index[:, edge_mask_rotatable]
        bond_batch = data["atom"].orig_batch[bonds[0]]

        if self.only_nearby_residues_atomic:
            bonds = data["atom"].atom_new_idx_map[bonds]
            assert torch.all(bonds < 1000000), "nearby atom filtering produced invalid sidechain bond ids"

        bond_pos = (data["atom"].pos[bonds[0]] + data["atom"].pos[bonds[1]]) / 2
        edge_index = radius(
            data["atom"].pos,
            bond_pos,
            self.lig_max_radius,
            batch_x=data["atom"].batch,
            batch_y=bond_batch,
        )
        edge_vec = data["atom"].pos[edge_index[1]] - bond_pos[edge_index[0]]
        edge_attr = edge_embedder(self.lig_distance_expansion(edge_vec.norm(dim=-1)))
        edge_sh = self.spherical_harmonics(edge_vec)
        edge_weight = self.get_edge_weight(edge_vec, self.lig_max_radius)
        return bonds, edge_index, edge_attr, edge_sh, edge_weight

    def node_sigma_emb(self, store):
        key = "t" if self.asyncronous_noise_schedule else "tr"
        return self.timestep_emb_func(store.node_t[key])

    def spherical_harmonics(self, edge_vec: torch.Tensor) -> torch.Tensor:
        return o3.spherical_harmonics(
            self.sh_irreps, edge_vec, normalize=True, normalization="component"
        )

    def get_edge_weight(self, edge_vec, max_norm):
        if self.smooth_edges:
            normalised_norm = torch.clip(edge_vec.norm(dim=-1) * np.pi / max_norm, max=np.pi)
            return 0.5 * (torch.cos(normalised_norm) + 1.0).unsqueeze(-1)
        return 1.0

    @staticmethod
    def atom_receptor_edge_index(data):
        if ("atom", "receptor") in data.edge_types:
            return data["atom", "receptor"].edge_index
        return data["atom", "atom_rec_contact", "receptor"].edge_index
