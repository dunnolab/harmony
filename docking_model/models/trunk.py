from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from e3nn import o3
from torch import nn

from docking_model.data.constants import lig_feature_dims, rec_atom_feature_dims, rec_residue_feature_dims
from docking_model.models.encoders import AtomEncoder
from docking_model.models.graph_builders import GraphBuilderOutput
from docking_model.models.layers.tensor_product import TensorProductConvLayer, get_irrep_seq


@dataclass
class TrunkOutput:
    ligand: torch.Tensor
    receptor: torch.Tensor
    atom: torch.Tensor
    ligand_atom_edge_index: torch.Tensor
    ligand_atom_edge_attr: torch.Tensor
    graph: GraphBuilderOutput


class EquivariantTrunk(nn.Module):
    """Embedding and equivariant message passing trunk."""

    def __init__(
        self,
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        ns: int = 16,
        nv: int = 4,
        num_conv_layers: int = 2,
        in_lig_edge_features: int = 5,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
        use_second_order_repr: bool = False,
        reduce_pseudoscalars: bool = False,
        norm_type: Literal["batch_norm", "layer_norm"] | None = None,
        norm_affine: bool = True,
        differentiate_convolutions: bool = True,
        tp_weights_layers: int = 2,
        use_oeq_kernels: bool = False,
        use_bb_orientation_feats: bool = False,
        lm_embedding_type: str | None = None,
    ):
        super().__init__()
        self.ns = ns
        self.nv = nv
        self.num_conv_layers = num_conv_layers
        self.differentiate_convolutions = differentiate_convolutions
        self.use_bb_orientation_feats = use_bb_orientation_feats
        self.activation = activation if activation is not None else nn.ReLU()
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        self.lig_node_embedding = AtomEncoder(
            emb_dim=ns, feature_dims=lig_feature_dims, sigma_embed_dim=sigma_embed_dim
        )
        self.lig_edge_embedding = nn.Sequential(
            nn.Linear(in_lig_edge_features + sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.rec_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_residue_feature_dims,
            sigma_embed_dim=sigma_embed_dim,
            lm_embedding_type="esm" if lm_embedding_type == "precomputed" else None,
        )
        self.rec_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.atom_node_embedding = AtomEncoder(
            emb_dim=ns,
            feature_dims=rec_atom_feature_dims,
            sigma_embed_dim=sigma_embed_dim,
        )
        self.atom_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.lr_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.ar_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.la_edge_embedding = nn.Sequential(
            nn.Linear(sigma_embed_dim + cross_distance_embed_dim, ns),
            self.activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )

        irrep_seq = get_irrep_seq(ns, nv, use_second_order_repr, reduce_pseudoscalars)
        conv_layers = []
        offset = 0
        for idx in range(num_conv_layers):
            if self.use_bb_orientation_feats and idx == 0:
                in_irreps = irrep_seq[min(idx, len(irrep_seq) - 1)] + " + 2x1o"
                offset = 1
            else:
                in_irreps = irrep_seq[min(idx + offset, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(idx + 1 + offset, len(irrep_seq) - 1)]
            conv_layers.append(
                TensorProductConvLayer(
                    in_irreps=in_irreps,
                    sh_irreps=self.sh_irreps,
                    out_irreps=out_irreps,
                    n_edge_features=3 * ns,
                    hidden_features=3 * ns,
                    residual=True,
                    norm_type=norm_type,
                    dropout=dropout,
                    faster=sh_lmax == 1 and not use_second_order_repr,
                    use_oeq_kernels=use_oeq_kernels,
                    tp_weights_layers=tp_weights_layers,
                    edge_groups=1 if not differentiate_convolutions else 9,
                    norm_affine=norm_affine,
                )
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.out_irreps = self.conv_layers[-1].out_irreps

    def forward(self, graphs: GraphBuilderOutput) -> TrunkOutput:
        lig_node_attr = self.lig_node_embedding(graphs.ligand.node_attr)
        lig_edge_attr = self.lig_edge_embedding(graphs.ligand.edge_attr)
        rec_node_attr = self.rec_node_embedding(graphs.receptor.node_attr)
        rec_edge_attr = self.rec_edge_embedding(graphs.receptor.edge_attr)
        atom_node_attr = self.atom_node_embedding(graphs.atom.node_attr)
        atom_edge_attr = self.atom_edge_embedding(graphs.atom.edge_attr)
        lr_edge_attr = self.lr_edge_embedding(graphs.cross.ligand_receptor.edge_attr)
        la_edge_attr = self.la_edge_embedding(graphs.cross.ligand_atom.edge_attr)
        ar_edge_attr = self.ar_edge_embedding(graphs.cross.atom_receptor.edge_attr)

        n_lig, n_rec = len(lig_node_attr), len(rec_node_attr)
        data = graphs.data
        if self.use_bb_orientation_feats:
            rec_node_attr = torch.cat([rec_node_attr, data["receptor"].bb_orientation], dim=1)
            lig_node_attr = torch.cat(
                [lig_node_attr, torch.zeros((len(lig_node_attr), 6), device=lig_node_attr.device)],
                dim=1,
            )
            atom_node_attr = torch.cat(
                [atom_node_attr, torch.zeros((len(atom_node_attr), 6), device=atom_node_attr.device)],
                dim=1,
            )

        rec_edge_index = graphs.receptor.edge_index.clone()
        atom_edge_index = graphs.atom.edge_index.clone()
        lr_edge_index = graphs.cross.ligand_receptor.edge_index.clone()
        la_edge_index = graphs.cross.ligand_atom.edge_index.clone()
        ar_edge_index = graphs.cross.atom_receptor.edge_index.clone()

        rec_edge_index[0], rec_edge_index[1] = rec_edge_index[0] + n_lig, rec_edge_index[1] + n_lig
        atom_edge_index[0], atom_edge_index[1] = (
            atom_edge_index[0] + n_lig + n_rec,
            atom_edge_index[1] + n_lig + n_rec,
        )
        lr_edge_index[1] = lr_edge_index[1] + n_lig
        la_edge_index[1] = la_edge_index[1] + n_lig + n_rec
        ar_edge_index[0], ar_edge_index[1] = ar_edge_index[0] + n_lig + n_rec, ar_edge_index[1] + n_lig

        node_attr = torch.cat([lig_node_attr, rec_node_attr, atom_node_attr], dim=0)
        edge_index = torch.cat(
            [
                graphs.ligand.edge_index,
                lr_edge_index,
                la_edge_index,
                rec_edge_index,
                torch.flip(lr_edge_index, dims=[0]),
                torch.flip(ar_edge_index, dims=[0]),
                atom_edge_index,
                torch.flip(la_edge_index, dims=[0]),
                ar_edge_index,
            ],
            dim=1,
        )
        edge_attr = torch.cat(
            [
                lig_edge_attr,
                lr_edge_attr,
                la_edge_attr,
                rec_edge_attr,
                lr_edge_attr,
                ar_edge_attr,
                atom_edge_attr,
                la_edge_attr,
                ar_edge_attr,
            ],
            dim=0,
        )
        edge_sh = torch.cat(
            [
                graphs.ligand.edge_sh,
                graphs.cross.ligand_receptor.edge_sh,
                graphs.cross.ligand_atom.edge_sh,
                graphs.receptor.edge_sh,
                graphs.cross.ligand_receptor.edge_sh,
                graphs.cross.atom_receptor.edge_sh,
                graphs.atom.edge_sh,
                graphs.cross.ligand_atom.edge_sh,
                graphs.cross.atom_receptor.edge_sh,
            ],
            dim=0,
        )
        edge_weights = [
            graphs.ligand.edge_weight,
            graphs.cross.ligand_receptor.edge_weight,
            graphs.cross.ligand_atom.edge_weight,
            graphs.receptor.edge_weight,
            graphs.cross.ligand_receptor.edge_weight,
            graphs.cross.atom_receptor.edge_weight,
            graphs.atom.edge_weight,
            graphs.cross.ligand_atom.edge_weight,
            graphs.cross.atom_receptor.edge_weight,
        ]
        edge_lengths = [
            len(graphs.ligand.edge_index[0]),
            len(graphs.cross.ligand_receptor.edge_index[0]),
            len(graphs.cross.ligand_atom.edge_index[0]),
            len(graphs.receptor.edge_index[0]),
            len(graphs.cross.ligand_receptor.edge_index[0]),
            len(graphs.cross.atom_receptor.edge_index[0]),
            len(graphs.atom.edge_index[0]),
            len(graphs.cross.ligand_atom.edge_index[0]),
            len(graphs.cross.atom_receptor.edge_index[0]),
        ]
        edge_weight = (
            torch.cat(
                [
                    weight
                    if torch.is_tensor(weight)
                    else torch.ones((length, 1), device=edge_index.device)
                    for weight, length in zip(edge_weights, edge_lengths)
                ],
                dim=0,
            )
            if any(torch.is_tensor(weight) for weight in edge_weights)
            else torch.ones((len(edge_index[0]), 1), device=edge_index.device)
        )
        slices = tuple(
            np.cumsum(
                list(
                    map(
                        len,
                        [
                            lig_edge_attr,
                            lr_edge_attr,
                            la_edge_attr,
                            rec_edge_attr,
                            lr_edge_attr,
                            ar_edge_attr,
                            atom_edge_attr,
                            la_edge_attr,
                            ar_edge_attr,
                        ],
                    )
                )
            ).tolist()
        )

        for conv in self.conv_layers:
            edge_attr_with_nodes = torch.cat(
                [edge_attr, node_attr[edge_index[0], : self.ns], node_attr[edge_index[1], : self.ns]],
                -1,
            )
            if self.differentiate_convolutions:
                s1, s2, s3, s4, s5, s6, s7, s8, _ = slices
                edge_attr_with_nodes = [
                    edge_attr_with_nodes[:s1],
                    edge_attr_with_nodes[s1:s2],
                    edge_attr_with_nodes[s2:s3],
                    edge_attr_with_nodes[s3:s4],
                    edge_attr_with_nodes[s4:s5],
                    edge_attr_with_nodes[s5:s6],
                    edge_attr_with_nodes[s6:s7],
                    edge_attr_with_nodes[s7:s8],
                    edge_attr_with_nodes[s8:],
                ]
            node_attr = conv(node_attr, edge_index, edge_attr_with_nodes, edge_sh, edge_weight=edge_weight)

        lig_node_attr = node_attr[:n_lig]
        rec_node_attr = node_attr[n_lig : n_lig + n_rec]
        atom_node_attr = node_attr[n_lig + n_rec :]
        return TrunkOutput(
            ligand=lig_node_attr,
            receptor=rec_node_attr,
            atom=atom_node_attr,
            ligand_atom_edge_index=graphs.cross.ligand_atom.edge_index,
            ligand_atom_edge_attr=la_edge_attr,
            graph=graphs,
        )
