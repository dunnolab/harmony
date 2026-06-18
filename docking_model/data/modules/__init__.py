from __future__ import annotations

import torch
from torch_geometric.data import HeteroData


class ComplexData(HeteroData):
    """Compatibility class for cached heterographs.

    Some cached graphs were saved as docking_model.data.modules.ComplexData. This
    keeps that pickle import path available while relying on PyG batching.
    """

    def __cat_dim__(self, key, value, store=None, *args, **kwargs):
        if key in {
            "posebusters_edge_index",
            "angle_2_index",
            "angle_index",
            "torsion_index",
            "atom_fragment_index",
            "lig_fragment_index",
        }:
            return -1
        return super().__cat_dim__(key, value, store=store, *args, **kwargs)

    def __inc__(self, key, value, store=None, *args, **kwargs):
        if key in {"posebusters_edge_index", "angle_index", "torsion_index"}:
            return self["ligand"].num_nodes
        if key == "atom_fragment_index":
            return torch.tensor(
                [self["atom", "atom_bond", "atom"].num_edges, self["atom"].num_nodes]
            ).view(-1, 1)
        if key == "lig_fragment_index":
            return torch.tensor(
                [
                    self["ligand", "lig_bond", "ligand"].num_edges,
                    self["ligand"].num_nodes,
                ]
            ).view(-1, 1)
        if key == "angle_2_index":
            return self["ligand", "lig_bond", "ligand"].num_edges
        return super().__inc__(key, value, store=store, *args, **kwargs)
