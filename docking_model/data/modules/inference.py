import os
import logging
import pandas as pd
import re

import torch
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader import DataLoader
from lightning.pytorch import LightningDataModule

from docking_model.data.feature.featurizer import Featurizer, FeaturizerConfig
from docking_model.data.parse.parser import ComplexParser
from docking_model.data.parse.molecule import read_single_mol, resolve_ligand_path
from docking_model.data.parse.protein import parse_pdb_from_path as parse_pdb_pmd
from docking_model.data.modules import ComplexData
from docking_model.data.feature.helpers import compute_nearby_atom_mask
from docking_model.data.feature.molecule import get_posebusters_edge_index
from docking_model.data.feature.protein import get_binding_pocket_masks
from docking_model.data.transforms.docking import construct_transform


def normalize_nearby_residues_selection_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in {"direct", "radius_based"}:
        raise ValueError(
            "nearby_residues_selection_mode must be one of {'direct', 'radius_based'}."
        )
    return normalized


class PredictionDataset(Dataset):
    def __init__(
        self,
        input_csv: str,
        featurizer_cfg: FeaturizerConfig,
        esm_embeddings_path: str = None,
        limit_complexes: int = None,
        complex_id: str = None,
        pocket_reduction: bool = False,
        pocket_radius: float = 5.0,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        pocket_selection_mode: str = "center_buffer",
        only_nearby_residues_atomic: bool = False,
        nearby_residues_selection_mode: str = "direct",
        nearby_residues_atomic_radius: float = 6.0,
        nearby_residues_atomic_min: int = 8,
        all_atoms = None,
        flexible_backbone = None,
        flexible_sidechains = None,
        load_reference_metrics: bool = True,
    ):
        super().__init__()
        self.input_df = pd.read_csv(input_csv, index_col=None)
        if complex_id is not None:
            if "pdbid" not in self.input_df.columns:
                raise ValueError(
                    f"Cannot select complex_id={complex_id}: input CSV has no 'pdbid' column"
                )
            complex_id_normalized = str(complex_id).upper()
            selected = self.input_df[
                self.input_df["pdbid"].astype(str).str.upper()
                == complex_id_normalized
            ]
            if selected.empty:
                raise ValueError(
                    f"complex_id={complex_id} was not found in input CSV {input_csv}"
                )
            self.input_df = selected.reset_index(drop=True)
        else:
            self.input_df = self.input_df.reset_index(drop=True)
        self.parser = ComplexParser(esm_embeddings_path=esm_embeddings_path)
        self.featurizer = Featurizer.from_config(cfg=featurizer_cfg)
        self.remove_hs = bool(featurizer_cfg.remove_hs)

        self.limit_complexes = limit_complexes

        self.pocket_reduction = pocket_reduction
        self.pocket_radius = pocket_radius
        self.pocket_buffer = pocket_buffer
        self.pocket_min_size = pocket_min_size
        self.pocket_selection_mode = pocket_selection_mode
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.nearby_residues_selection_mode = normalize_nearby_residues_selection_mode(
            nearby_residues_selection_mode
        )
        self.nearby_residues_atomic_radius = nearby_residues_atomic_radius
        self.nearby_residues_atomic_min = nearby_residues_atomic_min

        self.all_atoms = all_atoms
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains
        self.load_reference_metrics = load_reference_metrics

    def get(self, idx):
        complex_info = self.input_df.iloc[idx]

        input_dict = self.prepare_input_dict(complex_info)

        try:
            complex_inputs = self.parser.parse_complex(complex_dict=input_dict)
        except Exception as exc:
            logging.exception("Parsing failed for %s due to %s; skipping", input_dict["name"], exc)
            complex_inputs = None
        if complex_inputs is None:
            logging.warning("Parsing failed for %s; skipping", input_dict["name"])
            complex_graph = ComplexData()
            complex_graph["success"] = False
            complex_graph["name"] = input_dict["name"]
            return complex_graph

        try:
            output_features = self.featurizer.featurize_complex(
                complex_inputs=complex_inputs
            )
        except Exception as exc:
            logging.exception(
                "Featurization failed for %s due to %s; skipping",
                input_dict["name"],
                exc,
            )
            output_features = None
        if output_features is None:
            logging.warning(
                "Featurization failed for %s; skipping", input_dict["name"]
            )
            complex_graph = ComplexData()
            complex_graph["success"] = False
            complex_graph["name"] = input_dict["name"]
            return complex_graph

        complex_graph = output_features["complex_graph"]
        for metadata_key in [
            "base_dir",
            "apo_rec_path",
            "holo_rec_path_for_metrics",
            "ligand_input",
            "ligand_true_file",
            "pocket_ligand_file",
            "pocket_residues",
            "ligand_description",
        ]:
            metadata_value = input_dict.get(metadata_key)
            if metadata_value is not None:
                complex_graph[metadata_key] = str(metadata_value)
        complex_graph["atom"].pos = complex_graph["atom"].orig_apo_pos
        complex_graph["receptor"].pos = complex_graph["atom"].pos[
            complex_graph["atom"].ca_mask
        ]
        complex_graph["atom"].orig_aligned_apo_pos = complex_graph["atom"].orig_apo_pos
        if self.load_reference_metrics:
            self.set_reference_protein_pos(complex_graph, input_dict)
            self.set_reference_ligand_pos(complex_graph, input_dict)

        ligand_orig_pos = (
            complex_graph["ligand"].orig_pos
            if "orig_pos" in complex_graph["ligand"]
            else complex_graph["ligand"].pos
        )
        complex_graph["ligand"].orig_pos = torch.as_tensor(
            ligand_orig_pos,
            dtype=torch.float32,
            device=complex_graph["ligand"].pos.device,
        ).clone()

        if self.pocket_reduction:
            pocket_residue_idxs = self.load_pocket_residue_idxs(
                complex_info,
                complex_graph,
            )
            pocket_ligand_pos = None

            if pocket_residue_idxs is None:
                pocket_ligand_pos = self.load_pocket_ligand_pos(
                    complex_info=complex_info,
                    complex_graph=complex_graph,
                )

            if pocket_ligand_pos is not None:
                complex_graph = self.move_ligand_to_target_centroid(
                    complex_graph=complex_graph,
                    target_ligand_pos=pocket_ligand_pos,
                )

            pocket_info = self.prepare_pocket_info(
                complex_info=complex_info,
                complex_graph=complex_graph,
                pocket_ligand_pos=pocket_ligand_pos,
                pocket_residue_idxs=pocket_residue_idxs,
            )
            complex_graph = self.select_pocket_and_buffer(
                complex_graph, pocket_info=pocket_info
            )
            if self.only_nearby_residues_atomic:
                complex_graph = self.select_nearby_atoms(
                    complex_graph,
                    ligand_pos=complex_graph["ligand"].pos,
                )
        else:
            center = complex_graph["receptor"].pos.mean(dim=0, keepdim=True)
            complex_graph = self.center_complex(complex_graph, center) 

        complex_graph = get_posebusters_edge_index(complex_graph)

        return complex_graph

    def row_value(self, complex_info, *field_names):
        for field_name in field_names:
            try:
                value = getattr(complex_info, field_name)
            except Exception:
                continue
            if not pd.isna(value):
                return value
        return None

    def prepare_input_dict(self, complex_info):
        apo_rec_path = self.row_value(
            complex_info,
            "apo_protein_file",
            "apo_rec_path",
            "protein_file",
        )
        holo_rec_path_for_metrics = self.row_value(
            complex_info,
            "holo_protein_file",
            "holo_rec_path",
            "holo_path",
            "match_protein_file",
        )
        base_dir = self.row_value(complex_info, "base_dir")
        ligand_input = self.row_value(
            complex_info,
            "ligand_input",
            "ligand_file",
            "ligand",
        )
        ligand_description = self.row_value(complex_info, "ligand_description")
        ligand_true_file = self.row_value(complex_info, "ligand_true_file")
        pocket_ligand_file = self.row_value(complex_info, "pocket_ligand_file")
        pocket_residues = self.row_value(complex_info, "pocket_residues")

        if ligand_description is None:
            ligand_description = "filename"

        if base_dir is None and ligand_input is not None:
            base_dir = os.path.dirname(ligand_input)
        if base_dir is None and apo_rec_path is not None:
            base_dir = os.path.dirname(apo_rec_path)

        name = self.row_value(complex_info, "pdbid", "name", "complex_name")

        return {
            "name": name,
            "base_dir": base_dir,
            "apo_rec_path": apo_rec_path,
            "holo_rec_path_for_metrics": holo_rec_path_for_metrics,
            "ligand_input": ligand_input,
            "ligand_true_file": ligand_true_file,
            "pocket_ligand_file": pocket_ligand_file,
            "pocket_residues": pocket_residues,
            "ligand_description": ligand_description,
            "holo_rec_path": None,
        }

    def set_reference_protein_pos(self, complex_graph, input_dict):
        holo_rec_path = input_dict.get("holo_rec_path_for_metrics")
        if holo_rec_path is None:
            return

        complex_graph["reference_protein_loaded"] = False
        name = input_dict.get("name")
        base_dir = input_dict.get("base_dir")
        if base_dir is not None and not os.path.isabs(str(holo_rec_path)):
            holo_rec_path = os.path.join(str(base_dir), str(holo_rec_path))

        try:
            holo_struct = parse_pdb_pmd(holo_rec_path, remove_hs=True, reorder=True)
        except Exception as exc:
            logging.warning("%s: failed to load holo protein %s for metrics due to %s", name, holo_rec_path, exc)
            return

        holo_pos = torch.as_tensor(
            holo_struct.get_coordinates(0),
            dtype=torch.float32,
            device=complex_graph["atom"].pos.device,
        )
        if holo_pos.shape != complex_graph["atom"].orig_apo_pos.shape:
            logging.warning(
                "%s: holo protein atom count does not match inference protein for metrics: holo=%s, graph=%s",
                name,
                holo_pos.shape[0],
                complex_graph["atom"].orig_apo_pos.shape[0],
            )
            return

        complex_graph["atom"].orig_holo_pos = holo_pos
        complex_graph["reference_protein_loaded"] = True

    def set_reference_ligand_pos(self, complex_graph, input_dict):
        ligand_true_file = input_dict.get("ligand_true_file")
        if ligand_true_file is None:
            return

        complex_graph["reference_ligand_loaded"] = False
        name = input_dict.get("name")
        try:
            ligand_path = resolve_ligand_path(
                base_dir=input_dict.get("base_dir"),
                name=name,
                ligand_input=ligand_true_file,
            )
            reference_mol = read_single_mol(ligand_path, remove_hs=self.remove_hs)
        except Exception as exc:
            logging.warning("%s: failed to load true ligand %s for metrics due to %s", name, ligand_true_file, exc)
            return

        if reference_mol is None:
            logging.warning("%s: failed to load true ligand %s for metrics", name, ligand_true_file)
            return

        reference_pos = torch.as_tensor(
            reference_mol.GetConformer().GetPositions(),
            dtype=torch.float32,
            device=complex_graph["ligand"].pos.device,
        )
        if reference_pos.shape[0] != complex_graph["ligand"].pos.shape[0]:
            logging.warning(
                "%s: true ligand atom count does not match inference ligand for metrics: true=%s, graph=%s",
                name,
                reference_pos.shape[0],
                complex_graph["ligand"].pos.shape[0],
            )
            return

        complex_graph["ligand"].orig_pos = reference_pos
        complex_graph["reference_ligand_loaded"] = True

    def load_pocket_residue_idxs(self, complex_info, complex_graph):
        value = self.row_value(complex_info, "pocket_residues")
        if value is None:
            return None

        residue_numbers = [
            int(float(item))
            for item in re.split(r"[\s,;]+", str(value).strip())
            if item
        ]
        if not residue_numbers:
            return None

        num_residues = int(complex_graph["receptor"].x.shape[0])
        residue_idxs = torch.tensor(
            [residue_number - 1 for residue_number in residue_numbers],
            dtype=torch.long,
            device=complex_graph["receptor"].pos.device,
        )
        valid = (residue_idxs >= 0) & (residue_idxs < num_residues)
        if not bool(valid.all()):
            invalid = [
                str(residue_numbers[idx])
                for idx, is_valid in enumerate(valid.cpu().tolist())
                if not is_valid
            ]
            raise ValueError(
                f"{self.row_value(complex_info, 'pdbid', 'name', 'complex_name')}: "
                f"pocket_residues contains residues outside 1..{num_residues}: {','.join(invalid)}"
            )

        return torch.unique(residue_idxs, sorted=True)

    def load_pocket_ligand_pos(self, complex_info, complex_graph):
        pocket_ligand_input = self.row_value(
            complex_info, "pocket_ligand_file", "ligand_true_file"
        )
        if pocket_ligand_input is None:
            return None

        base_dir = self.row_value(complex_info, "base_dir")
        name = self.row_value(complex_info, "pdbid")

        try:
            ligand_path = resolve_ligand_path(
                base_dir=base_dir,
                name=name,
                ligand_input=pocket_ligand_input,
            )
            pocket_ligand = read_single_mol(ligand_path, remove_hs=True)
            if pocket_ligand is None:
                logging.warning(
                    "%s: failed to load pocket ligand from %s; using docking input ligand for pocketing",
                    name,
                    ligand_path,
                )
                return None

            return torch.as_tensor(
                pocket_ligand.GetConformer().GetPositions(),
                dtype=torch.float32,
                device=complex_graph["ligand"].pos.device,
            )
        except Exception as exc:
            logging.warning(
                "%s: failed to resolve pocket ligand '%s' due to %s; using docking input ligand for pocketing",
                name,
                pocket_ligand_input,
                exc,
            )
            return None

    def move_ligand_to_target_centroid(self, complex_graph, target_ligand_pos):
        ligand_centroid = complex_graph["ligand"].pos.mean(dim=0, keepdim=True)
        target_centroid = target_ligand_pos.mean(dim=0, keepdim=True)
        centroid_shift = target_centroid - ligand_centroid
        complex_graph["ligand"].pos = complex_graph["ligand"].pos + centroid_shift
        return complex_graph

    def compute_pocket(self, data, pocket_ligand_pos=None):
        apo_rec_pos = data["atom"].orig_apo_pos
        if pocket_ligand_pos is None:
            pocket_ligand_pos = data["ligand"].orig_pos
        (
            pocket_center,
            res_pocket_mask,
            atom_pocket_mask,
            nearby_residues,
        ) = get_binding_pocket_masks(
            apo_rec_pos,
            apo_rec_pos,
            pocket_ligand_pos,
            data["atom"].ca_mask,
            data["atom", "atom_rec_contact", "receptor"].edge_index[1],
            pocket_cutoff=self.pocket_radius,
            pocket_min_size=self.pocket_min_size,
            pocket_buffer=self.pocket_buffer,
        )
        return pocket_center, res_pocket_mask, atom_pocket_mask, nearby_residues

    def compute_pocket_from_residues(self, data, pocket_residue_idxs):
        ca_pos = data["receptor"].pos
        atom_rec_index = data["atom", "atom_rec_contact", "receptor"].edge_index[1]
        min_keep = min(max(int(self.pocket_min_size), 1), int(ca_pos.shape[0]))

        pocket_center = ca_pos.index_select(0, pocket_residue_idxs).mean(dim=0)
        res_subset_mask = (
            torch.linalg.norm(ca_pos - pocket_center, dim=-1) < self.pocket_buffer
        )
        res_subset_mask[pocket_residue_idxs] = True

        if res_subset_mask.sum() < min_keep:
            pocket_res_dists = torch.linalg.norm(ca_pos - pocket_center, dim=-1)
            _, closest_residues = torch.topk(
                pocket_res_dists,
                k=min_keep,
                largest=False,
            )
            res_subset_mask[closest_residues] = True

        atom_subset_mask = torch.index_select(res_subset_mask, -1, atom_rec_index)
        return pocket_center, res_subset_mask, atom_subset_mask, pocket_residue_idxs

    def prepare_pocket_info(
        self,
        complex_info,
        complex_graph,
        pocket_ligand_pos=None,
        pocket_residue_idxs=None,
    ):
        if pocket_residue_idxs is not None:
            pocket_center, res_subset_mask, atom_subset_mask, nearby_residues = self.compute_pocket_from_residues(
                complex_graph,
                pocket_residue_idxs=pocket_residue_idxs,
            )
        else:
            pocket_center, res_subset_mask, atom_subset_mask, nearby_residues = self.compute_pocket(
                complex_graph,
                pocket_ligand_pos=pocket_ligand_pos,
            )

        # Mask for only pocket residues
        num_residues = complex_graph["receptor"].x.shape[0]
        pocket_residue_mask = torch.zeros((num_residues,), dtype=torch.bool)
        pocket_residue_mask[nearby_residues] = True

        # Mask for atoms associated with pocket residues
        pocket_atom_mask = torch.index_select(
            pocket_residue_mask,
            dim=-1,
            index=complex_graph["atom", "atom_rec_contact", "receptor"].edge_index[1],
        )

        amber_subset_mask = ":" + ",".join(
            [
                str(idx + 1)
                for idx in torch.argwhere(res_subset_mask).view(-1).cpu().numpy().tolist()
            ]
        )
        amber_pocket_mask = ":" + ",".join(map(str, nearby_residues.view(-1).cpu().numpy()))
    
        pocket_info = {
            "amber_pocket_mask": amber_pocket_mask,
            "amber_subset_mask": amber_subset_mask,
            "pocket_residues_idxs": nearby_residues,
            "res_subset_mask": res_subset_mask,
            "atom_subset_mask": atom_subset_mask,
            "pocket_center": pocket_center,
            "pocket_atom_mask": pocket_atom_mask,
        }

        return pocket_info

    def select_pocket_and_buffer(self, complex_graph, pocket_info):
        res_mask = pocket_info["res_subset_mask"]
        atom_mask = pocket_info["atom_subset_mask"]
        amber_subset_mask = pocket_info["amber_subset_mask"]

        pocket_atom_mask = pocket_info["pocket_atom_mask"]

        complex_graph.amber_subset_mask = amber_subset_mask

        complex_graph["atom"].atom_mask = atom_mask
        complex_graph["atom"].nearby_atoms = pocket_atom_mask.clone()

        # Update atom numbering
        atom_numbering_old = torch.arange(complex_graph["atom"].pos.size(0))
        atom_numbering_old = atom_numbering_old[atom_mask]
        atom_numbering_new = torch.arange(atom_mask.sum())
        atom_numbering_dict = dict(
            zip(atom_numbering_old.numpy(), atom_numbering_new.numpy())
        )

        residue_numbering_old = torch.arange(complex_graph["receptor"].x.size(0))
        residue_numbering_old = residue_numbering_old[res_mask]
        residue_numbering_new = torch.arange(res_mask.sum())
        residue_numbering_dict = dict(
            zip(residue_numbering_old.numpy(), residue_numbering_new.numpy())
        )

        # Update pocket + buffer residue attributes
        complex_graph["receptor"].x = complex_graph["receptor"].x[res_mask]
        complex_graph["receptor"].pos = complex_graph["receptor"].pos[res_mask]
        complex_graph["receptor"].lens_receptors = complex_graph[
            "receptor"
        ].lens_receptors[res_mask]

        # Update pocket + buffer atom attributes
        complex_graph["atom"].x = complex_graph["atom"].x[atom_mask]
        complex_graph["atom"].vdw_radii = complex_graph["atom"].vdw_radii[atom_mask]

        complex_graph["atom"].pos = complex_graph["atom"].pos[atom_mask]
        complex_graph["atom"].orig_apo_pos = complex_graph["atom"].orig_apo_pos[
            atom_mask
        ]
        complex_graph["atom"].orig_aligned_apo_pos = complex_graph[
            "atom"
        ].orig_aligned_apo_pos[atom_mask]
        if "orig_holo_pos" in complex_graph["atom"]:
            complex_graph["atom"].orig_holo_pos = complex_graph["atom"].orig_holo_pos[
                atom_mask
            ]

        complex_graph["atom"].ca_mask = complex_graph["atom"].ca_mask[atom_mask]
        complex_graph["atom"].c_mask = complex_graph["atom"].c_mask[atom_mask]
        complex_graph["atom"].n_mask = complex_graph["atom"].n_mask[atom_mask]
        complex_graph["atom"].nearby_atoms = complex_graph["atom"].nearby_atoms[
            atom_mask
        ]

        # Gather edges between atoms in pocket + buffer
        atom_edge_index = complex_graph["atom", "atom_bond", "atom"].edge_index
        edges_in_subset = atom_mask[atom_edge_index[0]] & atom_mask[atom_edge_index[1]]

        # Create new edge numbering (used in fragment_index)
        edges_order_old = torch.arange(atom_edge_index.size(1))
        edges_order_old = edges_order_old[edges_in_subset]
        edges_order_new = torch.arange(edges_order_old.size(0))
        edge_numbering_dict = dict(
            zip(edges_order_old.numpy(), edges_order_new.numpy())
        )

        # Which edge rotates which atoms in topological sorted order
        atom_fragment_index = complex_graph["atom_bond", "atom"].atom_fragment_index
        fragment_old_edge_order, fragment_old_atom_idx = atom_fragment_index
        fragment_edge_subset_mask = edges_in_subset[fragment_old_edge_order]
        fragment_atom_subset_mask = atom_mask[fragment_old_atom_idx]

        # Gather edges in pocket and renumber them
        fragment_edge_subset = fragment_old_edge_order[fragment_edge_subset_mask]
        fragment_edge_subset.apply_(lambda x: edge_numbering_dict[x])

        # Gather atoms in pocket and renumber them
        fragment_atom_idx_subset = fragment_old_atom_idx[fragment_atom_subset_mask]
        fragment_atom_idx_subset.apply_(lambda x: atom_numbering_dict[x])

        # Update to new fragment index
        atom_fragment_index_subset = torch.stack(
            [fragment_edge_subset, fragment_atom_idx_subset], dim=0
        )
        complex_graph[
            "atom_bond", "atom"
        ].atom_fragment_index = atom_fragment_index_subset

        # Update receptor edge index
        atom_idx, res_idx = complex_graph[
            "atom", "atom_rec_contact", "receptor"
        ].edge_index
        atoms_subset = atom_idx[atom_mask]
        atom_res_idx_subset = res_idx[atom_mask]

        atoms_subset.apply_(lambda x: atom_numbering_dict[x])
        atom_res_idx_subset.apply_(lambda x: residue_numbering_dict[x])
        complex_graph["atom", "atom_rec_contact", "receptor"].edge_index = torch.stack(
            [atoms_subset, atom_res_idx_subset], dim=0
        )

        # Update edge index and edge mask
        complex_graph["atom", "atom_bond", "atom"].edge_index = atom_edge_index[
            :, edges_in_subset
        ]
        complex_graph["atom", "atom_bond", "atom"].edge_index.apply_(
            lambda x: atom_numbering_dict[x]
        )
        complex_graph["atom", "atom_bond", "atom"].edge_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].edge_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].squeeze_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].squeeze_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].ring_sub_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].ring_sub_mask[edges_in_subset]
        complex_graph["atom", "atom_bond", "atom"].ring_flip_mask = complex_graph[
            "atom", "atom_bond", "atom"
        ].ring_flip_mask[edges_in_subset]

        res_ids_rotatable = complex_graph["atom", "atom_bond", "atom"].res_to_rotate[
            :, 0
        ]
        res_ids_rotatable_subset = res_ids_rotatable[res_mask[res_ids_rotatable]]
        res_ids_rotatable_subset.apply_(lambda x: residue_numbering_dict[x])

        complex_graph["atom", "atom_bond", "atom"].res_to_rotate = torch.stack(
            [res_ids_rotatable_subset, torch.arange(len(res_ids_rotatable_subset))],
            dim=1,
        )

        complex_graph = self.center_complex(complex_graph, pocket_info["pocket_center"]) 

        return complex_graph
    
    def center_complex(self, complex_graph, pocket_center):
        complex_graph['receptor'].pos -= pocket_center

        if self.all_atoms:
            complex_graph['atom'].pos -= pocket_center

        if "orig_holo_pos" in complex_graph["atom"]:
            complex_graph["atom"].orig_holo_pos -= pocket_center
            complex_graph["receptor"].orig_holo_pos = complex_graph["atom"].orig_holo_pos[
                complex_graph["atom"].ca_mask
            ]

        if self.flexible_backbone or self.flexible_sidechains:
            complex_graph['atom'].orig_apo_pos -= pocket_center
            complex_graph['receptor'].pos = complex_graph['atom'].pos[complex_graph['atom'].ca_mask]
            complex_graph['atom'].orig_aligned_apo_pos -= pocket_center

            complex_graph["receptor"].orig_apo_pos = complex_graph["atom"].orig_apo_pos[
                complex_graph["atom"].ca_mask
            ]
            complex_graph["receptor"].orig_aligned_apo_pos = complex_graph["atom"].orig_aligned_apo_pos[
                complex_graph["atom"].ca_mask
            ]

        if "pos_sc_matched" in complex_graph["atom"]:
            complex_graph["atom"]["pos_sc_matched"] -= pocket_center 

        complex_graph['ligand'].pos -= pocket_center
        if "orig_pos" in complex_graph["ligand"]:
            complex_graph['ligand'].orig_pos -= pocket_center

        complex_graph.original_center = pocket_center

        return complex_graph

    def set_direct_nearby_residue_mask(self, complex_graph):
        nearby_atom_mask = complex_graph["atom"].nearby_atoms
        residue_index = complex_graph["atom", "atom_rec_contact", "receptor"].edge_index[1]
        nearby_residues = torch.zeros(
            complex_graph["receptor"].x.shape[0],
            dtype=torch.bool,
            device=nearby_atom_mask.device,
        )
        if nearby_atom_mask.any():
            nearby_residues[residue_index[nearby_atom_mask].unique()] = True
        complex_graph["receptor"].nearby_residues = nearby_residues
        return complex_graph

    def set_radius_based_nearby_atoms(self, complex_graph, ligand_pos):
        nearby_atoms, nearby_residues = compute_nearby_atom_mask(
            atom_pos=complex_graph["atom"].orig_apo_pos,
            lens_receptors=complex_graph["receptor"].lens_receptors,
            ligand_atoms=ligand_pos,
            nearby_residues_atomic_radius=self.nearby_residues_atomic_radius,
            nearby_residues_atomic_min=self.nearby_residues_atomic_min,
        )
        complex_graph["atom"].nearby_atoms = nearby_atoms
        complex_graph["receptor"].nearby_residues = nearby_residues
        return complex_graph

    def select_nearby_atoms(self, complex_graph, ligand_pos=None):
        if self.nearby_residues_selection_mode == "radius_based":
            if ligand_pos is None:
                ligand_pos = complex_graph["ligand"].pos
            complex_graph = self.set_radius_based_nearby_atoms(
                complex_graph,
                ligand_pos=ligand_pos,
            )
        else:
            complex_graph = self.set_direct_nearby_residue_mask(complex_graph)

        nearby_atom_mask = complex_graph["atom"].nearby_atoms
        # This presently captures only sidechains
        atom_edge_index = complex_graph["atom", "atom_bond", "atom"].edge_index
        nearby_atom_mask_edges = (
            nearby_atom_mask[atom_edge_index[0]] & nearby_atom_mask[atom_edge_index[1]]
        )

        # Update rotatable mask to only edges composed of nearby atoms
        complex_graph["atom", "atom_bond", "atom"].edge_mask[
            ~nearby_atom_mask_edges
        ] = False
        return complex_graph

    def len(self):
        if self.limit_complexes is not None:
            return min(int(self.limit_complexes), self.input_df.shape[0])
        return self.input_df.shape[0]


class InferenceDataModule(LightningDataModule):
    def __init__(
        self,
        input_csv,
        featurizer_cfg: FeaturizerConfig,
        limit_complexes: int = None,
        complex_id: str = None,
        esm_embeddings_path: str = None,
        pocket_reduction: bool = False,
        pocket_radius: float = 5.0,
        pocket_buffer: float = 20.0,
        pocket_min_size: int = 1,
        pocket_selection_mode: str = "center_buffer",
        only_nearby_residues_atomic: bool = False,
        nearby_residues_selection_mode: str = "direct",
        nearby_residues_atomic_radius: float = 6.0,
        nearby_residues_atomic_min: int = 8,
        all_atoms: bool = True,
        flexible_backbone: bool = False,
        flexible_sidechains: bool = False,
        load_reference_metrics: bool = True,
    ):
        super().__init__()
        self.input_csv = input_csv
        self.featurizer_cfg = featurizer_cfg
        self.limit_complexes = limit_complexes
        self.complex_id = complex_id

        self.pocket_reduction = pocket_reduction
        self.pocket_buffer = pocket_buffer
        self.pocket_radius = pocket_radius
        self.pocket_min_size = pocket_min_size
        self.pocket_selection_mode = pocket_selection_mode
        self.only_nearby_residues_atomic = only_nearby_residues_atomic
        self.nearby_residues_selection_mode = nearby_residues_selection_mode
        self.nearby_residues_atomic_radius = nearby_residues_atomic_radius
        self.nearby_residues_atomic_min = nearby_residues_atomic_min
        self.esm_embeddings_path = esm_embeddings_path

        self.all_atoms = all_atoms
        self.flexible_backbone = flexible_backbone
        self.flexible_sidechains = flexible_sidechains
        self.load_reference_metrics = load_reference_metrics

    def predict_dataloader(self):
        dataset = PredictionDataset(
            input_csv=self.input_csv,
            featurizer_cfg=self.featurizer_cfg,
            pocket_buffer=self.pocket_buffer,
            pocket_reduction=self.pocket_reduction,
            pocket_radius=self.pocket_radius,
            pocket_min_size=self.pocket_min_size,
            limit_complexes=self.limit_complexes,
            complex_id=self.complex_id,
            pocket_selection_mode=self.pocket_selection_mode,
            only_nearby_residues_atomic=self.only_nearby_residues_atomic,
            nearby_residues_selection_mode=self.nearby_residues_selection_mode,
            nearby_residues_atomic_radius=self.nearby_residues_atomic_radius,
            nearby_residues_atomic_min=self.nearby_residues_atomic_min,
            esm_embeddings_path=self.esm_embeddings_path,
            all_atoms=self.all_atoms,
            flexible_backbone=self.flexible_backbone,
            flexible_sidechains=self.flexible_sidechains,
            load_reference_metrics=self.load_reference_metrics,
        )
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False)
