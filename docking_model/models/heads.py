from __future__ import annotations

import logging
from typing import Callable

import torch
from e3nn import o3
from e3nn.o3 import Linear
from torch import nn
from torch_cluster import radius
from torch_scatter import scatter_mean, scatter_min

from docking_model.geometry.manifolds import so3, torus
from docking_model.models.graph_builders import GraphBuilderStack
from docking_model.models.layers.tensor_product import TensorProductConvLayer
from docking_model.models.tensor_ops import clamped_norm, drop_private_outputs
from docking_model.models.trunk import TrunkOutput


class LigandPoseHeads(nn.Module):
    """Ligand translation, rotation, and torsion heads."""

    def __init__(
        self,
        graph_builder: GraphBuilderStack,
        timestep_emb_func: Callable,
        trunk_out_irreps: str,
        sigma_embed_dim: int,
        ns: int,
        sh_lmax: int,
        distance_embed_dim: int,
        dropout: float,
        activation: nn.Module,
        no_torsion: bool,
        scale_by_sigma: bool,
        odd_parity: bool,
        fixed_center_conv: bool,
        clamped_norm_min: float,
        tor_fourier_enabled: bool,
        tor_fourier_num_freqs: int,
        norm_type=None,
        norm_affine: bool = True,
        use_second_order_repr: bool = False,
        use_oeq_kernels: bool = False,
    ):
        super().__init__()
        self.graph_builder = graph_builder
        self.timestep_emb_func = timestep_emb_func
        self.ns = ns
        self.no_torsion = no_torsion
        self.scale_by_sigma = scale_by_sigma
        self.odd_parity = odd_parity
        self.fixed_center_conv = fixed_center_conv
        self.clamped_norm_min = clamped_norm_min
        self.tor_fourier_enabled = tor_fourier_enabled
        self.tor_fourier_num_freqs = int(tor_fourier_num_freqs)
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)
        self.final_layer_activation = nn.Tanh()

        self.center_edge_embedding = nn.Sequential(
            nn.Linear(distance_embed_dim + sigma_embed_dim, ns),
            activation,
            nn.Dropout(dropout),
            nn.Linear(ns, ns),
        )
        self.final_conv = TensorProductConvLayer(
            in_irreps=trunk_out_irreps,
            sh_irreps=self.sh_irreps,
            out_irreps="2x1o + 2x1e" if not odd_parity else "1x1o + 1x1e",
            n_edge_features=2 * ns,
            residual=False,
            dropout=dropout,
            norm_type=norm_type,
            faster=sh_lmax == 1 and not use_second_order_repr,
            use_oeq_kernels=use_oeq_kernels,
            norm_affine=norm_affine,
        )
        self.tr_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_embed_dim, ns),
            nn.Dropout(dropout),
            activation,
            nn.Linear(ns, 1),
        )
        self.rot_final_layer = nn.Sequential(
            nn.Linear(1 + sigma_embed_dim, ns),
            nn.Dropout(dropout),
            activation,
            nn.Linear(ns, 1),
        )

        if not no_torsion:
            self.final_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns),
                activation,
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )
            self.final_tp_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.tor_bond_conv = TensorProductConvLayer(
                in_irreps=trunk_out_irreps,
                sh_irreps=self.final_tp_tor.irreps_out,
                out_irreps=f"{ns}x0o + {ns}x0e" if not odd_parity else f"{ns}x0o",
                n_edge_features=3 * ns,
                residual=False,
                dropout=dropout,
                norm_affine=norm_affine,
                norm_type=norm_type,
                use_oeq_kernels=use_oeq_kernels,
            )
            hidden_dim = 2 * ns if not odd_parity else ns
            self.tor_final_layer = nn.Sequential(
                nn.Linear(hidden_dim, ns, bias=False),
                self.final_layer_activation,
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False),
            )
            self.tor_fourier_head = (
                nn.Sequential(
                    nn.Linear(hidden_dim, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 2 * self.tor_fourier_num_freqs, bias=False),
                )
                if tor_fourier_enabled
                else None
            )

    def forward(self, trunk: TrunkOutput, data, include_torsion_context: bool = False) -> dict:
        sigma = trunk.graph.sigma
        tr_sigma = sigma["tr_sigma"]
        rot_sigma = sigma["rot_sigma"]
        tor_sigma = sigma["tor_sigma"]

        center_edge_index, center_edge_attr, center_edge_sh = self.graph_builder.build_center_conv_graph(data)
        center_edge_attr = self.center_edge_embedding(center_edge_attr)
        if self.fixed_center_conv:
            center_edge_attr = torch.cat(
                [center_edge_attr, trunk.ligand[center_edge_index[1], : self.ns]], -1
            )
        else:
            center_edge_attr = torch.cat(
                [center_edge_attr, trunk.ligand[center_edge_index[0], : self.ns]], -1
            )

        global_pred = self.final_conv(
            trunk.ligand,
            center_edge_index,
            center_edge_attr,
            center_edge_sh,
            out_nodes=data.num_graphs,
        )
        tr_pred = global_pred[:, :3] + (global_pred[:, 6:9] if not self.odd_parity else 0)
        rot_pred = global_pred[:, 3:6] + (global_pred[:, 9:] if not self.odd_parity else 0)

        graph_sigma_emb = self.timestep_emb_func(data.complex_t["t" if self.graph_builder.asyncronous_noise_schedule else "tr"])
        tr_norm = clamped_norm(tr_pred, dim=1, min=self.clamped_norm_min).unsqueeze(1)
        tr_pred = tr_pred / tr_norm * self.tr_final_layer(torch.cat([tr_norm, graph_sigma_emb], dim=1))
        rot_norm = clamped_norm(rot_pred, dim=1, min=self.clamped_norm_min).unsqueeze(1)
        rot_pred = rot_pred / rot_norm * self.rot_final_layer(torch.cat([rot_norm, graph_sigma_emb], dim=1))

        if self.scale_by_sigma:
            tr_pred = tr_pred / tr_sigma.unsqueeze(1)
            rot_pred = rot_pred * so3.score_norm(rot_sigma.cpu()).unsqueeze(1).to(data["ligand"].x.device)

        outputs = {"tr_pred": tr_pred, "rot_pred": rot_pred}
        outputs.update(self.torsion_predictions(trunk, data, tor_sigma, tr_pred.device, include_torsion_context))
        return outputs

    def torsion_predictions(self, trunk: TrunkOutput, data, tor_sigma, device, include_context: bool) -> dict:
        if self.no_torsion or data["ligand"].edge_mask.sum() == 0:
            outputs = {"tor_pred": torch.empty(0, device=device)}
            if self.tor_fourier_enabled:
                outputs.update(
                    {
                        "tor_fourier_a": torch.empty(
                            (0, self.tor_fourier_num_freqs), device=device, dtype=trunk.ligand.dtype
                        ),
                        "tor_fourier_b": torch.empty(
                            (0, self.tor_fourier_num_freqs), device=device, dtype=trunk.ligand.dtype
                        ),
                    }
                )
            if include_context:
                outputs.update(
                    {
                        "_tor_hidden": None,
                        "_tor_theta": None,
                        "_tor_graph_index": None,
                        "_tor_bonds": None,
                        "_tor_base_term": outputs["tor_pred"].clone(),
                    }
                )
            return outputs

        bonds, edge_index, edge_attr, edge_sh, edge_weight = self.graph_builder.build_bond_conv_graph(
            data, self.final_edge_embedding
        )
        bond_vec = data["ligand"].pos[bonds[1]] - data["ligand"].pos[bonds[0]]
        bond_attr = trunk.ligand[bonds[0]] + trunk.ligand[bonds[1]]
        bond_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization="component")
        edge_sh = self.final_tp_tor(edge_sh, bond_sh[edge_index[0]])
        edge_attr = torch.cat(
            [edge_attr, trunk.ligand[edge_index[1], : self.ns], bond_attr[edge_index[0], : self.ns]],
            -1,
        )
        hidden = self.tor_bond_conv(
            trunk.ligand,
            edge_index,
            edge_attr,
            edge_sh,
            out_nodes=data["ligand"].edge_mask.sum(),
            reduce="mean",
            edge_weight=edge_weight,
        )
        edge_sigma = tor_sigma[data["ligand"].batch][
            data["ligand", "lig_bond", "ligand"].edge_index[0]
        ][data["ligand"].edge_mask]
        tor_theta = None
        tor_graph_index = None
        if self.tor_fourier_enabled:
            tor_theta = getattr(data["ligand"], "tor_theta", None)
            if tor_theta is None:
                raise ValueError("tor_fourier_enabled=True requires data['ligand'].tor_theta.")
            tor_theta = tor_theta.to(hidden.device).view(-1)
            if tor_theta.numel() != hidden.size(0):
                raise ValueError(f"tor_theta size mismatch: expected {hidden.size(0)}, got {tor_theta.numel()}.")
            tor_graph_index = data["ligand"].batch[
                data["ligand", "lig_bond", "ligand"].edge_index[0]
            ][data["ligand"].edge_mask]
            fourier = self.tor_fourier_head(hidden).view(-1, self.tor_fourier_num_freqs, 2)
            fourier_a, fourier_b = fourier[..., 0], fourier[..., 1]
            modes = torch.arange(1, self.tor_fourier_num_freqs + 1, device=hidden.device, dtype=hidden.dtype).view(1, -1)
            phase = tor_theta.unsqueeze(-1) * modes
            sigma_gate = torch.exp(-0.5 * (modes**2) * (edge_sigma.to(hidden.dtype).unsqueeze(-1) ** 2))
            tor_pred = (
                modes * sigma_gate * (-fourier_a * torch.sin(phase) + fourier_b * torch.cos(phase))
            ).sum(dim=-1)
            outputs = {"tor_fourier_a": fourier_a, "tor_fourier_b": fourier_b}
        else:
            tor_pred = self.tor_final_layer(hidden).squeeze(1)
            if self.scale_by_sigma:
                tor_pred = tor_pred * torch.sqrt(
                    torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float().to(data["ligand"].x.device)
                )
            outputs = {}
        outputs["tor_pred"] = tor_pred
        if include_context:
            outputs.update(
                {
                    "_tor_hidden": hidden,
                    "_tor_theta": tor_theta,
                    "_tor_graph_index": tor_graph_index,
                    "_tor_bonds": bonds,
                    "_tor_base_term": tor_pred.clone(),
                }
            )
        return outputs


class ProteinFlexibilityHeads(nn.Module):
    """Backbone and sidechain flexibility heads."""

    def __init__(
        self,
        graph_builder: GraphBuilderStack,
        timestep_emb_func: Callable,
        trunk_out_irreps: str,
        sigma_embed_dim: int,
        ns: int,
        sh_lmax: int,
        distance_embed_dim: int,
        dropout: float,
        activation: nn.Module,
        flexible_sidechains: bool,
        flexible_backbone: bool,
        sidechain_tor_bridge: bool,
        scale_by_sigma: bool,
        odd_parity: bool,
        sc_tor_fourier_enabled: bool,
        sc_tor_fourier_num_freqs: int,
        sc_tor_fourier_sigma_conditioning: bool,
        sc_tor_fourier_gate_type: str,
        sc_tor_fourier_gate_hidden: int,
        sc_tor_fourier_poly_degree: int,
        sc_tor_fourier_joint_time: bool,
        tor_sc_coupling_enabled: bool,
        tor_fourier_enabled: bool = False,
        tor_fourier_num_freqs: int = 8,
        tor_sc_coupling_unary_enabled: bool = True,
        tor_sc_coupling_pairwise_enabled: bool = True,
        tor_sc_coupling_radius: float = 6.0,
        tor_sc_coupling_max_neighbors: int = 64,
        norm_type=None,
        norm_affine: bool = True,
        use_oeq_kernels: bool = False,
    ):
        super().__init__()
        self.graph_builder = graph_builder
        self.timestep_emb_func = timestep_emb_func
        self.ns = ns
        self.flexible_sidechains = flexible_sidechains
        self.flexible_backbone = flexible_backbone
        self.sidechain_tor_bridge = sidechain_tor_bridge
        self.scale_by_sigma = scale_by_sigma
        self.odd_parity = odd_parity
        self.sc_tor_fourier_enabled = sc_tor_fourier_enabled
        self.sc_tor_fourier_num_freqs = int(sc_tor_fourier_num_freqs)
        self.sc_tor_fourier_sigma_conditioning = sc_tor_fourier_sigma_conditioning
        self.sc_tor_fourier_gate_type = str(sc_tor_fourier_gate_type).lower()
        self.sc_tor_fourier_poly_degree = int(sc_tor_fourier_poly_degree)
        self.tor_sc_coupling_enabled = tor_sc_coupling_enabled
        self.tor_fourier_enabled = tor_fourier_enabled
        self.tor_fourier_num_freqs = int(tor_fourier_num_freqs)
        self.tor_sc_coupling_unary_enabled = tor_sc_coupling_unary_enabled
        self.tor_sc_coupling_pairwise_enabled = tor_sc_coupling_pairwise_enabled
        self.tor_sc_coupling_radius = float(tor_sc_coupling_radius)
        self.tor_sc_coupling_max_neighbors = int(tor_sc_coupling_max_neighbors)
        self.final_layer_activation = nn.Tanh()
        self.sh_irreps = o3.Irreps.spherical_harmonics(lmax=sh_lmax)

        hidden_dim = 2 * ns if not odd_parity else ns
        lig_tor_hidden_dim = hidden_dim
        if flexible_sidechains:
            self.sidechain_final_edge_embedding = nn.Sequential(
                nn.Linear(distance_embed_dim, ns),
                activation,
                nn.Dropout(dropout),
                nn.Linear(ns, ns),
            )
            self.final_tp_sc_tor = o3.FullTensorProduct(self.sh_irreps, "2e")
            self.sc_tor_bond_conv = TensorProductConvLayer(
                in_irreps=trunk_out_irreps,
                sh_irreps=self.final_tp_sc_tor.irreps_out,
                out_irreps=f"{ns}x0o + {ns}x0e" if not odd_parity else f"{ns}x0o",
                n_edge_features=3 * ns,
                residual=False,
                dropout=dropout,
                norm_type=norm_type,
                norm_affine=norm_affine,
                use_oeq_kernels=use_oeq_kernels,
            )
            self.sc_tor_final_layer = nn.Sequential(
                nn.Linear(hidden_dim, ns, bias=False),
                self.final_layer_activation,
                nn.Dropout(dropout),
                nn.Linear(ns, 1, bias=False),
            )
            self.sc_tor_fourier_head = (
                nn.Sequential(
                    nn.Linear(hidden_dim, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 2 * self.sc_tor_fourier_num_freqs, bias=False),
                )
                if sc_tor_fourier_enabled
                else None
            )
            self.sc_tor_fourier_gate_mlp = (
                nn.Sequential(
                    nn.Linear(1, sc_tor_fourier_gate_hidden),
                    activation,
                    nn.Dropout(dropout),
                    nn.Linear(sc_tor_fourier_gate_hidden, self.sc_tor_fourier_num_freqs),
                )
                if self.sc_tor_fourier_gate_type == "mlp"
                else None
            )
            self.sc_tor_fourier_bell_alpha_raw = (
                nn.Parameter(torch.full((self.sc_tor_fourier_num_freqs,), -8.0))
                if self.sc_tor_fourier_gate_type == "bell"
                else None
            )
            self.sc_tor_fourier_bell_beta_raw = (
                nn.Parameter(torch.full((self.sc_tor_fourier_num_freqs,), -8.0))
                if self.sc_tor_fourier_gate_type == "bell"
                else None
            )
            self.sc_tor_fourier_bell_log_amp = (
                nn.Parameter(torch.zeros(self.sc_tor_fourier_num_freqs))
                if self.sc_tor_fourier_gate_type == "bell"
                else None
            )
            self.sc_tor_fourier_poly_coeff = (
                nn.Parameter(torch.zeros(self.sc_tor_fourier_num_freqs, self.sc_tor_fourier_poly_degree))
                if self.sc_tor_fourier_gate_type == "poly"
                else None
            )
            self.sc_tor_fourier_time_head = (
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 2 * self.sc_tor_fourier_num_freqs, bias=False),
                )
                if sc_tor_fourier_joint_time
                else None
            )
            self.sc_tor_fourier_sigma_head = (
                nn.Sequential(
                    nn.Linear(hidden_dim + 1, ns, bias=False),
                    self.final_layer_activation,
                    nn.Dropout(dropout),
                    nn.Linear(ns, 2 * self.sc_tor_fourier_num_freqs, bias=False),
                )
                if sc_tor_fourier_sigma_conditioning and not sidechain_tor_bridge
                else None
            )
            if tor_sc_coupling_enabled:
                self.tor_coupling_sc_unary_head = (
                    nn.Sequential(
                        nn.Linear(lig_tor_hidden_dim, ns, bias=False),
                        self.final_layer_activation,
                        nn.Dropout(dropout),
                        nn.Linear(ns, 2 * self.sc_tor_fourier_num_freqs, bias=False),
                    )
                    if tor_sc_coupling_unary_enabled
                    else None
                )
                self.sc_coupling_lig_unary_head = (
                    nn.Sequential(
                        nn.Linear(hidden_dim, ns, bias=False),
                        self.final_layer_activation,
                        nn.Dropout(dropout),
                        nn.Linear(ns, 2 * self.tor_fourier_num_freqs, bias=False),
                    )
                    if tor_sc_coupling_unary_enabled
                    else None
                )
                self.tor_coupling_pair_head = (
                    nn.Sequential(
                        nn.Linear(lig_tor_hidden_dim, ns, bias=False),
                        self.final_layer_activation,
                        nn.Dropout(dropout),
                        nn.Linear(
                            ns,
                            2 * self.tor_fourier_num_freqs * self.sc_tor_fourier_num_freqs,
                            bias=False,
                        ),
                    )
                    if tor_sc_coupling_pairwise_enabled
                    else None
                )
                self.sc_coupling_pair_head = (
                    nn.Sequential(
                        nn.Linear(hidden_dim, ns, bias=False),
                        self.final_layer_activation,
                        nn.Dropout(dropout),
                        nn.Linear(
                            ns,
                            2 * self.tor_fourier_num_freqs * self.sc_tor_fourier_num_freqs,
                            bias=False,
                        ),
                    )
                    if tor_sc_coupling_pairwise_enabled
                    else None
                )
            else:
                self.tor_coupling_sc_unary_head = None
                self.sc_coupling_lig_unary_head = None
                self.tor_coupling_pair_head = None
                self.sc_coupling_pair_head = None

        if flexible_backbone:
            self.bb_o3_linear = Linear(
                irreps_in=trunk_out_irreps,
                irreps_out="2x1o + 2x1e",
                internal_weights=True,
                shared_weights=True,
            )
            self.bb_tr_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                activation,
                nn.Linear(ns, 1),
            )
            self.bb_rot_final_layer = nn.Sequential(
                nn.Linear(1 + sigma_embed_dim, ns),
                nn.Dropout(dropout),
                activation,
                nn.Linear(ns, 1),
            )

    def forward(self, trunk: TrunkOutput, data, ligand_outputs: dict | None = None) -> dict:
        outputs = self.sidechain_outputs(trunk, data, include_context=self.tor_sc_coupling_enabled)
        if self.tor_sc_coupling_enabled and ligand_outputs is not None:
            outputs.update(self.tor_sc_coupling_outputs(trunk, data, ligand_outputs, outputs))
        outputs.update(self.backbone_outputs(trunk, data))
        drop_private_outputs(outputs)
        return outputs

    def sidechain_outputs(self, trunk: TrunkOutput, data, include_context: bool) -> dict:
        device = trunk.ligand.device
        dtype = trunk.ligand.dtype
        if not self.flexible_sidechains:
            return {"sc_tor_pred": torch.empty(0, device=device, dtype=dtype)}

        num_flexible_bonds = int(data["atom", "atom_bond", "atom"].edge_mask.sum().item())
        if num_flexible_bonds == 0:
            outputs = {"sc_tor_pred": torch.empty(0, device=device, dtype=dtype)}
            if self.sc_tor_fourier_enabled:
                outputs.update(
                    {
                        "sc_tor_fourier_a": torch.empty((0, self.sc_tor_fourier_num_freqs), device=device, dtype=dtype),
                        "sc_tor_fourier_b": torch.empty((0, self.sc_tor_fourier_num_freqs), device=device, dtype=dtype),
                        "sc_tor_fourier_gate": torch.empty((0, self.sc_tor_fourier_num_freqs), device=device, dtype=dtype),
                    }
                )
            if include_context:
                outputs.update(
                    {
                        "_sc_tor_hidden": None,
                        "_sc_tor_theta": None,
                        "_sc_tor_graph_index": None,
                        "_sc_tor_bonds": None,
                        "_sc_tor_base_term": outputs["sc_tor_pred"].clone(),
                    }
                )
            return outputs

        try:
            bonds, edge_index, edge_attr, edge_sh, edge_weight = self.graph_builder.build_sidechain_conv_graph(
                data, self.sidechain_final_edge_embedding
            )
            bond_vec = data["atom"].pos[bonds[1]] - data["atom"].pos[bonds[0]]
            bond_attr = trunk.atom[bonds[0]] + trunk.atom[bonds[1]]
            bond_sh = o3.spherical_harmonics("2e", bond_vec, normalize=True, normalization="component")
            edge_sh = self.final_tp_sc_tor(edge_sh, bond_sh[edge_index[0]])
            edge_attr = torch.cat(
                [edge_attr, trunk.atom[edge_index[1], : self.ns], bond_attr[edge_index[0], : self.ns]],
                -1,
            )
            hidden = self.sc_tor_bond_conv(
                trunk.atom,
                edge_index,
                edge_attr,
                edge_sh,
                out_nodes=num_flexible_bonds,
                reduce="mean",
                edge_weight=edge_weight,
            )
            if self.sc_tor_fourier_enabled:
                pred, extra, theta, graph_index = self.sidechain_fourier_pred(hidden, bonds, trunk, data)
                outputs = {"sc_tor_pred": pred, **extra}
                if include_context:
                    outputs.update(
                        {
                            "_sc_tor_hidden": hidden,
                            "_sc_tor_theta": theta,
                            "_sc_tor_graph_index": graph_index,
                            "_sc_tor_bonds": bonds,
                            "_sc_tor_base_term": pred.clone(),
                        }
                    )
                return outputs

            pred = self.sc_tor_final_layer(hidden).squeeze(1)
            if self.scale_by_sigma and not self.sidechain_tor_bridge:
                graph_index = self.compute_sc_tor_graph_index(data, bonds)
                edge_sigma = trunk.graph.sigma["sc_tor_sigma"][graph_index]
                norm = torch.sqrt(
                    torch.tensor(torus.score_norm(edge_sigma.cpu().numpy())).float().to(data["atom"].x.device)
                )
                pred = pred * norm
            outputs = {"sc_tor_pred": pred}
            if include_context:
                outputs.update(
                    {
                        "_sc_tor_hidden": hidden,
                        "_sc_tor_theta": None,
                        "_sc_tor_graph_index": self.compute_sc_tor_graph_index(data, bonds),
                        "_sc_tor_bonds": bonds,
                        "_sc_tor_base_term": pred.clone(),
                    }
                )
            return outputs
        except Exception:
            logging.exception("Exception while predicting flexible sidechains for %s", data["name"])
            raise

    def sidechain_fourier_pred(self, hidden, bonds, trunk: TrunkOutput, data):
        theta = getattr(data, "sidechain_tor_theta", None)
        if theta is None:
            raise ValueError("sc_tor_fourier_enabled=True requires data.sidechain_tor_theta.")
        theta = theta.to(hidden.device).view(-1)
        if theta.numel() != hidden.size(0):
            raise ValueError(f"sidechain_tor_theta size mismatch: expected {hidden.size(0)}, got {theta.numel()}.")
        fourier = self.sc_tor_fourier_head(hidden).view(-1, self.sc_tor_fourier_num_freqs, 2)
        fourier_a, fourier_b = fourier[..., 0], fourier[..., 1]
        graph_index = self.compute_sc_tor_graph_index(data, bonds)
        if graph_index.numel() != hidden.size(0):
            raise ValueError(f"sidechain graph-index size mismatch: expected {hidden.size(0)}, got {graph_index.numel()}.")
        modes = torch.arange(1, self.sc_tor_fourier_num_freqs + 1, device=hidden.device, dtype=hidden.dtype).view(1, -1)
        phase = theta.unsqueeze(-1) * modes
        if self.sidechain_tor_bridge:
            t_edge = trunk.graph.t["sc_tor"][graph_index].to(hidden.device)
            gate = self.compute_sc_tor_fourier_gate(t_edge, hidden.dtype)
            if self.sc_tor_fourier_time_head is not None:
                joint_input = torch.cat([hidden, t_edge.to(hidden.dtype).unsqueeze(-1)], dim=-1)
                delta = self.sc_tor_fourier_time_head(joint_input).view(-1, self.sc_tor_fourier_num_freqs, 2)
                fourier_a = fourier_a + delta[..., 0]
                fourier_b = fourier_b + delta[..., 1]
            pred = (gate * (fourier_a * torch.sin(phase) + fourier_b * torch.cos(phase))).sum(dim=-1)
        else:
            edge_sigma = trunk.graph.sigma["sc_tor_sigma"][graph_index].to(hidden.dtype)
            if self.sc_tor_fourier_sigma_head is not None:
                sigma_input = torch.cat([hidden, edge_sigma.clamp_min(1e-8).log().unsqueeze(-1)], dim=-1)
                delta = self.sc_tor_fourier_sigma_head(sigma_input).view(-1, self.sc_tor_fourier_num_freqs, 2)
                fourier_a = fourier_a + delta[..., 0]
                fourier_b = fourier_b + delta[..., 1]
            gate = torch.exp(-0.5 * (modes**2) * (edge_sigma.unsqueeze(-1) ** 2))
            pred = (modes * gate * (-fourier_a * torch.sin(phase) + fourier_b * torch.cos(phase))).sum(dim=-1)
        return pred, {
            "sc_tor_fourier_a": fourier_a,
            "sc_tor_fourier_b": fourier_b,
            "sc_tor_fourier_gate": gate,
        }, theta, graph_index

    def tor_sc_coupling_outputs(self, trunk: TrunkOutput, data, ligand_outputs: dict, sc_outputs: dict) -> dict:
        tor_pred = ligand_outputs.get("tor_pred")
        sc_tor_pred = sc_outputs.get("sc_tor_pred")
        tor_hidden = ligand_outputs.get("_tor_hidden")
        tor_theta = ligand_outputs.get("_tor_theta")
        tor_graph_index = ligand_outputs.get("_tor_graph_index")
        tor_bonds = ligand_outputs.get("_tor_bonds")
        tor_base_term = ligand_outputs.get("_tor_base_term")
        sc_tor_hidden = sc_outputs.get("_sc_tor_hidden")
        sc_tor_theta = sc_outputs.get("_sc_tor_theta")
        sc_graph_index = sc_outputs.get("_sc_tor_graph_index")
        sc_tor_bonds = sc_outputs.get("_sc_tor_bonds")
        sc_tor_base_term = sc_outputs.get("_sc_tor_base_term")

        empty = self.empty_coupling_outputs(trunk.ligand.device, trunk.ligand.dtype, tor_pred, sc_tor_pred)
        valid = (
            torch.is_tensor(tor_pred)
            and torch.is_tensor(sc_tor_pred)
            and tor_hidden is not None
            and sc_tor_hidden is not None
            and tor_theta is not None
            and sc_tor_theta is not None
            and tor_theta.numel() > 0
            and sc_tor_theta.numel() > 0
            and tor_graph_index is not None
            and sc_graph_index is not None
            and tor_bonds is not None
            and sc_tor_bonds is not None
        )
        if not valid:
            return {"tor_pred": tor_pred, **empty}

        lig_modes = torch.arange(1, self.tor_fourier_num_freqs + 1, device=tor_hidden.device, dtype=tor_hidden.dtype).view(1, -1)
        sc_modes = torch.arange(1, self.sc_tor_fourier_num_freqs + 1, device=sc_tor_hidden.device, dtype=sc_tor_hidden.dtype).view(1, -1)
        lig_phase = tor_theta.unsqueeze(-1) * lig_modes
        lig_cos = torch.cos(lig_phase)
        lig_sin = torch.sin(lig_phase)
        sc_phase = sc_tor_theta.unsqueeze(-1) * sc_modes
        sc_cos = torch.cos(sc_phase)
        sc_sin = torch.sin(sc_phase)

        tor_centers = 0.5 * (data["ligand"].pos[tor_bonds[0]] + data["ligand"].pos[tor_bonds[1]])
        sc_centers = 0.5 * (data["atom"].pos[sc_tor_bonds[0]] + data["atom"].pos[sc_tor_bonds[1]])
        local_edge_index = radius(
            sc_centers,
            tor_centers,
            self.tor_sc_coupling_radius,
            sc_graph_index,
            tor_graph_index,
            max_num_neighbors=self.tor_sc_coupling_max_neighbors,
        )
        if local_edge_index.numel() > 0:
            lig_local_idx = local_edge_index[0].long()
            sc_local_idx = local_edge_index[1].long()
            sc_cos_for_lig = scatter_mean(sc_cos[sc_local_idx], lig_local_idx, dim=0, dim_size=tor_theta.numel())
            sc_sin_for_lig = scatter_mean(sc_sin[sc_local_idx], lig_local_idx, dim=0, dim_size=tor_theta.numel())
            lig_cos_for_sc = scatter_mean(lig_cos[lig_local_idx], sc_local_idx, dim=0, dim_size=sc_tor_theta.numel())
            lig_sin_for_sc = scatter_mean(lig_sin[lig_local_idx], sc_local_idx, dim=0, dim_size=sc_tor_theta.numel())
            local_neighbors_lig = torch.bincount(lig_local_idx, minlength=tor_theta.numel()).to(tor_hidden.dtype)
            local_neighbors_sc = torch.bincount(sc_local_idx, minlength=sc_tor_theta.numel()).to(sc_tor_hidden.dtype)
        else:
            sc_cos_for_lig = tor_hidden.new_zeros((tor_theta.numel(), self.sc_tor_fourier_num_freqs))
            sc_sin_for_lig = tor_hidden.new_zeros((tor_theta.numel(), self.sc_tor_fourier_num_freqs))
            lig_cos_for_sc = sc_tor_hidden.new_zeros((sc_tor_theta.numel(), self.tor_fourier_num_freqs))
            lig_sin_for_sc = sc_tor_hidden.new_zeros((sc_tor_theta.numel(), self.tor_fourier_num_freqs))
            local_neighbors_lig = tor_hidden.new_zeros(tor_theta.shape)
            local_neighbors_sc = sc_tor_hidden.new_zeros(sc_tor_theta.shape)

        tor_unary_term = tor_pred.new_zeros(tor_pred.shape)
        sc_unary_term = sc_tor_pred.new_zeros(sc_tor_pred.shape)
        tor_pair_term = tor_pred.new_zeros(tor_pred.shape)
        sc_pair_term = sc_tor_pred.new_zeros(sc_tor_pred.shape)
        tor_unary = None
        sc_unary = None
        tor_pair = None
        sc_pair = None

        if self.tor_sc_coupling_unary_enabled:
            tor_unary = self.tor_coupling_sc_unary_head(tor_hidden).view(-1, self.sc_tor_fourier_num_freqs, 2)
            sc_unary = self.sc_coupling_lig_unary_head(sc_tor_hidden).view(-1, self.tor_fourier_num_freqs, 2)
            tor_unary_term = (tor_unary[..., 0] * sc_cos_for_lig + tor_unary[..., 1] * sc_sin_for_lig).sum(dim=-1)
            sc_unary_term = (sc_unary[..., 0] * lig_cos_for_sc + sc_unary[..., 1] * lig_sin_for_sc).sum(dim=-1)
            tor_pred = tor_pred + tor_unary_term
            sc_tor_pred = sc_tor_pred + sc_unary_term

        if self.tor_sc_coupling_pairwise_enabled:
            tor_pair = self.tor_coupling_pair_head(tor_hidden).view(
                -1, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2
            )
            sc_pair = self.sc_coupling_pair_head(sc_tor_hidden).view(
                -1, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2
            )
            cos_delta_lig = lig_cos.unsqueeze(-1) * sc_cos_for_lig.unsqueeze(1) + lig_sin.unsqueeze(-1) * sc_sin_for_lig.unsqueeze(1)
            sin_delta_lig = lig_sin.unsqueeze(-1) * sc_cos_for_lig.unsqueeze(1) - lig_cos.unsqueeze(-1) * sc_sin_for_lig.unsqueeze(1)
            cos_delta_sc = lig_cos_for_sc.unsqueeze(-1) * sc_cos.unsqueeze(1) + lig_sin_for_sc.unsqueeze(-1) * sc_sin.unsqueeze(1)
            sin_delta_sc = lig_sin_for_sc.unsqueeze(-1) * sc_cos.unsqueeze(1) - lig_cos_for_sc.unsqueeze(-1) * sc_sin.unsqueeze(1)
            tor_pair_term = (tor_pair[..., 0] * cos_delta_lig + tor_pair[..., 1] * sin_delta_lig).sum(dim=(-1, -2))
            sc_pair_term = (sc_pair[..., 0] * cos_delta_sc + sc_pair[..., 1] * sin_delta_sc).sum(dim=(-1, -2))
            tor_pred = tor_pred + tor_pair_term
            sc_tor_pred = sc_tor_pred + sc_pair_term

        tor_base_term = tor_pred if tor_base_term is None else tor_base_term
        sc_tor_base_term = sc_tor_pred if sc_tor_base_term is None else sc_tor_base_term
        coupling_outputs = self.coupling_diagnostics(
            tor_pred,
            sc_tor_pred,
            tor_base_term,
            tor_unary_term,
            tor_pair_term,
            sc_tor_base_term,
            sc_unary_term,
            sc_pair_term,
            local_neighbors_lig,
            local_neighbors_sc,
        )
        coupling_outputs.update(
            {
                "tor_pred": tor_pred,
                "sc_tor_pred": sc_tor_pred,
                "tor_sc_coupling_unary_coeff_lig": self.or_empty(
                    tor_unary, (0, self.sc_tor_fourier_num_freqs, 2), tor_pred
                ),
                "tor_sc_coupling_unary_coeff_sc": self.or_empty(
                    sc_unary, (0, self.tor_fourier_num_freqs, 2), sc_tor_pred
                ),
                "tor_sc_coupling_pair_coeff_lig": self.or_empty(
                    tor_pair,
                    (0, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2),
                    tor_pred,
                ),
                "tor_sc_coupling_pair_coeff_sc": self.or_empty(
                    sc_pair,
                    (0, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2),
                    sc_tor_pred,
                ),
            }
        )
        return coupling_outputs

    def empty_coupling_outputs(self, device, dtype, tor_pred, sc_tor_pred) -> dict:
        reference = tor_pred if torch.is_tensor(tor_pred) else torch.empty(0, device=device, dtype=dtype)
        sc_reference = sc_tor_pred if torch.is_tensor(sc_tor_pred) else torch.empty(0, device=device, dtype=dtype)
        diagnostics = self.coupling_diagnostics(
            reference,
            sc_reference,
            reference,
            reference.new_zeros(reference.shape),
            reference.new_zeros(reference.shape),
            sc_reference,
            sc_reference.new_zeros(sc_reference.shape),
            sc_reference.new_zeros(sc_reference.shape),
            reference.new_zeros(reference.shape),
            sc_reference.new_zeros(sc_reference.shape),
        )
        diagnostics.update(
            {
                "tor_sc_coupling_unary_coeff_lig": torch.empty(
                    (0, self.sc_tor_fourier_num_freqs, 2), device=device, dtype=dtype
                ),
                "tor_sc_coupling_unary_coeff_sc": torch.empty(
                    (0, self.tor_fourier_num_freqs, 2), device=device, dtype=dtype
                ),
                "tor_sc_coupling_pair_coeff_lig": torch.empty(
                    (0, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2),
                    device=device,
                    dtype=dtype,
                ),
                "tor_sc_coupling_pair_coeff_sc": torch.empty(
                    (0, self.tor_fourier_num_freqs, self.sc_tor_fourier_num_freqs, 2),
                    device=device,
                    dtype=dtype,
                ),
            }
        )
        return diagnostics

    def coupling_diagnostics(
        self,
        tor_pred,
        sc_tor_pred,
        tor_base_term,
        tor_unary_term,
        tor_pair_term,
        sc_tor_base_term,
        sc_tor_unary_term,
        sc_tor_pair_term,
        local_neighbors_lig,
        local_neighbors_sc,
    ) -> dict:
        tor_pred_base_rms = safe_rms(tor_base_term, tor_pred)
        tor_pred_unary_rms = safe_rms(tor_unary_term, tor_pred)
        tor_pred_pair_rms = safe_rms(tor_pair_term, tor_pred)
        sc_tor_pred_base_rms = safe_rms(sc_tor_base_term, sc_tor_pred)
        sc_tor_pred_unary_rms = safe_rms(sc_tor_unary_term, sc_tor_pred)
        sc_tor_pred_pair_rms = safe_rms(sc_tor_pair_term, sc_tor_pred)
        if local_neighbors_lig.numel() > 0:
            local_neighbors_lig_mean = local_neighbors_lig.mean()
            local_coverage_lig = (local_neighbors_lig > 0).to(local_neighbors_lig.dtype).mean()
        else:
            local_neighbors_lig_mean = tor_pred.new_tensor(0.0)
            local_coverage_lig = tor_pred.new_tensor(0.0)
        if local_neighbors_sc.numel() > 0:
            local_neighbors_sc_mean = local_neighbors_sc.mean()
            local_coverage_sc = (local_neighbors_sc > 0).to(local_neighbors_sc.dtype).mean()
        else:
            local_neighbors_sc_mean = sc_tor_pred.new_tensor(0.0)
            local_coverage_sc = sc_tor_pred.new_tensor(0.0)
        return {
            "tor_pred_base_rms": tor_pred_base_rms,
            "tor_pred_unary_rms": tor_pred_unary_rms,
            "tor_pred_pair_rms": tor_pred_pair_rms,
            "tor_pred_unary_rel": tor_pred_unary_rms / (tor_pred_base_rms + 1e-8),
            "tor_pred_pair_rel": tor_pred_pair_rms / (tor_pred_base_rms + 1e-8),
            "sc_tor_pred_base_rms": sc_tor_pred_base_rms,
            "sc_tor_pred_unary_rms": sc_tor_pred_unary_rms,
            "sc_tor_pred_pair_rms": sc_tor_pred_pair_rms,
            "sc_tor_pred_unary_rel": sc_tor_pred_unary_rms / (sc_tor_pred_base_rms + 1e-8),
            "sc_tor_pred_pair_rel": sc_tor_pred_pair_rms / (sc_tor_pred_base_rms + 1e-8),
            "tor_sc_local_neighbors_lig_mean": local_neighbors_lig_mean,
            "tor_sc_local_neighbors_sc_mean": local_neighbors_sc_mean,
            "tor_sc_local_coverage_lig": local_coverage_lig,
            "tor_sc_local_coverage_sc": local_coverage_sc,
        }

    @staticmethod
    def or_empty(value, shape, reference: torch.Tensor) -> torch.Tensor:
        if value is not None:
            return value
        return torch.empty(shape, device=reference.device, dtype=reference.dtype)

    def backbone_outputs(self, trunk: TrunkOutput, data) -> dict:
        if not self.flexible_backbone:
            empty = torch.empty(0, device=trunk.ligand.device)
            return {"bb_tr_pred": empty, "bb_rot_pred": empty}
        pred = self.bb_o3_linear(trunk.receptor)
        bb_tr_pred = pred[:, :3] + pred[:, 6:9]
        bb_rot_pred = pred[:, 3:6] + pred[:, 9:]
        bb_tr_sigma_emb = self.timestep_emb_func(data["receptor"].node_t["bb_tr"])
        bb_rot_sigma_emb = self.timestep_emb_func(data["receptor"].node_t["bb_rot"])
        bb_tr_norm = clamped_norm(bb_tr_pred, dim=1, min=1e-6).unsqueeze(1)
        bb_tr_pred = bb_tr_pred / bb_tr_norm * self.bb_tr_final_layer(torch.cat([bb_tr_norm, bb_tr_sigma_emb], dim=1))
        bb_rot_norm = clamped_norm(bb_rot_pred, dim=1, min=1e-6).unsqueeze(1)
        bb_rot_pred = bb_rot_pred / bb_rot_norm * self.bb_rot_final_layer(torch.cat([bb_rot_norm, bb_rot_sigma_emb], dim=1))
        return {"bb_tr_pred": bb_tr_pred, "bb_rot_pred": bb_rot_pred}

    def compute_sc_tor_graph_index(self, data, bonds):
        edge_store = data["atom", "atom_bond", "atom"]
        source_idx = edge_store.edge_index[0, edge_store.edge_mask]
        if source_idx.numel() == bonds.size(1):
            return data["atom"].orig_batch[source_idx]
        return data["atom"].orig_batch[bonds[0]]

    def compute_sc_tor_fourier_gate(self, t_edge: torch.Tensor, dtype: torch.dtype):
        t = t_edge.to(dtype).view(-1, 1).clamp(0.0, 1.0)
        if self.sc_tor_fourier_gate_type == "none":
            return t.new_ones((t.numel(), self.sc_tor_fourier_num_freqs))
        if self.sc_tor_fourier_gate_type == "mlp":
            return 1.0 + 0.5 * torch.tanh(self.sc_tor_fourier_gate_mlp(t))
        if self.sc_tor_fourier_gate_type == "bell":
            eps = 1e-6
            alpha = torch.nn.functional.softplus(self.sc_tor_fourier_bell_alpha_raw).view(1, -1)
            beta = torch.nn.functional.softplus(self.sc_tor_fourier_bell_beta_raw).view(1, -1)
            amp = self.sc_tor_fourier_bell_log_amp.exp().view(1, -1)
            return (amp * ((t + eps) ** alpha) * ((1.0 - t + eps) ** beta)).clamp_min(1e-4)
        powers = torch.cat([t**k for k in range(1, self.sc_tor_fourier_poly_degree + 1)], dim=1)
        return 1.0 + 0.5 * torch.tanh(powers @ self.sc_tor_fourier_poly_coeff.t())


class ConfidenceAffinityHeads(nn.Module):
    """Confidence and affinity heads."""

    def __init__(
        self,
        ns: int,
        num_conv_layers: int,
        num_confidence_outputs: int,
        activation: nn.Module,
        confidence_dropout: float = 0.0,
        atom_lig_confidence: bool = False,
        confidence_head_type: str = "pooled_mlp",
        confidence_contact_cutoff: float | None = None,
        confidence_use_time_features: bool = False,
    ):
        super().__init__()
        self.ns = ns
        self.num_conv_layers = num_conv_layers
        self.atom_lig_confidence = atom_lig_confidence
        self.confidence_head_type = str(confidence_head_type).lower()
        if self.confidence_head_type not in {"pooled_mlp", "contact_pool"}:
            raise ValueError("confidence_head_type must be one of {'pooled_mlp', 'contact_pool'}.")
        self.confidence_contact_cutoff = confidence_contact_cutoff
        self.confidence_use_time_features = confidence_use_time_features
        scalar_dim = 2 * ns if num_conv_layers >= 3 else ns
        self.confidence_scalar_dim = scalar_dim

        confidence_input = scalar_dim * 3
        if self.confidence_head_type == "contact_pool":
            self.confidence_contact_pair_mlp = nn.Sequential(
                nn.Linear(2 * scalar_dim + ns, ns),
                nn.LayerNorm(ns),
                activation,
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
            )
            confidence_input += ns + 4
        else:
            self.confidence_contact_pair_mlp = None
        if confidence_use_time_features:
            confidence_input += 2

        self.confidence_predictor = nn.Sequential(
            nn.Linear(confidence_input, ns),
            nn.LayerNorm(ns),
            activation,
            nn.Dropout(confidence_dropout),
            nn.Linear(ns, ns),
            nn.LayerNorm(ns),
            activation,
            nn.Dropout(confidence_dropout),
            nn.Linear(ns, num_confidence_outputs),
        )
        self.affinity_predictor = nn.Sequential(
            nn.Linear(scalar_dim * 3, ns),
            nn.LayerNorm(ns),
            activation,
            nn.Dropout(confidence_dropout),
            nn.Linear(ns, ns),
            nn.LayerNorm(ns),
            activation,
            nn.Dropout(confidence_dropout),
            nn.Linear(ns, 1),
        )
        if atom_lig_confidence:
            self.atom_confidence_predictor = nn.Sequential(
                nn.Linear(scalar_dim, ns),
                nn.LayerNorm(ns),
                activation,
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, ns),
                nn.LayerNorm(ns),
                activation,
                nn.Dropout(confidence_dropout),
                nn.Linear(ns, num_confidence_outputs),
            )

    def forward(self, trunk: TrunkOutput, data) -> dict:
        lig_scalar_node_attr = self.get_scalar_confidence_attr(trunk.ligand)
        rec_scalar_node_attr = self.get_scalar_confidence_attr(trunk.receptor)
        atom_scalar_node_attr = self.get_scalar_confidence_attr(trunk.atom)
        scalar_lig_attr = scatter_mean(lig_scalar_node_attr, data["ligand"].batch, dim=0)
        scalar_rec_attr = scatter_mean(rec_scalar_node_attr, data["receptor"].batch, dim=0)
        scalar_atom_attr = scatter_mean(atom_scalar_node_attr, data["atom"].batch, dim=0)
        pooled = torch.cat(
            [scalar_lig_attr.detach(), scalar_rec_attr.detach(), scalar_atom_attr.detach()],
            dim=1,
        )
        confidence_input = self.build_confidence_head_input(
            data,
            pooled,
            lig_scalar_node_attr,
            atom_scalar_node_attr,
            trunk.ligand_atom_edge_index,
            trunk.ligand_atom_edge_attr,
        )
        confidence = self.confidence_predictor(confidence_input).squeeze(dim=-1)
        affinity = self.affinity_predictor(pooled).squeeze(dim=-1)
        outputs = {"filtering_pred": confidence, "affinity_pred": affinity}
        if self.atom_lig_confidence:
            atom_input = scatter_mean(atom_scalar_node_attr, data["atom"].batch, dim=0)
            outputs["filtering_atom_pred"] = self.atom_confidence_predictor(atom_input.detach()).squeeze(dim=-1)
        return outputs

    def get_scalar_confidence_attr(self, node_attr):
        if self.num_conv_layers >= 3:
            return torch.cat([node_attr[:, : self.ns], node_attr[:, -self.ns :]], dim=1)
        return node_attr[:, : self.ns]

    def build_confidence_head_input(
        self,
        data,
        pooled_confidence_input,
        lig_scalar_node_attr,
        atom_scalar_node_attr,
        la_edge_index,
        la_edge_attr,
    ):
        inputs = [pooled_confidence_input]
        if self.confidence_head_type == "contact_pool":
            contact_pooled, contact_stats = self.build_confidence_contact_summary(
                data, lig_scalar_node_attr, atom_scalar_node_attr, la_edge_index, la_edge_attr
            )
            inputs.extend([contact_pooled, contact_stats])
        if self.confidence_use_time_features:
            t_graph = data.complex_t["t"].view(-1).to(pooled_confidence_input.dtype)
            inputs.append(torch.stack([t_graph, 1.0 - t_graph], dim=1))
        return torch.cat(inputs, dim=1)

    def build_confidence_contact_summary(
        self,
        data,
        lig_scalar_node_attr,
        atom_scalar_node_attr,
        la_edge_index,
        la_edge_attr,
    ):
        num_graphs = data.num_graphs
        device = lig_scalar_node_attr.device
        dtype = lig_scalar_node_attr.dtype
        if la_edge_index.numel() == 0:
            return torch.zeros(num_graphs, self.ns, device=device, dtype=dtype), torch.zeros(num_graphs, 4, device=device, dtype=dtype)

        lig_idx = la_edge_index[0].long()
        atom_idx = la_edge_index[1].long()
        edge_distance = (data["atom"].pos[atom_idx] - data["ligand"].pos[lig_idx]).norm(dim=-1)
        if self.confidence_contact_cutoff is not None:
            keep = edge_distance <= self.confidence_contact_cutoff
            lig_idx, atom_idx, la_edge_attr, edge_distance = lig_idx[keep], atom_idx[keep], la_edge_attr[keep], edge_distance[keep]
        if lig_idx.numel() == 0:
            return torch.zeros(num_graphs, self.ns, device=device, dtype=dtype), torch.zeros(num_graphs, 4, device=device, dtype=dtype)

        graph_index = data["ligand"].batch[lig_idx].long()
        pair_input = torch.cat(
            [lig_scalar_node_attr[lig_idx].detach(), atom_scalar_node_attr[atom_idx].detach(), la_edge_attr.detach()],
            dim=1,
        )
        contact_hidden = self.confidence_contact_pair_mlp(pair_input)
        contact_pooled = scatter_mean(contact_hidden, graph_index, dim=0, dim_size=num_graphs)

        contact_count = torch.zeros(num_graphs, device=device, dtype=dtype)
        contact_count.index_add_(0, graph_index, torch.ones_like(edge_distance, dtype=dtype))
        distance_sum = torch.zeros(num_graphs, device=device, dtype=dtype)
        distance_sum.index_add_(0, graph_index, edge_distance.to(dtype))
        inv_distance_sum = torch.zeros(num_graphs, device=device, dtype=dtype)
        inv_distance_sum.index_add_(0, graph_index, edge_distance.to(dtype).add(1.0e-6).reciprocal())
        distance_min, _ = scatter_min(edge_distance.to(dtype), graph_index, dim=0, dim_size=num_graphs)
        valid = contact_count > 0
        distance_mean = torch.zeros_like(distance_sum)
        inv_distance_mean = torch.zeros_like(inv_distance_sum)
        distance_mean[valid] = distance_sum[valid] / contact_count[valid]
        inv_distance_mean[valid] = inv_distance_sum[valid] / contact_count[valid]
        distance_min = torch.where(valid, distance_min, torch.zeros_like(distance_min))
        stats = torch.stack(
            [torch.log1p(contact_count), distance_mean, distance_min, inv_distance_mean],
            dim=1,
        )
        return contact_pooled, stats


def safe_rms(value: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if value is None or value.numel() == 0:
        return reference.new_tensor(0.0)
    return torch.sqrt(value.detach().square().mean() + 1e-12)
