from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from docking_model.models.graph_builders import GraphBuilderStack
from docking_model.models.heads import ConfidenceAffinityHeads, LigandPoseHeads, ProteinFlexibilityHeads
from docking_model.models.tensor_ops import drop_private_outputs
from docking_model.models.trunk import EquivariantTrunk


class DockingModel(nn.Module):
    """Split score model with graph builders, equivariant trunk, and prediction heads."""

    def __init__(
        self,
        t_to_sigma,
        timestep_emb_func,
        in_lig_edge_features: int = 5,
        sigma_embed_dim: int = 32,
        sh_lmax: int = 2,
        ns: int = 16,
        nv: int = 4,
        num_conv_layers: int = 2,
        lig_max_radius: float = 5.0,
        rec_max_radius: float = 30.0,
        cross_max_distance: float = 250.0,
        center_max_distance: float = 30.0,
        distance_embed_dim: int = 32,
        cross_distance_embed_dim: int = 32,
        no_torsion: bool = False,
        scale_by_sigma: bool = True,
        use_second_order_repr: bool = False,
        batch_norm: bool = True,
        norm_type: Literal["batch_norm", "layer_norm"] | None = None,
        dynamic_max_cross: bool = False,
        dropout: float = 0.0,
        smooth_edges: bool = False,
        odd_parity: bool = False,
        lm_embedding_type: str | None = None,
        confidence_mode: bool = False,
        confidence_dropout: float = 0.0,
        confidence_no_batchnorm: bool = False,
        asyncronous_noise_schedule: bool = False,
        num_confidence_outputs: int = 1,
        fixed_center_conv: bool = False,
        no_aminoacid_identities: bool = False,
        flexible_sidechains: bool = False,
        flexible_backbone: bool = False,
        differentiate_convolutions: bool = True,
        tp_weights_layers: int = 2,
        use_oeq_kernels: bool = False,
        reduce_pseudoscalars: bool = False,
        c_alpha_radius: float = 20.0,
        c_alpha_max_neighbors: int | None = None,
        atom_radius: float = 5.0,
        atom_max_neighbors: int | None = None,
        sidechain_tor_bridge: bool = False,
        use_bb_orientation_feats: bool = False,
        only_nearby_residues_atomic: bool = False,
        atom_lig_confidence: bool = False,
        confidence_head_type: str = "pooled_mlp",
        confidence_contact_cutoff: float | None = None,
        confidence_use_time_features: bool = False,
        activation_func: str = "ReLU",
        norm_affine: bool = True,
        clamped_norm_min: float = 1.0e-6,
        lig_transform_type: str = "diffusion",
        tor_fourier_enabled: bool = False,
        tor_fourier_num_freqs: int = 8,
        sc_tor_fourier_enabled: bool = False,
        sc_tor_fourier_num_freqs: int = 8,
        sc_tor_fourier_sigma_conditioning: bool = True,
        sc_tor_fourier_gate_type: str = "none",
        sc_tor_fourier_gate_hidden: int = 64,
        sc_tor_fourier_poly_degree: int = 3,
        sc_tor_fourier_joint_time: bool = False,
        tor_sc_coupling_enabled: bool = False,
        tor_sc_coupling_unary_enabled: bool = True,
        tor_sc_coupling_pairwise_enabled: bool = True,
        tor_sc_coupling_radius: float = 6.0,
        tor_sc_coupling_max_neighbors: int = 64,
        **kwargs,
    ):
        super().__init__()
        if no_aminoacid_identities and lm_embedding_type is not None:
            raise ValueError("no_aminoacid_identities=True is incompatible with language-model embeddings.")
        if tor_fourier_enabled and no_torsion:
            raise ValueError("tor_fourier_enabled=True requires no_torsion=False.")
        if tor_fourier_enabled and lig_transform_type != "diffusion":
            raise ValueError("tor_fourier_enabled=True supports only ligand diffusion.")
        if tor_fourier_enabled and tor_fourier_num_freqs < 1:
            raise ValueError("tor_fourier_num_freqs must be >= 1.")
        if sc_tor_fourier_enabled and not flexible_sidechains:
            raise ValueError("sc_tor_fourier_enabled=True requires flexible_sidechains=True.")
        if sc_tor_fourier_enabled:
            if sc_tor_fourier_num_freqs < 1:
                raise ValueError("sc_tor_fourier_num_freqs must be >= 1.")
            if str(sc_tor_fourier_gate_type).lower() not in {"none", "mlp", "bell", "poly"}:
                raise ValueError("sc_tor_fourier_gate_type must be one of {'none','mlp','bell','poly'}.")
            if sc_tor_fourier_gate_hidden < 1:
                raise ValueError("sc_tor_fourier_gate_hidden must be >= 1.")
            if sc_tor_fourier_poly_degree < 1:
                raise ValueError("sc_tor_fourier_poly_degree must be >= 1.")
            if not sidechain_tor_bridge:
                if str(sc_tor_fourier_gate_type).lower() != "none":
                    raise ValueError("VE sidechain Fourier requires sc_tor_fourier_gate_type='none'.")
                if sc_tor_fourier_joint_time:
                    raise ValueError("VE sidechain Fourier does not support sc_tor_fourier_joint_time=True.")
        if tor_sc_coupling_enabled:
            if not tor_fourier_enabled:
                raise ValueError("tor_sc_coupling_enabled=True requires tor_fourier_enabled=True.")
            if not sc_tor_fourier_enabled:
                raise ValueError("tor_sc_coupling_enabled=True requires sc_tor_fourier_enabled=True.")
            if not (tor_sc_coupling_unary_enabled or tor_sc_coupling_pairwise_enabled):
                raise ValueError("tor_sc_coupling_enabled=True requires unary and/or pairwise coupling enabled.")
            if tor_sc_coupling_radius <= 0:
                raise ValueError("tor_sc_coupling_radius must be > 0.")
            if tor_sc_coupling_max_neighbors < 1:
                raise ValueError("tor_sc_coupling_max_neighbors must be >= 1.")

        activation = build_activation(activation_func)
        self.confidence_mode = confidence_mode
        self.no_aminoacid_identities = no_aminoacid_identities
        self.no_torsion = no_torsion
        self.flexible_sidechains = flexible_sidechains
        self.flexible_backbone = flexible_backbone
        self.sidechain_tor_bridge = sidechain_tor_bridge
        self.use_bb_orientation_feats = use_bb_orientation_feats

        self.graph_builder = GraphBuilderStack(
            t_to_sigma=t_to_sigma,
            timestep_emb_func=timestep_emb_func,
            in_lig_edge_features=in_lig_edge_features,
            sigma_embed_dim=sigma_embed_dim,
            sh_lmax=sh_lmax,
            lig_max_radius=lig_max_radius,
            rec_max_radius=rec_max_radius,
            cross_max_distance=cross_max_distance,
            center_max_distance=center_max_distance,
            distance_embed_dim=distance_embed_dim,
            cross_distance_embed_dim=cross_distance_embed_dim,
            dynamic_max_cross=dynamic_max_cross,
            smooth_edges=smooth_edges,
            asyncronous_noise_schedule=asyncronous_noise_schedule,
            no_aminoacid_identities=no_aminoacid_identities,
            flexible_sidechains=flexible_sidechains,
            flexible_backbone=flexible_backbone,
            c_alpha_radius=c_alpha_radius,
            c_alpha_max_neighbors=c_alpha_max_neighbors,
            atom_radius=atom_radius,
            atom_max_neighbors=atom_max_neighbors,
            only_nearby_residues_atomic=only_nearby_residues_atomic,
        )
        self.trunk = EquivariantTrunk(
            sigma_embed_dim=sigma_embed_dim,
            sh_lmax=sh_lmax,
            ns=ns,
            nv=nv,
            num_conv_layers=num_conv_layers,
            in_lig_edge_features=in_lig_edge_features,
            distance_embed_dim=distance_embed_dim,
            cross_distance_embed_dim=cross_distance_embed_dim,
            dropout=dropout,
            activation=activation,
            use_second_order_repr=use_second_order_repr,
            reduce_pseudoscalars=reduce_pseudoscalars,
            norm_type=norm_type,
            norm_affine=norm_affine,
            differentiate_convolutions=differentiate_convolutions,
            tp_weights_layers=tp_weights_layers,
            use_oeq_kernels=use_oeq_kernels,
            use_bb_orientation_feats=use_bb_orientation_feats,
            lm_embedding_type=lm_embedding_type,
        )
        self.confidence_heads = ConfidenceAffinityHeads(
            ns=ns,
            num_conv_layers=num_conv_layers,
            num_confidence_outputs=num_confidence_outputs,
            activation=activation,
            confidence_dropout=confidence_dropout,
            atom_lig_confidence=atom_lig_confidence,
            confidence_head_type=confidence_head_type,
            confidence_contact_cutoff=confidence_contact_cutoff,
            confidence_use_time_features=confidence_use_time_features,
        )
        self.ligand_heads = LigandPoseHeads(
            graph_builder=self.graph_builder,
            timestep_emb_func=timestep_emb_func,
            trunk_out_irreps=self.trunk.out_irreps,
            sigma_embed_dim=sigma_embed_dim,
            ns=ns,
            sh_lmax=sh_lmax,
            distance_embed_dim=distance_embed_dim,
            dropout=dropout,
            activation=activation,
            no_torsion=no_torsion,
            scale_by_sigma=scale_by_sigma,
            odd_parity=odd_parity,
            fixed_center_conv=fixed_center_conv,
            clamped_norm_min=clamped_norm_min,
            tor_fourier_enabled=tor_fourier_enabled,
            tor_fourier_num_freqs=tor_fourier_num_freqs,
            norm_type=norm_type,
            norm_affine=norm_affine,
            use_second_order_repr=use_second_order_repr,
            use_oeq_kernels=use_oeq_kernels,
        )
        self.protein_heads = ProteinFlexibilityHeads(
            graph_builder=self.graph_builder,
            timestep_emb_func=timestep_emb_func,
            trunk_out_irreps=self.trunk.out_irreps,
            sigma_embed_dim=sigma_embed_dim,
            ns=ns,
            sh_lmax=sh_lmax,
            distance_embed_dim=distance_embed_dim,
            dropout=dropout,
            activation=activation,
            flexible_sidechains=flexible_sidechains,
            flexible_backbone=flexible_backbone,
            sidechain_tor_bridge=sidechain_tor_bridge,
            scale_by_sigma=scale_by_sigma,
            odd_parity=odd_parity,
            sc_tor_fourier_enabled=sc_tor_fourier_enabled,
            sc_tor_fourier_num_freqs=sc_tor_fourier_num_freqs,
            sc_tor_fourier_sigma_conditioning=sc_tor_fourier_sigma_conditioning,
            sc_tor_fourier_gate_type=sc_tor_fourier_gate_type,
            sc_tor_fourier_gate_hidden=sc_tor_fourier_gate_hidden,
            sc_tor_fourier_poly_degree=sc_tor_fourier_poly_degree,
            sc_tor_fourier_joint_time=sc_tor_fourier_joint_time,
            tor_sc_coupling_enabled=tor_sc_coupling_enabled,
            tor_fourier_enabled=tor_fourier_enabled,
            tor_fourier_num_freqs=tor_fourier_num_freqs,
            tor_sc_coupling_unary_enabled=tor_sc_coupling_unary_enabled,
            tor_sc_coupling_pairwise_enabled=tor_sc_coupling_pairwise_enabled,
            tor_sc_coupling_radius=tor_sc_coupling_radius,
            tor_sc_coupling_max_neighbors=tor_sc_coupling_max_neighbors,
            norm_type=norm_type,
            norm_affine=norm_affine,
            use_oeq_kernels=use_oeq_kernels,
        )

    def forward(self, data) -> dict:
        graphs = self.graph_builder(data)
        trunk = self.trunk(graphs)
        confidence_outputs = self.confidence_heads(trunk, data)
        if self.confidence_mode:
            return confidence_outputs

        outputs = {}
        ligand_outputs = self.ligand_heads(
            trunk, data, include_torsion_context=self.protein_heads.tor_sc_coupling_enabled
        )
        outputs.update(ligand_outputs)
        outputs.update(self.protein_heads(trunk, data, ligand_outputs=ligand_outputs))
        outputs.update(confidence_outputs)
        drop_private_outputs(outputs)
        return outputs

def build_activation(name: str) -> nn.Module:
    lowered = str(name).lower()
    if lowered == "relu":
        return nn.ReLU()
    if lowered == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation function: {name}")
