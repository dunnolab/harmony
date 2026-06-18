from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal


ProteinFlexibility = Literal["rigid", "sidechain", "full"]
DockingRegion = Literal["blind", "pocket"]
SidechainTorsionMode = Literal["bridge", "diffusion"]
PrecisionMode = Literal["fp32", "amp", "fp16", "fp16-mixed", "16-mixed", "bf16", "bf16-mixed"]


class ConfigSection:
    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = data or {}
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class ProteinConfig(ConfigSection):
    flexibility: ProteinFlexibility = "full"
    sidechain_tor_transform_type: SidechainTorsionMode = "bridge"
    use_bb_orientation_feats: bool = False

    @property
    def flexible_backbone(self) -> bool:
        return self.flexibility == "full"

    @property
    def flexible_sidechains(self) -> bool:
        return self.flexibility in {"sidechain", "full"}

    @property
    def sidechain_tor_bridge(self) -> bool:
        return self.sidechain_tor_transform_type == "bridge"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "flexibility" not in values:
            flexible_backbone = bool(values.get("flexible_backbone", False))
            flexible_sidechains = bool(values.get("flexible_sidechains", flexible_backbone))
            if flexible_backbone:
                values["flexibility"] = "full"
            elif flexible_sidechains:
                values["flexibility"] = "sidechain"
            else:
                values["flexibility"] = "rigid"

        if "sidechain_tor_transform_type" not in values and "sidechain_tor_bridge" in values:
            values["sidechain_tor_transform_type"] = (
                "bridge" if bool(values["sidechain_tor_bridge"]) else "diffusion"
            )
        return super().from_dict(values)


@dataclass
class LigandConfig(ConfigSection):
    torsions: bool = True

    @property
    def no_torsion(self) -> bool:
        return not self.torsions

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "torsions" not in values and "no_torsion" in values:
            values["torsions"] = not bool(values["no_torsion"])
        return super().from_dict(values)


@dataclass
class PocketConfig(ConfigSection):
    region: DockingRegion = "pocket"
    radius: float = 5.0
    buffer: float = 20.0
    min_size: int = 1
    all_atoms: bool = True

    @property
    def enabled(self) -> bool:
        return self.region == "pocket"

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "pocket_reduction" in values:
            values["region"] = "pocket" if bool(values["pocket_reduction"]) else "blind"
        copy_key(values, "pocket_radius", "radius")
        copy_key(values, "pocket_buffer", "buffer")
        copy_key(values, "pocket_min_size", "min_size")
        if "all_atoms" in values:
            values["all_atoms"] = bool(values["all_atoms"])
        return super().from_dict(values)


@dataclass
class NearbyAtomsConfig(ConfigSection):
    restrict_to_nearby: bool = True
    selection_mode: Literal["direct", "radius_based"] = "direct"
    radius: float = 6.0
    min_atoms: int = 8

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        copy_key(values, "only_nearby_residues_atomic", "restrict_to_nearby")
        copy_key(values, "nearby_residues_selection_mode", "selection_mode")
        copy_key(values, "nearby_residues_atomic_radius", "radius")
        copy_key(values, "nearby_residues_atomic_min", "min_atoms")
        if "restrict_to_nearby" in values:
            values["restrict_to_nearby"] = bool(values["restrict_to_nearby"])
        return super().from_dict(values)


@dataclass
class BackbonePriorConfig(ConfigSection):
    bb_random_prior: bool = False
    bb_random_prior_noise: Literal["gaussian", "harmonic"] = "gaussian"
    bb_random_prior_ot: bool = False
    bb_random_prior_std: float = 0.5
    bb_random_prior_ot_inf: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        for key in ["bb_random_prior", "bb_random_prior_ot", "bb_random_prior_ot_inf"]:
            if key in values:
                values[key] = bool(values[key])
        return super().from_dict(values)


@dataclass
class UnbalancedConfig(ConfigSection):
    match_max_rmsd: float | None = None


@dataclass
class ModelConfig(ConfigSection):
    name: str = "docking_model"
    checkpoint: str | None = None
    all_atoms: bool = True
    no_torsion: bool = False
    flexible_backbone: bool = False
    flexible_sidechains: bool = True
    sidechain_tor_bridge: bool = True
    use_bb_orientation_feats: bool = False
    only_nearby_residues_atomic: bool = True

    esm_embeddings_path: str | None = None
    esm_embeddings_model: str | None = None

    confidence_mode: bool = False
    num_confidence_outputs: int = 1
    rmsd_classification_cutoff: list[float] = field(default_factory=list)
    confidence_head_type: Literal["pooled_mlp", "contact_pool"] = "pooled_mlp"
    confidence_contact_cutoff: float | None = None
    confidence_use_time_features: bool = False
    atom_lig_confidence: bool = False
    confidence_dropout: float = 0.0
    confidence_no_batchnorm: bool = False

    in_lig_edge_features: int = 5
    max_radius: float | None = None
    ligand_max_radius: float = 5.0
    receptor_radius: float = 15.0
    c_alpha_max_neighbors: int | None = 24
    atom_radius: float = 5.0
    atom_max_neighbors: int | None = 12
    cross_max_distance: float = 80.0
    dynamic_max_cross: bool = False

    sigma_embed_type: Literal["sinusoidal", "fourier"] = "sinusoidal"
    sigma_embed_dim: int = 32
    embedding_scale: float = 1000.0
    distance_embed_dim: int = 32
    cross_distance_embed_dim: int = 32
    ns: int = 60
    nv: int = 15
    num_conv_layers: int = 6
    sh_lmax: int = 2
    use_second_order_repr: bool = False
    reduce_pseudoscalars: bool = False
    differentiate_convolutions: bool = True
    tp_weights_layers: int = 2

    no_batch_norm: bool = False
    norm_type: Literal["batch_norm", "layer_norm", "none"] = "layer_norm"
    norm_affine: bool = True
    dropout: float = 0.1
    activation_func: str = "silu"
    clamped_norm_min: float = 1.0e-6

    use_oeq_kernels: bool = True
    scale_by_sigma: bool = True
    smooth_edges: bool = False
    odd_parity: bool = False
    not_fixed_center_conv: bool = False
    no_aminoacid_identities: bool = False

    lig_transform_type: Literal["diffusion", "flow"] = "diffusion"
    tor_fourier_enabled: bool = False
    tor_fourier_num_freqs: int = 8
    sc_tor_fourier_enabled: bool = False
    sc_tor_fourier_num_freqs: int = 4
    sc_tor_fourier_sigma_conditioning: bool = True
    sc_tor_fourier_gate_type: Literal["none", "mlp", "bell", "poly"] = "none"
    sc_tor_fourier_gate_hidden: int = 64
    sc_tor_fourier_poly_degree: int = 3
    sc_tor_fourier_joint_time: bool = False
    tor_sc_coupling_enabled: bool = False
    tor_sc_coupling_unary_enabled: bool = True
    tor_sc_coupling_pairwise_enabled: bool = True
    tor_sc_coupling_radius: float = 6.0
    tor_sc_coupling_max_neighbors: int = 64

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "max_radius" in values and "ligand_max_radius" not in values:
            values["ligand_max_radius"] = values["max_radius"]
        if "embedding_type" in values and "sigma_embed_type" not in values:
            values["sigma_embed_type"] = values["embedding_type"]
        if "sigma_embed_scale" in values and "embedding_scale" not in values:
            values["embedding_scale"] = values["sigma_embed_scale"]
        protein_values = dict(values.get("protein") or {})
        ligand_values = dict(values.get("ligand") or {})
        for old, new in [
            ("flexible_backbone", "flexible_backbone"),
            ("flexible_sidechains", "flexible_sidechains"),
            ("sidechain_tor_bridge", "sidechain_tor_bridge"),
            ("use_bb_orientation_feats", "use_bb_orientation_feats"),
        ]:
            if old in protein_values and new not in values:
                values[new] = protein_values[old]
        if "no_torsion" not in values and "torsions" in ligand_values:
            values["no_torsion"] = not bool(ligand_values["torsions"])
        for key in [
            "all_atoms",
            "no_torsion",
            "flexible_backbone",
            "flexible_sidechains",
            "sidechain_tor_bridge",
            "use_bb_orientation_feats",
            "only_nearby_residues_atomic",
            "confidence_mode",
            "confidence_use_time_features",
            "atom_lig_confidence",
            "confidence_no_batchnorm",
            "dynamic_max_cross",
            "use_second_order_repr",
            "reduce_pseudoscalars",
            "differentiate_convolutions",
            "no_batch_norm",
            "norm_affine",
            "use_oeq_kernels",
            "scale_by_sigma",
            "smooth_edges",
            "odd_parity",
            "not_fixed_center_conv",
            "no_aminoacid_identities",
            "tor_fourier_enabled",
            "sc_tor_fourier_enabled",
            "sc_tor_fourier_sigma_conditioning",
            "sc_tor_fourier_joint_time",
            "tor_sc_coupling_enabled",
            "tor_sc_coupling_unary_enabled",
            "tor_sc_coupling_pairwise_enabled",
        ]:
            if key in values:
                values[key] = bool(values[key])
        return super().from_dict(values)


@dataclass
class SigmaConfig(ConfigSection):
    tr_sigma_min: float = 0.1
    tr_sigma_max: float = 15.0
    rot_sigma_min: float = 0.03
    rot_sigma_max: float = 1.55
    tor_sigma_min: float = 0.0314
    tor_sigma_max: float = 3.14
    bb_tr_sigma: float | None = 1.8
    bb_rot_sigma: float | None = 0.6
    sidechain_tor_sigma: float | None = 0.8
    sidechain_tor_sigma_min: float | None = 0.0314
    sidechain_tor_sigma_max: float | None = 3.14
    sidechain_tor_bridge: bool = True
    sidechain_tor_transform_type: SidechainTorsionMode = "bridge"

    def __post_init__(self):
        validate_sigma_bounds("tr", self.tr_sigma_min, self.tr_sigma_max)
        validate_sigma_bounds("rot", self.rot_sigma_min, self.rot_sigma_max)
        validate_sigma_bounds("tor", self.tor_sigma_min, self.tor_sigma_max)
        if self.sidechain_tor_sigma_min is not None and self.sidechain_tor_sigma_max is not None:
            validate_sigma_bounds("sidechain_tor", self.sidechain_tor_sigma_min, self.sidechain_tor_sigma_max)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "sidechain_tor_transform_type" not in values and "sidechain_tor_bridge" in values:
            values["sidechain_tor_transform_type"] = (
                "bridge" if bool(values["sidechain_tor_bridge"]) else "diffusion"
            )
        if "sidechain_tor_bridge" in values:
            values["sidechain_tor_bridge"] = bool(values["sidechain_tor_bridge"])
        return super().from_dict(values)


@dataclass
class TimeConfig(ConfigSection):
    sampling_alpha: float = 2.0
    sampling_beta: float = 2.0
    bb_tr_bridge_alpha: float | None = 2.0
    bb_rot_bridge_alpha: float | None = 2.0
    sc_tor_bridge_alpha: float | None = 2.0


@dataclass
class SamplerConfig(ConfigSection):
    inference_steps: int = 20
    samples_per_complex: int = 1
    batch_size: int = 1
    no_random: bool = False
    no_final_step_noise: bool = True
    ode: bool = False
    initial_noise_std_proportion: float = 1.0
    reset_sidechain_ve_to_apo_before_randomize: bool = False
    k_samples_per_complex: int = 1
    all_atoms: bool = True
    no_torsion: bool = False
    flexible_backbone: bool = False
    flexible_sidechains: bool = True
    sidechain_tor_bridge: bool = True
    sidechain_tor_transform_type: SidechainTorsionMode = "bridge"
    use_bb_orientation_feats: bool = False
    bb_tr_bridge_alpha: float | None = 2.0
    bb_rot_bridge_alpha: float | None = 2.0
    sc_tor_bridge_alpha: float | None = 2.0
    sampling_alpha: float = 2.0
    sampling_beta: float = 2.0
    lig_transform_type: Literal["diffusion", "flow"] = "diffusion"
    precision: str | None = None
    sigma: dict[str, Any] | None = None
    return_full_trajectory: bool = False
    diff_temp_sampling_tr: float = 1.0
    diff_temp_sampling_rot: float = 1.0
    diff_temp_sampling_tor: float = 1.0
    diff_temp_psi_tr: float = 0.0
    diff_temp_psi_rot: float = 0.0
    diff_temp_psi_tor: float = 0.0
    diff_temp_sigma_data_tr: float = 0.5
    diff_temp_sigma_data_rot: float = 0.5
    diff_temp_sigma_data_tor: float = 0.5
    flow_temp_scale_0_tr: float = 1.0
    flow_temp_scale_0_rot: float = 1.0
    flow_temp_scale_0_tor: float = 1.0
    flow_temp_scale_1_tr: float = 1.0
    flow_temp_scale_1_rot: float = 1.0
    flow_temp_scale_1_tor: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "samples_per_complex" not in values and "k_samples_per_complex" in values:
            values["samples_per_complex"] = values["k_samples_per_complex"]
        if "sidechain_tor_transform_type" not in values and "sidechain_tor_bridge" in values:
            values["sidechain_tor_transform_type"] = (
                "bridge" if bool(values["sidechain_tor_bridge"]) else "diffusion"
            )
        for key in [
            "no_random",
            "no_final_step_noise",
            "ode",
            "reset_sidechain_ve_to_apo_before_randomize",
            "all_atoms",
            "no_torsion",
            "flexible_backbone",
            "flexible_sidechains",
            "sidechain_tor_bridge",
            "use_bb_orientation_feats",
            "return_full_trajectory",
        ]:
            if key in values:
                values[key] = bool(values[key])
        return super().from_dict(values)

    @property
    def diff_temp_sampling(self) -> tuple[float, float, float]:
        return (self.diff_temp_sampling_tr, self.diff_temp_sampling_rot, self.diff_temp_sampling_tor)

    @property
    def diff_temp_psi(self) -> tuple[float, float, float]:
        return (self.diff_temp_psi_tr, self.diff_temp_psi_rot, self.diff_temp_psi_tor)

    @property
    def diff_temp_sigma_data(self) -> tuple[float, float, float]:
        return (self.diff_temp_sigma_data_tr, self.diff_temp_sigma_data_rot, self.diff_temp_sigma_data_tor)

    @property
    def flow_temp_scale_0(self) -> tuple[float, float, float]:
        return (self.flow_temp_scale_0_tr, self.flow_temp_scale_0_rot, self.flow_temp_scale_0_tor)

    @property
    def flow_temp_scale_1(self) -> tuple[float, float, float]:
        return (self.flow_temp_scale_1_tr, self.flow_temp_scale_1_rot, self.flow_temp_scale_1_tor)


@dataclass
class DataConfig(ConfigSection):
    task: str = "docking"
    dataset: str = "pdbbind"
    input_csv: str | None = None
    preprocess_raw: bool = False
    cache_path: str | None = None
    split_train: str | None = None
    split_val: str | None = None
    affinity_csv: str | None = None
    cluster_file: str | None = None
    require_ligand: bool = True
    limit_complexes: int = 0
    complexes_per_cluster: int = 1
    multiplicity: int = 1
    batch_size: int = 4
    num_workers: int = 0
    pin_memory: bool = False
    drop_last: bool = False
    run_val_inference: bool = False
    num_inference_complexes: int = 500
    matching: bool = True
    matching_popsize: int = 15
    matching_maxiter: int = 15
    keep_original: bool = False
    remove_hs: bool = False
    num_conformers: int = 1
    max_lig_size: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "num_dataloader_workers" in values and "num_workers" not in values:
            values["num_workers"] = values["num_dataloader_workers"]
        if "dataloader_drop_last" in values and "drop_last" not in values:
            values["drop_last"] = values["dataloader_drop_last"]
        for key in [
            "require_ligand",
            "pin_memory",
            "drop_last",
            "run_val_inference",
            "preprocess_raw",
            "matching",
            "keep_original",
            "remove_hs",
        ]:
            if key in values:
                values[key] = bool(values[key])
        return super().from_dict(values)


@dataclass
class InferenceConfig(ConfigSection):
    model_parameters: str | None = None
    checkpoint: str | None = None
    docking_model_dir: str | None = None
    docking_ckpt: str | None = None
    filtering_model_dir: str | None = None
    filtering_ckpt: str | None = None
    esm_embeddings_path: str | None = None
    input_csv: str | None = None
    output_dir: str = "inference_outputs"
    results_table_csv: str | None = None
    cache_path: str | None = None
    split_path: str | None = None
    limit_complexes: int | None = None
    complex_id: str | None = None
    save_trajectory: bool = False
    export_trajectory_files: bool = False
    trajectory_max_ranks: int | None = None
    posebusters_metrics: bool = False
    wandb_max_complex_examples: int = 24
    use_ema_weights: bool = True
    batch_size: int | None = None
    actual_steps: int | None = None
    inference_steps: int | None = None
    samples_per_complex: int | None = None
    k_samples_per_complex: int | None = None
    ode: bool | None = None
    pocket_reduction: bool | None = None
    pocket_radius: float | None = None
    pocket_buffer: float | None = None
    pocket_min_size: int | None = None
    only_nearby_residues_atomic: bool | None = None
    nearby_residues_selection_mode: str | None = None
    nearby_residues_atomic_radius: float | None = None
    nearby_residues_atomic_min: int | None = None
    reset_sidechain_ve_to_apo_before_randomize: bool | None = None
    flexible_backbone: bool | None = None
    flexible_sidechains: bool | None = None
    all_atoms: bool | None = None
    debug_backbone: bool = False
    debug_sidechains: bool = False
    diff_temp_sampling_tr: float = 1.0
    diff_temp_sampling_rot: float = 1.0
    diff_temp_sampling_tor: float = 1.0
    diff_temp_psi_tr: float = 0.0
    diff_temp_psi_rot: float = 0.0
    diff_temp_psi_tor: float = 0.0
    diff_temp_sigma_data_tr: float = 0.5
    diff_temp_sigma_data_rot: float = 0.5
    diff_temp_sigma_data_tor: float = 0.5
    flow_temp_scale_0_tr: float = 1.0
    flow_temp_scale_0_rot: float = 1.0
    flow_temp_scale_0_tor: float = 1.0
    flow_temp_scale_1_tr: float = 1.0
    flow_temp_scale_1_rot: float = 1.0
    flow_temp_scale_1_tor: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        for key in [
            "save_trajectory",
            "export_trajectory_files",
            "posebusters_metrics",
            "use_ema_weights",
            "ode",
            "pocket_reduction",
            "only_nearby_residues_atomic",
            "reset_sidechain_ve_to_apo_before_randomize",
            "flexible_backbone",
            "flexible_sidechains",
            "all_atoms",
            "debug_backbone",
            "debug_sidechains",
        ]:
            if key in values and values[key] is not None:
                values[key] = bool(values[key])
        return super().from_dict(values)


@dataclass
class LossConfig(ConfigSection):
    tr_weight: float = 1.0
    rot_weight: float = 1.0
    tor_weight: float = 1.0
    bb_tr_weight: float = 1.0
    bb_rot_weight: float = 1.0
    sc_tor_weight: float = 1.0
    confidence_weight: float = 0.1
    affinity_weight: float = 0.1
    confidence_target: str = "pli_lddt"
    confidence_rmsd_threshold: float = 2.0
    confidence_aux_rmsd_lt2_weight: float = 0.0
    confidence_pos_weight: float | None = None
    confidence_time_weighting: Literal["none", "late_exp", "late_only"] = "none"
    confidence_time_power: float = 2.0
    confidence_time_cutoff: float = 0.3
    lig_transform_type: Literal["diffusion", "flow"] = "diffusion"
    sidechain_tor_bridge: bool = True
    sidechain_tor_transform_type: SidechainTorsionMode = "bridge"
    tor_fourier_enabled: bool = False
    tor_fourier_reg_weight: float = 1.0e-4
    sc_tor_fourier_enabled: bool = False
    sc_tor_fourier_reg_weight: float = 1.0e-4
    sc_tor_gate_reg_weight: float = 1.0e-5
    tor_sc_coupling_enabled: bool = False
    tor_sc_coupling_reg_weight: float = 1.0e-5
    flexible_backbone: bool = False
    flexible_sidechains: bool = True
    no_torsion: bool = False
    all_atoms: bool = True
    tr_sigma_min: float = 0.1
    tr_sigma_max: float = 15.0
    rot_sigma_min: float = 0.03
    rot_sigma_max: float = 1.55
    use_new_pipeline: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "sidechain_tor_transform_type" not in values and "sidechain_tor_bridge" in values:
            values["sidechain_tor_transform_type"] = (
                "bridge" if bool(values["sidechain_tor_bridge"]) else "diffusion"
            )
        for key in [
            "sidechain_tor_bridge",
            "tor_fourier_enabled",
            "sc_tor_fourier_enabled",
            "tor_sc_coupling_enabled",
            "flexible_backbone",
            "flexible_sidechains",
            "no_torsion",
            "all_atoms",
            "use_new_pipeline",
        ]:
            if key in values:
                values[key] = bool(values[key])
        return super().from_dict(values)


@dataclass
class TransformsConfig(ConfigSection):
    lig_transform_type: Literal["diffusion", "flow"] = "diffusion"
    tor_fourier_enabled: bool = False
    flexible_backbone: bool = False
    flexible_sidechains: bool = True
    pocket: PocketConfig = field(default_factory=PocketConfig)
    unbalanced: UnbalancedConfig = field(default_factory=UnbalancedConfig)
    nearby_atoms: NearbyAtomsConfig = field(default_factory=NearbyAtomsConfig)
    ligand: LigandConfig = field(default_factory=LigandConfig)
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    time_args: TimeConfig = field(default_factory=TimeConfig)
    sigma_args: SigmaConfig = field(default_factory=SigmaConfig)
    bb_prior: BackbonePriorConfig = field(default_factory=BackbonePriorConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        for key in ["tor_fourier_enabled", "flexible_backbone", "flexible_sidechains"]:
            if key in values:
                values[key] = bool(values[key])
        values["pocket"] = PocketConfig.from_dict(values.get("pocket"))
        values["unbalanced"] = UnbalancedConfig.from_dict(values.get("unbalanced"))
        values["nearby_atoms"] = NearbyAtomsConfig.from_dict(values.get("nearby_atoms"))
        values["ligand"] = LigandConfig.from_dict(values.get("ligand"))
        values["protein"] = ProteinConfig.from_dict(values.get("protein"))
        values["time_args"] = TimeConfig.from_dict(values.get("time_args"))
        values["sigma_args"] = SigmaConfig.from_dict(values.get("sigma_args"))
        values["bb_prior"] = BackbonePriorConfig.from_dict(values.get("bb_prior"))
        return super().from_dict(values)


@dataclass
class OptimizerConfig(ConfigSection):
    name: Literal["adam", "adamw"] = "adamw"
    lr: float = 1.0e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float | None = None


@dataclass
class TrainingConfig(ConfigSection):
    epochs: int = 1
    device: str = "auto"
    devices: int | str = 1
    strategy: Literal["auto", "ddp", "none"] = "auto"
    find_unused_parameters: bool = True
    precision: PrecisionMode = "fp32"
    log_every: int = 25
    val_inference_freq: int | None = None
    use_ema: bool = False
    ema_rate: float = 0.999
    scheduler: Literal["none", "plateau", "cosineannealing", "exponential"] = "none"
    scheduler_patience: int = 30
    scheduler_gamma: float = 0.99
    inference_earlystop_metric: str | None = None
    inference_earlystop_goal: Literal["min", "max"] = "max"
    check_unused_params: bool = False
    check_nan_grads: bool = False
    except_on_nan_grads: bool = False
    skip_nan_grad_updates: bool = False
    output_dir: str = "runs/docking-model"
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        optimizer_values = dict(values.get("optimizer") or {})
        if "lr" in values and "lr" not in optimizer_values:
            optimizer_values["lr"] = values["lr"]
        if "w_decay" in values and "weight_decay" not in optimizer_values:
            optimizer_values["weight_decay"] = values["w_decay"]
        if "adamw" in values and "name" not in optimizer_values:
            optimizer_values["name"] = "adamw" if str(values["adamw"]).lower() == "adamw" else "adam"
        values["optimizer"] = OptimizerConfig.from_dict(optimizer_values)
        if "scheduler" in values and values["scheduler"] is None:
            values["scheduler"] = "none"
        for key in [
            "use_ema",
            "find_unused_parameters",
            "check_unused_params",
            "check_nan_grads",
            "except_on_nan_grads",
            "skip_nan_grad_updates",
        ]:
            if key in values:
                values[key] = bool(values[key])
        allowed = {field.name for field in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in allowed})


@dataclass
class LoggerConfig(ConfigSection):
    wandb: bool = False
    entity: str | None = None
    project: str = "docking-model"
    name: str | None = None
    tags: list[str] = field(default_factory=list)
    group: str | None = None
    run_id: str | None = None
    resume: str = "allow"
    mode: Literal["online", "offline", "disabled"] | None = None
    dir: str | None = None
    log_config: bool = True
    finish_on_exit: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        if "wandb" in values:
            values["wandb"] = bool(values["wandb"])
        for key in ["log_config", "finish_on_exit"]:
            if key in values:
                values[key] = bool(values[key])
        if values.get("tags") is None:
            values["tags"] = []
        elif isinstance(values.get("tags"), str):
            values["tags"] = [values["tags"]]
        return super().from_dict(values)


@dataclass
class DockingConfig(ConfigSection):
    seed: int = 42
    run_name: str = "docking-model"
    logger: LoggerConfig = field(default_factory=LoggerConfig)
    protein: ProteinConfig = field(default_factory=ProteinConfig)
    ligand: LigandConfig = field(default_factory=LigandConfig)
    pocket: PocketConfig = field(default_factory=PocketConfig)
    nearby_atoms: NearbyAtomsConfig = field(default_factory=NearbyAtomsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    sigma: SigmaConfig = field(default_factory=SigmaConfig)
    time: TimeConfig = field(default_factory=TimeConfig)
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    source_path: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None):
        values = dict(data or {})
        model_values = values.get("model", {}) or {}
        transforms = values.get("transforms", {}) or {}
        inference_values = dict(values.get("inference") or {})
        protein_values = dict(values.get("protein") or model_values.get("protein") or transforms.get("protein") or model_values)
        ligand_values = dict(values.get("ligand") or model_values.get("ligand") or transforms.get("ligand") or model_values)
        protein_cfg = ProteinConfig.from_dict(protein_values)
        ligand_cfg = LigandConfig.from_dict(ligand_values)

        has_pocket_source = "pocket" in values or "pocket" in transforms
        pocket_values = dict(values.get("pocket") or transforms.get("pocket") or {})
        if not has_pocket_source:
            copy_if_present(inference_values, pocket_values, "pocket_reduction")
            copy_if_present(inference_values, pocket_values, "pocket_radius")
            copy_if_present(inference_values, pocket_values, "pocket_buffer")
            copy_if_present(inference_values, pocket_values, "pocket_min_size")
        if is_interpolation(pocket_values.get("all_atoms")) and "all_atoms" in model_values:
            pocket_values["all_atoms"] = model_values["all_atoms"]
        has_nearby_source = "nearby_atoms" in values or "nearby_atoms" in transforms
        nearby_values = dict(values.get("nearby_atoms") or transforms.get("nearby_atoms") or {})
        if not has_nearby_source:
            copy_if_present(inference_values, nearby_values, "only_nearby_residues_atomic")
            copy_if_present(inference_values, nearby_values, "nearby_residues_selection_mode")
            copy_if_present(inference_values, nearby_values, "nearby_residues_atomic_radius")
            copy_if_present(inference_values, nearby_values, "nearby_residues_atomic_min")
        pocket_cfg = PocketConfig.from_dict(pocket_values)
        nearby_cfg = NearbyAtomsConfig.from_dict(nearby_values)

        sigma_values = dict(values.get("sigma") or transforms.get("sigma_args") or {})
        sigma_values["sidechain_tor_bridge"] = protein_cfg.sidechain_tor_bridge
        sigma_values["sidechain_tor_transform_type"] = protein_cfg.sidechain_tor_transform_type
        sigma_cfg = SigmaConfig.from_dict(sigma_values)

        time_values = dict(values.get("time") or transforms.get("time_args") or {})
        time_cfg = TimeConfig.from_dict(time_values)

        model_values = dict(model_values)
        model_lig_transform_type = model_values.get("lig_transform_type", transforms.get("lig_transform_type", "diffusion"))
        model_values["all_atoms"] = pocket_cfg.all_atoms
        model_values["no_torsion"] = ligand_cfg.no_torsion
        model_values["flexible_backbone"] = protein_cfg.flexible_backbone
        model_values["flexible_sidechains"] = protein_cfg.flexible_sidechains
        model_values["sidechain_tor_bridge"] = protein_cfg.sidechain_tor_bridge
        model_values["use_bb_orientation_feats"] = protein_cfg.use_bb_orientation_feats
        model_values["only_nearby_residues_atomic"] = nearby_cfg.restrict_to_nearby

        transform_values = dict(transforms)
        transform_values["pocket"] = pocket_values
        transform_values["nearby_atoms"] = nearby_values
        transform_values["ligand"] = {"torsions": ligand_cfg.torsions}
        transform_values["protein"] = {
            "flexibility": protein_cfg.flexibility,
            "sidechain_tor_transform_type": protein_cfg.sidechain_tor_transform_type,
            "use_bb_orientation_feats": protein_cfg.use_bb_orientation_feats,
        }
        transform_values["time_args"] = time_values
        transform_values["sigma_args"] = sigma_values
        transform_values["flexible_backbone"] = protein_cfg.flexible_backbone
        transform_values["flexible_sidechains"] = protein_cfg.flexible_sidechains
        transform_values["lig_transform_type"] = model_lig_transform_type
        transform_values["tor_fourier_enabled"] = model_values.get("tor_fourier_enabled", False)
        transforms_cfg = TransformsConfig.from_dict(transform_values)

        sampler_values = dict(values.get("sampler") or {})
        for key in ["inference_steps", "samples_per_complex", "k_samples_per_complex", "batch_size", "ode"]:
            if inference_values.get(key) is not None and not is_interpolation(inference_values.get(key)):
                sampler_values[key] = inference_values[key]
        if inference_values.get("actual_steps") is not None and not is_interpolation(inference_values.get("actual_steps")):
            sampler_values["inference_steps"] = inference_values["actual_steps"]
        for key, value in {
            "all_atoms": pocket_cfg.all_atoms,
            "no_torsion": ligand_cfg.no_torsion,
            "flexible_backbone": protein_cfg.flexible_backbone,
            "flexible_sidechains": protein_cfg.flexible_sidechains,
            "sidechain_tor_bridge": protein_cfg.sidechain_tor_bridge,
            "sidechain_tor_transform_type": protein_cfg.sidechain_tor_transform_type,
            "use_bb_orientation_feats": protein_cfg.use_bb_orientation_feats,
            "sampling_alpha": time_cfg.sampling_alpha,
            "sampling_beta": time_cfg.sampling_beta,
            "bb_tr_bridge_alpha": time_cfg.bb_tr_bridge_alpha,
            "bb_rot_bridge_alpha": time_cfg.bb_rot_bridge_alpha,
            "sc_tor_bridge_alpha": time_cfg.sc_tor_bridge_alpha,
            "lig_transform_type": model_lig_transform_type,
        }.items():
            sampler_values[key] = value
        for key in [
            "reset_sidechain_ve_to_apo_before_randomize",
            "all_atoms",
            "flexible_backbone",
            "flexible_sidechains",
            "diff_temp_sampling_tr",
            "diff_temp_sampling_rot",
            "diff_temp_sampling_tor",
            "diff_temp_psi_tr",
            "diff_temp_psi_rot",
            "diff_temp_psi_tor",
            "diff_temp_sigma_data_tr",
            "diff_temp_sigma_data_rot",
            "diff_temp_sigma_data_tor",
            "flow_temp_scale_0_tr",
            "flow_temp_scale_0_rot",
            "flow_temp_scale_0_tor",
            "flow_temp_scale_1_tr",
            "flow_temp_scale_1_rot",
            "flow_temp_scale_1_tor",
        ]:
            if inference_values.get(key) is not None and not is_interpolation(inference_values.get(key)):
                sampler_values.setdefault(key, inference_values[key])
        if inference_values.get("save_trajectory") is not None:
            if bool(inference_values["save_trajectory"]):
                sampler_values["return_full_trajectory"] = True
            else:
                sampler_values.setdefault("return_full_trajectory", False)

        loss_values = dict(values.get("loss") or {})
        for key, value in {
            "lig_transform_type": model_lig_transform_type,
            "sidechain_tor_bridge": protein_cfg.sidechain_tor_bridge,
            "sidechain_tor_transform_type": protein_cfg.sidechain_tor_transform_type,
            "tor_fourier_enabled": model_values.get("tor_fourier_enabled", False),
            "sc_tor_fourier_enabled": model_values.get("sc_tor_fourier_enabled", False),
            "tor_sc_coupling_enabled": model_values.get("tor_sc_coupling_enabled", False),
            "flexible_backbone": protein_cfg.flexible_backbone,
            "flexible_sidechains": protein_cfg.flexible_sidechains,
            "no_torsion": ligand_cfg.no_torsion,
            "all_atoms": pocket_cfg.all_atoms,
            "tr_sigma_min": sigma_cfg.tr_sigma_min,
            "tr_sigma_max": sigma_cfg.tr_sigma_max,
            "rot_sigma_min": sigma_cfg.rot_sigma_min,
            "rot_sigma_max": sigma_cfg.rot_sigma_max,
        }.items():
            loss_values[key] = value

        if inference_values.get("esm_embeddings_path") is None and model_values.get("esm_embeddings_path") is not None:
            inference_values["esm_embeddings_path"] = model_values["esm_embeddings_path"]
        inference_values["pocket_reduction"] = pocket_cfg.enabled
        inference_values["pocket_radius"] = pocket_cfg.radius
        inference_values["pocket_buffer"] = pocket_cfg.buffer
        inference_values["pocket_min_size"] = pocket_cfg.min_size
        inference_values["only_nearby_residues_atomic"] = nearby_cfg.restrict_to_nearby
        inference_values["nearby_residues_selection_mode"] = nearby_cfg.selection_mode
        inference_values["nearby_residues_atomic_radius"] = nearby_cfg.radius
        inference_values["nearby_residues_atomic_min"] = nearby_cfg.min_atoms
        inference_values["flexible_backbone"] = protein_cfg.flexible_backbone
        inference_values["flexible_sidechains"] = protein_cfg.flexible_sidechains
        inference_values["all_atoms"] = pocket_cfg.all_atoms

        return cls(
            seed=values.get("seed", 42),
            run_name=values.get("run_name", "docking-model"),
            logger=LoggerConfig.from_dict(values.get("logger")),
            protein=protein_cfg,
            ligand=ligand_cfg,
            pocket=pocket_cfg,
            nearby_atoms=nearby_cfg,
            model=ModelConfig.from_dict(model_values),
            sigma=sigma_cfg,
            time=time_cfg,
            transforms=transforms_cfg,
            sampler=SamplerConfig.from_dict(sampler_values),
            data=DataConfig.from_dict(values.get("data")),
            inference=InferenceConfig.from_dict(inference_values),
            loss=LossConfig.from_dict(loss_values),
            training=TrainingConfig.from_dict(values.get("training")),
            source_path=values.get("source_path"),
        )


def copy_key(values: dict[str, Any], old: str, new: str) -> None:
    if old in values:
        values[new] = values[old]


def copy_if_present(source: dict[str, Any], target: dict[str, Any], key: str) -> None:
    if source.get(key) is not None and not is_interpolation(source.get(key)):
        target[key] = source[key]


def validate_sigma_bounds(name: str, sigma_min: float, sigma_max: float) -> None:
    if sigma_min <= 0 or sigma_max <= 0:
        raise ValueError(f"{name} sigma bounds must be positive.")
    if sigma_min > sigma_max:
        raise ValueError(f"{name} sigma_min cannot exceed sigma_max.")


def is_interpolation(value: Any) -> bool:
    return isinstance(value, str) and value.strip().startswith("${") and value.strip().endswith("}")
