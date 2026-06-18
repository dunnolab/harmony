from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from docking_model.geometry.manifolds import so3, torus
from docking_model.geometry.ops import rigid_transform_kabsch
from docking_model.metrics.docking import pli_lddt_score


class DockingLoss(nn.Module):
    """Docking score-matching objective."""

    def __init__(self, args, t_to_sigma):
        super().__init__()
        self.args = args
        self.t_to_sigma = t_to_sigma
        self.loss_weights = {
            "tr_loss": args.tr_weight,
            "rot_loss": args.rot_weight,
            "tor_loss": args.tor_weight,
            "bb_tr_loss": args.bb_tr_weight,
            "bb_rot_loss": args.bb_rot_weight,
            "sc_tor_loss": args.sc_tor_weight,
            "pli_lddt_loss": getattr(args, "confidence_weight", 0.0),
            "affinity_loss": getattr(args, "affinity_weight", 0.0),
            "tor_fourier_reg_loss": getattr(args, "tor_fourier_reg_weight", 0.0),
            "sc_tor_fourier_reg_loss": getattr(args, "sc_tor_fourier_reg_weight", 0.0),
            "sc_tor_gate_reg_loss": getattr(args, "sc_tor_gate_reg_weight", 0.0),
            "tor_sc_coupling_reg_loss": getattr(args, "tor_sc_coupling_reg_weight", 0.0),
        }

    def forward(self, outputs: dict[str, torch.Tensor], batch, apply_mean: bool = True) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        t_dict = self.batch_t_dict(batch)
        sigma_dict = self.t_to_sigma(t_dict)

        loss_terms: dict[str, torch.Tensor] = {}
        if all(key in outputs for key in ("tr_pred", "rot_pred", "tor_pred")):
            loss_terms.update(self.compute_ligand_loss(outputs, batch, t_dict, sigma_dict, apply_mean=apply_mean))
            loss_terms.update(self.compute_protein_loss(outputs, batch, t_dict, sigma_dict, apply_mean=apply_mean))
        loss_terms.update(self.compute_confidence_loss(outputs, batch, t_dict, sigma_dict, apply_mean=apply_mean))
        loss_terms.update(self.compute_affinity_loss(outputs, batch))

        if not loss_terms:
            raise ValueError("No compatible loss terms were found for this batch.")

        reference = self.reference_tensor(outputs, batch)
        total = reference.new_tensor(0.0)
        metrics: dict[str, torch.Tensor] = {}
        for name, value in loss_terms.items():
            value = self.as_tensor(value, reference)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} is non-finite during loss computation.")
            metrics[name] = value.detach().mean()
            if "base_loss" in name:
                continue
            weight = float(self.loss_weights.get(name, 0.0))
            if weight != 0.0:
                total = total + weight * value.mean()
        metrics["loss"] = total.detach()
        return total, metrics

    def compute_ligand_loss(self, outputs, batch, t_dict, sigma_dict, apply_mean: bool = True) -> dict[str, torch.Tensor]:
        if getattr(self.args, "lig_transform_type", "diffusion") != "diffusion":
            raise NotImplementedError("Only diffusion ligand losses are supported.")

        tr_pred = outputs["tr_pred"]
        rot_pred = outputs["rot_pred"]
        tr_score = batch.tr_score
        rot_score = batch.rot_score
        tr_sigma = torch.as_tensor(sigma_dict["tr_sigma"], device=tr_pred.device, dtype=tr_pred.dtype).view(-1, 1)
        rot_sigma = torch.as_tensor(sigma_dict["rot_sigma"], device=rot_pred.device, dtype=rot_pred.dtype).view(-1)
        loss_weight = self.loss_weight(batch, tr_pred.device, tr_pred.dtype)

        tr_loss = (tr_pred - tr_score) ** 2 * tr_sigma**2
        tr_base_loss = tr_score**2 * tr_sigma**2
        rot_norm = self.so3_score_norm(rot_sigma, rot_pred)
        rot_loss = ((rot_pred - rot_score) / rot_norm) ** 2
        rot_base_loss = (rot_score / rot_norm) ** 2

        tr_loss = self.reduce_graph_vector(tr_loss, tr_base=tr_score, loss_weight=loss_weight, apply_mean=apply_mean)
        tr_base_loss = self.reduce_graph_vector(tr_base_loss, tr_base=tr_score, loss_weight=loss_weight, apply_mean=apply_mean)
        rot_loss = self.reduce_graph_vector(rot_loss, tr_base=rot_score, loss_weight=loss_weight, apply_mean=apply_mean)
        rot_base_loss = self.reduce_graph_vector(rot_base_loss, tr_base=rot_score, loss_weight=loss_weight, apply_mean=apply_mean)

        losses = {
            "tr_loss": tr_loss,
            "rot_loss": rot_loss,
            "tr_base_loss": tr_base_loss,
            "rot_base_loss": rot_base_loss,
        }

        tor_loss, tor_base_loss, tor_fourier_reg = self.compute_torsion_loss(outputs, batch, rot_loss, apply_mean)
        losses["tor_loss"] = tor_loss
        losses["tor_base_loss"] = tor_base_loss
        if getattr(self.args, "tor_fourier_enabled", False):
            losses["tor_fourier_reg_loss"] = tor_fourier_reg
        if getattr(self.args, "tor_sc_coupling_enabled", False):
            losses.update(self.coupling_diagnostics(outputs, tr_pred))
        return losses

    def compute_protein_loss(self, outputs, batch, t_dict, sigma_dict, apply_mean: bool = True) -> dict[str, torch.Tensor]:
        reference = outputs.get("sc_tor_pred", outputs["rot_pred"])
        losses: dict[str, torch.Tensor] = {}

        sc_tor_loss, sc_tor_base_loss, sc_tor_fourier_reg, sc_tor_gate_reg = self.compute_sidechain_loss(
            outputs, batch, t_dict, reference, apply_mean
        )
        losses["sc_tor_loss"] = sc_tor_loss
        if getattr(self.args, "sc_tor_fourier_enabled", False):
            losses["sc_tor_fourier_reg_loss"] = sc_tor_fourier_reg
            losses["sc_tor_gate_reg_loss"] = sc_tor_gate_reg
        losses["sc_tor_base_loss"] = sc_tor_base_loss

        bb_tr_loss, bb_rot_loss, bb_tr_base_loss, bb_rot_base_loss = self.compute_backbone_loss(
            outputs, batch, t_dict, reference, apply_mean
        )
        losses["bb_tr_loss"] = bb_tr_loss
        losses["bb_rot_loss"] = bb_rot_loss
        losses["bb_tr_base_loss"] = bb_tr_base_loss
        losses["bb_rot_base_loss"] = bb_rot_base_loss

        if getattr(self.args, "tor_sc_coupling_enabled", False):
            losses.update(self.sidechain_coupling_diagnostics(outputs, reference))
        return losses

    def compute_affinity_loss(self, outputs, batch) -> dict[str, torch.Tensor]:
        if "affinity_pred" not in outputs or not hasattr(batch, "affinity"):
            return {}
        target = batch.affinity.view(-1).float()
        mask = getattr(batch, "affinity_mask", torch.ones_like(target)).view(-1).to(target.device) > 0.5
        if not mask.any():
            return {"affinity_loss": target.new_tensor(0.0)}
        pred = outputs["affinity_pred"].view(-1).float()
        return {"affinity_loss": F.mse_loss(pred[mask], target[mask])}

    def compute_confidence_loss(self, outputs, batch, t_dict, sigma_dict, apply_mean: bool = True) -> dict[str, torch.Tensor]:
        del sigma_dict, apply_mean
        if "filtering_pred" not in outputs:
            return {}

        primary_logits, aux_logits = self.split_confidence_predictions(outputs)
        time_weights = self.confidence_time_weights(t_dict, primary_logits)
        confidence_target = str(getattr(self.args, "confidence_target", "pli_lddt")).lower()
        losses: dict[str, torch.Tensor] = {"confidence_time_weight_mean": time_weights.mean()}

        if confidence_target in {"rmsd_lt2", "rmsd_binary", "ligand_rmsd_lt2", "ligand_rmsd_binary"}:
            rmsd_values = self.aligned_ligand_rmsd_targets(batch, primary_logits.device)
            threshold = float(getattr(self.args, "confidence_rmsd_threshold", 2.0))
            cls_targets = (rmsd_values < threshold).float().detach()
            cls_loss = self.bce_loss(primary_logits, cls_targets, time_weights)
            pred_prob = torch.sigmoid(primary_logits)
            losses.update(
                {
                    "pli_lddt_loss": cls_loss,
                    "confidence_cls_pos_rate": cls_targets.mean(),
                    "confidence_cls_prob_mean": pred_prob.mean(),
                    "confidence_cls_acc": ((pred_prob >= 0.5).float() == cls_targets).float().mean(),
                    "confidence_cls_rmsd_mean": torch.where(torch.isfinite(rmsd_values), rmsd_values, torch.zeros_like(rmsd_values)).mean(),
                }
            )
            return losses

        pli_true, aligned_rmsd = self.pli_lddt_targets(batch, need_rmsd=getattr(self.args, "confidence_aux_rmsd_lt2_weight", 0.0) > 0)
        pred_conf = torch.sigmoid(primary_logits).view(-1)
        pli_loss = self.weighted_mean(torch.abs(pred_conf - pli_true.view(-1)), time_weights)
        total_confidence_loss = pli_loss
        losses.update(
            {
                "pli_lddt_loss": total_confidence_loss,
                "confidence_primary_pli_lddt_loss": pli_loss,
                "pli_lddt_baseline_mean": pli_true.mean(),
                "pli_lddt_pearson": self.pearson(pred_conf, pli_true),
                "pli_lddt_spearman": self.spearman(pred_conf, pli_true),
            }
        )

        aux_weight = float(getattr(self.args, "confidence_aux_rmsd_lt2_weight", 0.0))
        if aux_weight > 0:
            if aux_logits is None:
                raise ValueError("confidence_aux_rmsd_lt2_weight > 0 requires at least two confidence outputs.")
            threshold = float(getattr(self.args, "confidence_rmsd_threshold", 2.0))
            rmsd_targets = (aligned_rmsd < threshold).float().detach()
            aux_loss = self.bce_loss(aux_logits, rmsd_targets, time_weights)
            aux_prob = torch.sigmoid(aux_logits)
            total_confidence_loss = pli_loss + aux_weight * aux_loss
            losses.update(
                {
                    "pli_lddt_loss": total_confidence_loss,
                    "confidence_aux_rmsd_lt2_loss": aux_loss,
                    "confidence_aux_rmsd_lt2_acc": ((aux_prob >= 0.5).float() == rmsd_targets).float().mean(),
                    "confidence_aux_rmsd_lt2_pos_rate": rmsd_targets.mean(),
                    "confidence_aux_rmsd_lt2_prob_mean": aux_prob.mean(),
                    "confidence_aligned_rmsd_mean": torch.where(
                        torch.isfinite(aligned_rmsd), aligned_rmsd, torch.zeros_like(aligned_rmsd)
                    ).mean(),
                }
            )
        return losses

    def compute_torsion_loss(self, outputs, batch, reference, apply_mean: bool):
        if getattr(self.args, "no_torsion", False):
            zero = reference.new_zeros(1 if apply_mean else reference.shape[:1])
            return zero, zero, zero

        tor_pred = outputs["tor_pred"]
        if tor_pred.numel() == 0:
            zero = reference.new_zeros(1 if apply_mean else batch.num_graphs)
            return zero, zero, zero

        tor_score = batch.tor_score
        edge_sigma = self.edge_sigma(batch.tor_sigma_edge, tor_pred)
        norm2 = torch.as_tensor(torus.score_norm(edge_sigma.detach().cpu().numpy()), device=tor_pred.device, dtype=tor_pred.dtype).clamp_min(1e-8)
        tor_loss = (tor_pred - tor_score) ** 2 / norm2
        tor_base_loss = (tor_score**2 / norm2).detach()
        tor_fourier_reg = self.fourier_reg(outputs, "tor", tor_loss)
        index = batch["ligand"].batch[batch["ligand", "lig_bond", "ligand"].edge_index[0][batch["ligand"].edge_mask]]
        return (
            self.reduce_one_edge_aux(tor_loss, index, batch, apply_mean),
            self.reduce_one_edge_aux(tor_base_loss, index, batch, apply_mean),
            self.reduce_one_edge_aux(tor_fourier_reg, index, batch, apply_mean),
        )

    def compute_sidechain_loss(self, outputs, batch, t_dict, reference, apply_mean: bool):
        if not getattr(self.args, "flexible_sidechains", False):
            zero = reference.new_zeros(1 if apply_mean else batch.num_graphs)
            return zero, zero, zero, zero

        sc_pred = outputs["sc_tor_pred"]
        if sc_pred.numel() == 0:
            zero = reference.new_zeros(1 if apply_mean else batch.num_graphs)
            return zero, zero, zero, zero

        sc_score = batch.sidechain_tor_score
        edge_sigma = self.edge_sigma(batch.sidechain_tor_sigma_edge, sc_pred)
        norm2 = torch.as_tensor(torus.score_norm(edge_sigma.detach().cpu().numpy()), device=sc_pred.device, dtype=sc_pred.dtype).clamp_min(1e-8)
        if self.sidechain_tor_bridge:
            index = self.sidechain_batch_index(batch, sc_pred)
            t_sc = t_dict["sc_tor"][index].to(sc_pred.device, sc_pred.dtype)
            sc_loss = ((sc_pred - sc_score) * ((1.0 - t_sc).detach() + 1e-7)) ** 2 / norm2
        else:
            index = self.sidechain_batch_index(batch, sc_pred)
            sc_loss = (sc_pred - sc_score) ** 2 / norm2
        sc_base_loss = (sc_score**2 / norm2).detach()
        sc_fourier_reg = self.fourier_reg(outputs, "sc_tor", sc_loss)
        gate = outputs.get("sc_tor_fourier_gate")
        sc_gate_reg = (gate - 1.0).square().mean(dim=-1) if torch.is_tensor(gate) and gate.numel() > 0 else sc_loss.new_zeros(sc_loss.shape)
        return (
            self.reduce_one_edge_aux(sc_loss, index, batch, apply_mean),
            self.reduce_one_edge_aux(sc_base_loss, index, batch, apply_mean),
            self.reduce_one_edge_aux(sc_fourier_reg, index, batch, apply_mean),
            self.reduce_one_edge_aux(sc_gate_reg, index, batch, apply_mean),
        )

    def compute_backbone_loss(self, outputs, batch, t_dict, reference, apply_mean: bool):
        if not getattr(self.args, "flexible_backbone", False):
            zero = reference.new_zeros(1 if apply_mean else batch.num_graphs)
            return zero, zero, zero, zero

        bb_tr_pred = outputs["bb_tr_pred"]
        bb_rot_pred = outputs["bb_rot_pred"]
        bb_tr_drift = batch.bb_tr_drift
        bb_rot_drift = batch.bb_rot_drift
        if bb_tr_pred.ndim == bb_tr_drift.ndim + 1:
            bb_tr_pred = bb_tr_pred.transpose(0, 1)
            bb_rot_pred = bb_rot_pred.transpose(0, 1)

        receptor_batch = batch["receptor"].batch
        t_bb_tr = t_dict["bb_tr"][receptor_batch].to(bb_tr_pred.device, bb_tr_pred.dtype)
        t_bb_rot = t_dict["bb_rot"][receptor_batch].to(bb_rot_pred.device, bb_rot_pred.dtype)
        tr_norm = (1.0 - t_bb_tr).detach().square()
        rot_norm = (1.0 - t_bb_rot).detach().square()
        bb_tr_loss = ((bb_tr_pred - bb_tr_drift) ** 2).sum(-1).float() * tr_norm
        bb_rot_loss = ((bb_rot_pred - bb_rot_drift) ** 2).sum(-1).float() * rot_norm
        bb_tr_base_loss = (bb_tr_drift**2).sum(-1).float() * tr_norm
        bb_rot_base_loss = (bb_rot_drift**2).sum(-1).float() * rot_norm
        if bb_tr_loss.ndim == bb_tr_base_loss.ndim + 1:
            bb_tr_loss = bb_tr_loss.mean(0)
            bb_rot_loss = bb_rot_loss.mean(0)
        return (
            self.reduce_node_loss(bb_tr_loss, receptor_batch, batch, apply_mean),
            self.reduce_node_loss(bb_rot_loss, receptor_batch, batch, apply_mean),
            self.reduce_node_loss(bb_tr_base_loss, receptor_batch, batch, apply_mean),
            self.reduce_node_loss(bb_rot_base_loss, receptor_batch, batch, apply_mean),
        )

    def batch_t_dict(self, batch) -> dict[str, torch.Tensor]:
        keys = ["tr", "rot", "tor", "t"]
        if getattr(self.args, "flexible_sidechains", False):
            keys.append("sc_tor")
        if getattr(self.args, "flexible_backbone", False):
            keys.extend(["bb_tr", "bb_rot"])
        return {key: batch.complex_t[key] for key in keys if key in batch.complex_t and batch.complex_t[key] is not None}

    @property
    def sidechain_tor_bridge(self) -> bool:
        if hasattr(self.args, "sidechain_tor_transform_type"):
            return str(self.args.sidechain_tor_transform_type).lower() == "bridge"
        return bool(getattr(self.args, "sidechain_tor_bridge", False))

    @staticmethod
    def reference_tensor(outputs, batch):
        for value in outputs.values():
            if torch.is_tensor(value):
                return value
        return batch["ligand"].pos

    @staticmethod
    def as_tensor(value, reference):
        if torch.is_tensor(value):
            return value.to(reference.device)
        return reference.new_tensor(value)

    @staticmethod
    def so3_score_norm(rot_sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        norm = torch.as_tensor(so3.score_norm(rot_sigma.detach().cpu()), device=reference.device, dtype=reference.dtype)
        return norm.view(-1, 1).clamp_min(1e-8)

    @staticmethod
    def loss_weight(batch, device, dtype) -> torch.Tensor | None:
        if not hasattr(batch, "loss_weight"):
            return None
        value = batch.loss_weight
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        return value.to(device=device, dtype=dtype).view(-1)

    @staticmethod
    def edge_sigma(value: Any, reference: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=reference.device, dtype=reference.dtype).view(-1)
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value).to(device=reference.device, dtype=reference.dtype).view(-1)
        if isinstance(value, (list, tuple)):
            parts = []
            for item in value:
                if item is None:
                    continue
                if torch.is_tensor(item):
                    parts.append(item.detach().cpu().numpy())
                else:
                    parts.append(np.asarray(item))
            if not parts:
                return reference.new_empty(0)
            return torch.from_numpy(np.concatenate(parts)).to(device=reference.device, dtype=reference.dtype).view(-1)
        return torch.as_tensor(value, device=reference.device, dtype=reference.dtype).view(-1)

    @staticmethod
    def reduce_graph_vector(values, tr_base, loss_weight, apply_mean: bool):
        if loss_weight is not None:
            values = values * loss_weight.view(-1, 1)
            denom = tr_base.size(1) * loss_weight.sum() + 1.0e-4
            return values.sum() / denom if apply_mean else values.sum(dim=1) / (tr_base.size(1) + 1.0e-4)
        return values.mean() if apply_mean else values.mean(dim=1)

    def reduce_one_edge_aux(self, values, index, batch, apply_mean: bool):
        if values.numel() == 0:
            return values.new_zeros(1 if apply_mean else batch.num_graphs)
        loss_weight = self.loss_weight(batch, values.device, values.dtype)
        if apply_mean:
            if loss_weight is not None:
                edge_weight = loss_weight[index]
                return (values * edge_weight).sum() / (edge_weight.sum() + 1.0e-4)
            return values.mean()
        summed = values.new_zeros(batch.num_graphs)
        counts = values.new_zeros(batch.num_graphs)
        edge_weight = loss_weight[index] if loss_weight is not None else values.new_ones(values.shape)
        summed.index_add_(0, index, values * edge_weight)
        counts.index_add_(0, index, edge_weight)
        return summed / (counts + 1.0e-4)

    def reduce_node_loss(self, values, index, batch, apply_mean: bool):
        loss_weight = self.loss_weight(batch, values.device, values.dtype)
        if apply_mean:
            if loss_weight is not None:
                node_weight = loss_weight[index]
                return (values * node_weight).sum() / (node_weight.sum() + 1.0e-4)
            return values.mean()
        summed = values.new_zeros(batch.num_graphs)
        counts = values.new_zeros(batch.num_graphs)
        node_weight = loss_weight[index] if loss_weight is not None else values.new_ones(values.shape)
        summed.index_add_(0, index, values * node_weight)
        counts.index_add_(0, index, node_weight)
        return summed / (counts + 1.0e-4)

    def sidechain_batch_index(self, batch, reference):
        bonds = batch["atom", "atom_bond", "atom"].edge_index[:, batch["atom", "atom_bond", "atom"].edge_mask]
        if bonds.numel() == 0:
            return torch.empty(0, device=reference.device, dtype=torch.long)
        source = bonds[0]
        if hasattr(batch["atom"], "orig_batch"):
            return batch["atom"].orig_batch[source].to(reference.device)
        return batch["atom"].batch[source].to(reference.device)

    @staticmethod
    def fourier_reg(outputs, prefix: str, reference: torch.Tensor):
        a = outputs.get(f"{prefix}_fourier_a")
        b = outputs.get(f"{prefix}_fourier_b")
        if not (torch.is_tensor(a) and torch.is_tensor(b) and a.numel() > 0 and b.numel() > 0):
            return reference.new_zeros(reference.shape)
        modes = torch.arange(1, a.shape[1] + 1, device=a.device, dtype=a.dtype).view(1, -1)
        return ((modes**2) * (a.square() + b.square())).sum(dim=-1)

    @staticmethod
    def coupling_diagnostics(outputs, reference):
        losses: dict[str, torch.Tensor] = {}
        terms = [
            outputs.get("tor_sc_coupling_unary_coeff_lig"),
            outputs.get("tor_sc_coupling_unary_coeff_sc"),
            outputs.get("tor_sc_coupling_pair_coeff_lig"),
            outputs.get("tor_sc_coupling_pair_coeff_sc"),
        ]
        non_empty = [value.square().mean() for value in terms if torch.is_tensor(value) and value.numel() > 0]
        losses["tor_sc_coupling_reg_loss"] = torch.stack(non_empty).mean() if non_empty else reference.new_tensor(0.0)
        for key in [
            "tor_pred_base_rms",
            "tor_pred_unary_rms",
            "tor_pred_pair_rms",
            "tor_pred_unary_rel",
            "tor_pred_pair_rel",
            "tor_sc_local_neighbors_lig_mean",
            "tor_sc_local_coverage_lig",
        ]:
            value = outputs.get(key)
            losses[key] = value.detach().mean() if torch.is_tensor(value) else reference.new_tensor(0.0)
        return losses

    @staticmethod
    def sidechain_coupling_diagnostics(outputs, reference):
        losses: dict[str, torch.Tensor] = {}
        for key in [
            "sc_tor_pred_base_rms",
            "sc_tor_pred_unary_rms",
            "sc_tor_pred_pair_rms",
            "sc_tor_pred_unary_rel",
            "sc_tor_pred_pair_rel",
            "tor_sc_local_neighbors_sc_mean",
            "tor_sc_local_coverage_sc",
        ]:
            value = outputs.get(key)
            losses[key] = value.detach().mean() if torch.is_tensor(value) else reference.new_tensor(0.0)
        return losses

    @staticmethod
    def split_confidence_predictions(outputs):
        pred = outputs["filtering_pred"].float()
        if pred.ndim == 1:
            return pred, None
        return pred[:, 0], pred[:, 1] if pred.size(1) > 1 else None

    def confidence_time_weights(self, t_dict, reference):
        mode = str(getattr(self.args, "confidence_time_weighting", "none")).lower()
        graph_t = t_dict.get("t", t_dict["tr"]).view(-1).to(device=reference.device, dtype=reference.dtype)
        if mode == "none":
            return torch.ones_like(graph_t)
        if mode == "late_exp":
            power = float(getattr(self.args, "confidence_time_power", 2.0))
            return (1.0 - graph_t.clamp(0.0, 1.0)).pow(power)
        if mode == "late_only":
            cutoff = float(getattr(self.args, "confidence_time_cutoff", 0.3))
            return (graph_t <= cutoff).to(reference.dtype)
        raise ValueError(f"Unsupported confidence_time_weighting={mode}.")

    def bce_loss(self, logits, targets, weights):
        pos_weight = getattr(self.args, "confidence_pos_weight", None)
        pos_weight_tensor = logits.new_tensor(float(pos_weight)) if pos_weight is not None else None
        values = F.binary_cross_entropy_with_logits(logits.view(-1), targets.view(-1), pos_weight=pos_weight_tensor, reduction="none")
        return self.weighted_mean(values, weights)

    @staticmethod
    def weighted_mean(values, weights):
        values = values.view(-1).float()
        weights = weights.view(-1).to(values.device, dtype=values.dtype)
        return (values * weights).sum() / weights.sum().clamp_min(1.0e-6)

    def aligned_ligand_rmsd_targets(self, batch, device):
        values = []
        for idx in range(batch.num_graphs):
            lig_mask = batch["ligand"].batch == idx
            atom_mask = batch["atom"].batch == idx
            values.append(
                self.aligned_ligand_rmsd(
                    batch,
                    idx,
                    batch["atom"].pos[atom_mask],
                    batch["ligand"].pos[lig_mask],
                    self.true_atom_pos(batch)[atom_mask],
                    batch["ligand"].orig_pos[lig_mask],
                )
            )
        return torch.stack(values).to(device)

    def pli_lddt_targets(self, batch, need_rmsd: bool):
        atom_pos_t = batch["atom"].pos
        ligand_pos_t = batch["ligand"].pos
        true_atom_pos = self.true_atom_pos(batch)
        true_ligand_pos = batch["ligand"].orig_pos
        pli_values = []
        rmsd_values = []
        for idx in range(batch.num_graphs):
            atom_mask = batch["atom"].batch == idx
            ligand_mask = batch["ligand"].batch == idx
            rec_pred = atom_pos_t[atom_mask]
            lig_pred = ligand_pos_t[ligand_mask]
            rec_true = true_atom_pos[atom_mask]
            lig_true = true_ligand_pos[ligand_mask]
            if rec_pred.numel() == 0 or lig_pred.numel() == 0:
                pli_values.append(atom_pos_t.new_tensor(0.0))
                if need_rmsd:
                    rmsd_values.append(atom_pos_t.new_tensor(float("inf")))
                continue
            try:
                pli_values.append(
                    pli_lddt_score(
                        rec_coords_predicted=rec_pred.unsqueeze(0),
                        lig_coords_predicted=lig_pred.unsqueeze(0),
                        rec_coords_true=rec_true.unsqueeze(0),
                        lig_coords_true=lig_true.unsqueeze(0),
                    )[0]
                )
            except ValueError:
                pli_values.append(atom_pos_t.new_tensor(0.0))
            if need_rmsd:
                rmsd_values.append(self.aligned_ligand_rmsd(batch, idx, rec_pred, lig_pred, rec_true, lig_true))
        pli = torch.stack(pli_values).detach()
        rmsd = torch.stack(rmsd_values).detach() if need_rmsd else None
        return pli, rmsd

    @staticmethod
    def true_atom_pos(batch):
        atom = batch["atom"]
        if "pos_sc_matched" in atom:
            return atom.pos_sc_matched
        if "orig_holo_pos" in atom:
            return atom.orig_holo_pos
        return atom.pos

    def aligned_ligand_rmsd(self, batch, sample_idx: int, rec_pred, ligand_pred, rec_true, ligand_true):
        if rec_pred.numel() == 0 or ligand_pred.numel() == 0:
            return ligand_pred.new_tensor(float("inf"))
        align_selector = None
        if hasattr(batch["atom"], "nearby_atoms") and batch["atom"].nearby_atoms.numel() == batch["atom"].batch.numel():
            atom_mask = batch["atom"].batch == sample_idx
            local = batch["atom"].nearby_atoms[atom_mask]
            if local.numel() == rec_pred.size(0) and local.any():
                align_selector = local
        if align_selector is None and rec_pred.size(0) >= 3:
            align_selector = torch.ones(rec_pred.size(0), device=rec_pred.device, dtype=torch.bool)
        if align_selector is not None and align_selector.sum() >= 3:
            try:
                rot, trans = rigid_transform_kabsch(rec_pred[align_selector].unsqueeze(0), rec_true[align_selector].unsqueeze(0))
                ligand_pred = torch.matmul(rot[0], ligand_pred.transpose(0, 1)).transpose(0, 1) + trans[0]
            except RuntimeError:
                pass
        return torch.sqrt(torch.mean(torch.sum((ligand_pred - ligand_true) ** 2, dim=-1)))

    @staticmethod
    def pearson(pred, true):
        pred = pred.float().reshape(-1)
        true = true.float().reshape(-1)
        pred = pred - pred.mean()
        true = true - true.mean()
        denom = torch.sqrt((pred * pred).sum() * (true * true).sum()).clamp_min(1.0e-8)
        return (pred * true).sum() / denom

    def spearman(self, pred, true):
        pred_rank = torch.argsort(torch.argsort(pred.float().reshape(-1))).float()
        true_rank = torch.argsort(torch.argsort(true.float().reshape(-1))).float()
        return self.pearson(pred_rank, true_rank)
