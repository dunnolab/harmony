from __future__ import annotations

import contextlib
import copy
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from docking_model.metrics.docking import compute_valinf_metrics, select_best_prediction_by_ligand_rmsd
from docking_model.models.optim.ema import ExponentialMovingAverage
from docking_model.runtime.distributed import any_rank_has, all_gather_object, is_distributed, is_main_process, unwrap_model
from docking_model.runtime.checkpoint import save_checkpoint
from docking_model.sampling.engine import SamplingEngine, SamplingResult


@dataclass
class EpochResult:
    loss: float
    steps: int
    metrics: dict[str, float]


class DockingEngine:
    """Plain PyTorch runtime for training, validation, and inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        loss_fn: Callable | None,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device | str,
        sampler: SamplingEngine | None = None,
        gradient_clip_norm: float | None = None,
        scheduler=None,
        training_cfg=None,
        logger=None,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = torch.device(device)
        self.sampler = sampler
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.training_cfg = training_cfg
        self.logger = logger
        self.model.to(self.device)
        self.raw_model = unwrap_model(self.model)

        precision = str(getattr(self.training_cfg, "precision", "fp32") or "fp32").lower()
        if precision in {"amp", "fp16", "float16", "16", "16-mixed", "fp16-mixed"}:
            self.autocast_dtype = torch.float16
        elif precision in {"bf16", "bfloat16", "bf16-mixed"}:
            self.autocast_dtype = torch.bfloat16
        else:
            self.autocast_dtype = None

        self.use_amp = self.device.type == "cuda" and self.autocast_dtype is not None
        self.use_grad_scaler = self.device.type == "cuda" and self.autocast_dtype == torch.float16
        self.check_nan_grads = bool(getattr(self.training_cfg, "check_nan_grads", False))
        self.except_on_nan_grads = bool(getattr(self.training_cfg, "except_on_nan_grads", False))
        self.skip_nan_grad_updates = bool(getattr(self.training_cfg, "skip_nan_grad_updates", False))

        try:
            self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        except TypeError:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)
        self.ema = (
            ExponentialMovingAverage(self.model.parameters(), decay=float(getattr(training_cfg, "ema_rate", 0.999)))
            if optimizer is not None
            else None
        )
        self._ema_weights_for_save: dict[str, torch.Tensor] | None = None
        self._best_val_loss = float("inf")
        self._best_inference_metrics: dict[str, float] = {}
        self._global_step = 0

    def train_one_epoch(self, loader) -> EpochResult:
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError("Training requires both loss_fn and optimizer.")

        self.model.train()
        total_loss = 0.0
        steps = 0
        merged_metrics: dict[str, float] = {}

        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad(set_to_none=True)

            with self.autocast_context():
                outputs = self.model(batch)

            bad_outputs = self.nonfinite_tensor_stats(outputs)
            loss = None
            metrics = {}
            loss_error = None

            if not bad_outputs:
                with self.autocast_context():
                    try:
                        loss, metrics = self.loss_fn(outputs, batch)
                    except (FloatingPointError, ValueError) as exc:
                        if not self.is_numeric_exception(exc):
                            raise
                        loss_error = exc

            bad_loss = bool(bad_outputs) or loss_error is not None or loss is None or not bool(torch.isfinite(loss).all().item())
            reason = self.format_nonfinite_stats("model output", bad_outputs) if bad_outputs else str(loss_error) if loss_error is not None else "non-finite loss"
            if self.handle_bad_update(bad_loss, reason):
                self.optimizer.zero_grad(set_to_none=True)
                continue

            self.scaler.scale(loss).backward()

            unscaled = False
            if self.use_grad_scaler and (self.check_nan_grads or self.gradient_clip_norm is not None):
                self.scaler.unscale_(self.optimizer)
                unscaled = True

            bad_gradients = self.has_bad_gradients()
            if self.handle_bad_update(bad_gradients, "non-finite gradients"):
                self.optimizer.zero_grad(set_to_none=True)
                if self.use_grad_scaler and unscaled:
                    self.scaler.update()
                continue
            if bad_gradients and self.except_on_nan_grads:
                raise FloatingPointError("Encountered non-finite gradients.")

            self.check_unused_parameters()

            if self.gradient_clip_norm is not None:
                if self.use_grad_scaler and not unscaled:
                    self.scaler.unscale_(self.optimizer)
                    unscaled = True
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.ema is not None:
                self.ema.update(self.model.parameters())

            total_loss += float(loss.detach().cpu())
            steps += 1
            self._global_step += 1

            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = float(value.detach().cpu())
                merged_metrics[key] = merged_metrics.get(key, 0.0) + float(value)

        return self.finalize_epoch_result(total_loss, steps, merged_metrics)

    @torch.no_grad()
    def validate_loss(self, loader) -> EpochResult:
        if self.loss_fn is None:
            raise ValueError("Validation requires loss_fn.")

        self.model.eval()
        total_loss = 0.0
        steps = 0
        merged_metrics: dict[str, float] = {}

        for batch in loader:
            batch = batch.to(self.device)

            with self.autocast_context():
                outputs = self.model(batch)

            bad_outputs = self.nonfinite_tensor_stats(outputs)
            if bad_outputs:
                logging.warning("Skipping validation batch because %s.", self.format_nonfinite_stats("model output", bad_outputs))
                continue

            with self.autocast_context():
                try:
                    loss, metrics = self.loss_fn(outputs, batch)
                except (FloatingPointError, ValueError) as exc:
                    if not self.is_numeric_exception(exc):
                        raise
                    logging.warning("Skipping validation batch because %s.", exc)
                    continue

            if not bool(torch.isfinite(loss).all().item()):
                logging.warning("Skipping validation batch because loss is non-finite.")
                continue

            total_loss += float(loss.detach().cpu())
            steps += 1

            for key, value in metrics.items():
                if torch.is_tensor(value):
                    value = float(value.detach().cpu())
                merged_metrics[key] = merged_metrics.get(key, 0.0) + float(value)

        return self.finalize_epoch_result(total_loss, steps, merged_metrics)

    @torch.no_grad()
    def predict(self, loader) -> list[SamplingResult]:
        if self.sampler is None:
            raise ValueError("Inference requires a SamplingEngine.")

        results = []
        for batch in loader:
            data_list = [batch.to("cpu")]
            results.append(self.sampler.generate(data_list=data_list, model=self.raw_model, device=self.device))
        return results

    @torch.no_grad()
    def validate_inference(self, loader) -> tuple[list[SamplingResult], dict[str, float]]:
        if self.sampler is None:
            raise ValueError("Validation inference requires a SamplingEngine.")

        metric_sums: dict[str, float] = {}
        metric_counts: dict[str, int] = {}
        result_count = 0

        for batch in loader:
            reference = batch.to("cpu")
            result = self.generate_validation_sample(reference)
            result_count += 1

            batch_metrics = {
                f"valinf_{key}": value
                for key, value in compute_valinf_metrics(reference, result.predictions).items()
            }
            for key, value in batch_metrics.items():
                if value != value:
                    continue
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                metric_counts[key] = metric_counts.get(key, 0) + 1
            del result

        if is_distributed():
            gathered = all_gather_object(
                {"sums": metric_sums, "counts": metric_counts, "result_count": result_count}
            )
            metric_sums = {}
            metric_counts = {}
            result_count = 0
            for item in gathered:
                result_count += int(item["result_count"])
                for key, value in item["sums"].items():
                    metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
                for key, value in item["counts"].items():
                    metric_counts[key] = metric_counts.get(key, 0) + int(value)

        metrics = {key: metric_sums[key] / metric_counts[key] for key in metric_sums}
        metrics["valinf_complexes"] = float(result_count)
        return [], metrics

    def fit(
        self,
        train_loader,
        val_loader=None,
        inference_loader=None,
        epochs: int = 1,
        checkpoint_dir: str | Path | None = None,
    ) -> list[dict]:
        history = []
        checkpoint_path = Path(checkpoint_dir).expanduser() if checkpoint_dir else None
        if checkpoint_path is not None:
            if is_main_process():
                checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(epochs):
            self.set_loader_epoch(train_loader, epoch)
            train_result = self.train_one_epoch(train_loader)

            with self.validation_weight_scope():
                self.set_loader_epoch(val_loader, epoch)
                val_result = self.validate_loss(val_loader) if val_loader is not None else None
                inference_result = None
                inference_metrics: dict[str, float] = {}
                inference_batches = 0

                if self.should_run_inference(epoch, inference_loader):
                    inference_result, inference_metrics = self.validate_inference(inference_loader)
                    inference_batches = int(inference_metrics.get("valinf_complexes", len(inference_result)))
                    inference_result = None
                    if val_result is not None:
                        val_result.metrics.update(inference_metrics)

            history.append(
                {
                    "epoch": epoch,
                    "train": train_result,
                    "val": val_result,
                    "inference": inference_result,
                    "inference_batches": inference_batches,
                    "inference_metrics": inference_metrics,
                }
            )

            self.step_scheduler(train_result, val_result)

            if checkpoint_path is not None and is_main_process():
                self.save_epoch_checkpoints(
                    checkpoint_path=checkpoint_path,
                    epoch=epoch,
                    train_result=train_result,
                    val_result=val_result,
                    inference_metrics=inference_metrics,
                )

            if is_main_process():
                self.log_epoch(epoch, train_result, val_result, inference_metrics)

        return history

    def autocast_context(self):
        if not self.use_amp:
            return contextlib.nullcontext()
        return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)

    @contextlib.contextmanager
    def validation_weight_scope(self):
        if self.ema is None or not bool(getattr(self.training_cfg, "use_ema", False)):
            yield
            return

        self.ema.store(self.model.parameters())
        self.ema.copy_to(self.model.parameters())

        try:
            yield
            self._ema_weights_for_save = copy.deepcopy(self.raw_model.state_dict())
        finally:
            self.ema.restore(self.model.parameters())

    def generate_validation_sample(self, reference) -> SamplingResult:
        k_samples = 1
        if self.sampler is not None:
            k_samples = int(
                getattr(
                    self.sampler.cfg,
                    "samples_per_complex",
                    getattr(self.sampler.cfg, "k_samples_per_complex", 1),
                )
            )
        # Training-time valinf is an oracle score-model check, not standalone confidence-ranked inference.
        base_overrides = {
            "precision": "fp32",
            "samples_per_complex": 1,
            "k_samples_per_complex": 1,
            "batch_size": 1,
            "save_trajectory": False,
            "return_full_trajectory": False,
            "run_confidence": False,
            "rank_by_confidence": False,
        }
        if k_samples <= 1:
            return self.sampler.generate(
                data_list=[reference],
                model=self.raw_model,
                device=self.device,
                overrides=base_overrides,
            )

        best_result: SamplingResult | None = None
        best_rmsd = float("inf")

        for _ in range(k_samples):
            result = self.sampler.generate(
                data_list=[copy.deepcopy(reference)],
                model=self.raw_model,
                device=self.device,
                overrides=base_overrides,
            )

            prediction, rmsd = select_best_prediction_by_ligand_rmsd(reference, result.predictions)
            if prediction is None:
                continue

            if rmsd < best_rmsd:
                best_rmsd = rmsd
                best_result = SamplingResult(
                    predictions=[prediction],
                    confidences=None,
                    details=dict(result.details or {}),
                )

        if best_result is None:
            return self.sampler.generate(
                data_list=[reference],
                model=self.raw_model,
                device=self.device,
                overrides=base_overrides,
            )

        best_result.details = dict(best_result.details or {})
        best_result.details["best_of_k_rmsd"] = best_rmsd
        best_result.details["k_samples_per_complex"] = k_samples
        return best_result

    def save_epoch_checkpoints(
        self,
        checkpoint_path: Path,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult | None,
        inference_metrics: dict[str, float],
    ) -> None:
        metadata = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "val_loss": None if val_result is None else val_result.loss,
            "metrics": {} if val_result is None else dict(val_result.metrics),
        }
        use_ema = bool(getattr(self.training_cfg, "use_ema", False))
        ema_weights = self._ema_weights_for_save if use_ema else None
        if use_ema and ema_weights is None and self.ema is not None:
            self.ema.store(self.model.parameters())
            self.ema.copy_to(self.model.parameters())
            try:
                ema_weights = copy.deepcopy(self.raw_model.state_dict())
            finally:
                self.ema.restore(self.model.parameters())
        self.write_checkpoint(checkpoint_path / "last_model.pt", metadata, ema_weights)

        score_loss = val_result.loss if val_result is not None else train_result.loss
        if score_loss <= self._best_val_loss:
            self._best_val_loss = score_loss
            self.write_checkpoint(checkpoint_path / "best_model.pt", metadata, ema_weights)

        if inference_metrics:
            for metric_name in self.inference_checkpoint_metrics():
                if metric_name not in inference_metrics:
                    continue
                value = float(inference_metrics[metric_name])
                best = self._best_inference_metrics.get(metric_name)
                goal = getattr(self.training_cfg, "inference_earlystop_goal", "max")
                if best is None or (value > best if goal == "max" else value < best):
                    self._best_inference_metrics[metric_name] = value
                    self.write_checkpoint(
                        checkpoint_path / f"best_inference_epoch_model_{metric_name}.pt",
                        {**metadata, "inference_metric": metric_name, "inference_metric_value": value},
                        ema_weights,
                    )
            model_args = getattr(self.model, "args", None)
            raw_model_args = getattr(self.raw_model, "args", None)
            flexible_sidechains = bool(getattr(self.raw_model, "flexible_sidechains", getattr(raw_model_args or model_args, "flexible_sidechains", False)))
            flexible_backbone = bool(getattr(self.raw_model, "flexible_backbone", getattr(raw_model_args or model_args, "flexible_backbone", False)))
            if flexible_sidechains and "valinf_aa_rmsds_lt1" in inference_metrics:
                self.save_named_best_metric(checkpoint_path, "best_inference_epoch_model_aa.pt", "valinf_aa_rmsds_lt1", inference_metrics, metadata, ema_weights, goal="max")
            if flexible_backbone and "valinf_bb_rmsds_lt1" in inference_metrics:
                self.save_named_best_metric(checkpoint_path, "best_inference_epoch_model_bb.pt", "valinf_bb_rmsds_lt1", inference_metrics, metadata, ema_weights, goal="max")

    def write_checkpoint(self, path: Path, metadata: dict, ema_weights: dict[str, torch.Tensor] | None) -> None:
        save_checkpoint(
            path,
            model=self.raw_model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema=self.ema,
            ema_weights=ema_weights,
            **metadata,
        )

    def log_epoch(
        self,
        epoch: int,
        train_result: EpochResult,
        val_result: EpochResult | None,
        inference_metrics: dict[str, float],
    ) -> None:
        if self.logger is None:
            return
        payload: dict[str, float | int] = {
            "epoch": epoch + 1,
            "global_step": self._global_step,
            "train_loss": train_result.loss,
            "train_steps": train_result.steps,
        }
        payload.update(self.prefixed_metrics("train", train_result.metrics))
        if val_result is not None:
            payload["val_loss"] = val_result.loss
            payload["val_steps"] = val_result.steps
            payload.update(self.prefixed_metrics("val", val_result.metrics, skip_prefixes=("valinf_",)))
        payload.update(inference_metrics)
        current_lr = self.current_lr()
        if current_lr is not None:
            payload["lr"] = current_lr
        self.logger.log(payload, step=epoch + 1)

    @staticmethod
    def prefixed_metrics(prefix: str, metrics: dict[str, float], skip_prefixes: tuple[str, ...] = ()) -> dict[str, float]:
        prefixed = {}
        for key, value in metrics.items():
            if any(key.startswith(skip_prefix) for skip_prefix in skip_prefixes):
                prefixed[key] = value
            elif key == "loss":
                prefixed[f"{prefix}_loss_total"] = value
            else:
                prefixed[f"{prefix}_{key}"] = value
        return prefixed

    def current_lr(self) -> float | None:
        if self.optimizer is None or not self.optimizer.param_groups:
            return None
        return float(self.optimizer.param_groups[0].get("lr", 0.0))

    def finalize_epoch_result(
        self,
        total_loss: float,
        steps: int,
        metric_sums: dict[str, float],
    ) -> EpochResult:
        if is_distributed():
            gathered = all_gather_object(
                {
                    "loss": float(total_loss),
                    "steps": int(steps),
                    "metrics": {key: float(value) for key, value in metric_sums.items()},
                }
            )
            total_loss = sum(float(item["loss"]) for item in gathered)
            steps = sum(int(item["steps"]) for item in gathered)
            merged: dict[str, float] = {}
            for item in gathered:
                for key, value in item["metrics"].items():
                    merged[key] = merged.get(key, 0.0) + float(value)
            metric_sums = merged

        metrics = {key: value / steps for key, value in metric_sums.items()} if steps else {}
        return EpochResult(loss=total_loss / max(steps, 1), steps=steps, metrics=metrics)

    @staticmethod
    def set_loader_epoch(loader, epoch: int) -> None:
        if loader is None:
            return
        sampler = getattr(loader, "sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)

    def inference_checkpoint_metrics(self) -> list[str]:
        value = getattr(self.training_cfg, "inference_earlystop_metric", None)
        if not value:
            return []
        return [part.strip() for part in str(value).split(",") if part.strip()]

    def save_named_best_metric(
        self,
        checkpoint_path: Path,
        filename: str,
        metric_name: str,
        inference_metrics: dict[str, float],
        metadata: dict,
        ema_weights: dict[str, torch.Tensor] | None,
        goal: str,
    ) -> None:
        value = float(inference_metrics[metric_name])
        state_key = f"named:{filename}"
        best = self._best_inference_metrics.get(state_key)
        better = value > best if goal == "max" and best is not None else value < best if best is not None else True
        if better:
            self._best_inference_metrics[state_key] = value
            self.write_checkpoint(
                checkpoint_path / filename,
                {**metadata, "inference_metric": metric_name, "inference_metric_value": value},
                ema_weights,
            )

    def should_run_inference(self, epoch: int, inference_loader) -> bool:
        if inference_loader is None:
            return False
        freq = getattr(self.training_cfg, "val_inference_freq", None)
        if freq is None:
            return True
        if freq <= 0:
            return False
        return (epoch + 1) % int(freq) == 0

    def skip_bad_update(self, reason: str) -> bool:
        if self.skip_nan_grad_updates:
            if reason and is_main_process():
                logging.warning("Skipping optimizer update because %s.", reason)
            return True
        return False

    def handle_bad_update(self, bad_local: bool, reason: str) -> bool:
        bad_any = any_rank_has(bool(bad_local), device=self.device)
        if not bad_any:
            return False
        if self.skip_bad_update(reason if bad_local else "another rank found non-finite values"):
            return True
        if bad_local:
            raise FloatingPointError(f"Encountered {reason}.")
        raise FloatingPointError("Another distributed rank encountered non-finite values.")

    @staticmethod
    def nonfinite_tensor_stats(values: dict) -> dict[str, tuple[int, int]]:
        stats = {}
        for key, value in values.items():
            if not torch.is_tensor(value) or value.numel() == 0 or torch.isfinite(value).all():
                continue
            stats[key] = (int(torch.isnan(value).sum().item()), int(torch.isinf(value).sum().item()))
        return stats

    @staticmethod
    def format_nonfinite_stats(label: str, stats: dict[str, tuple[int, int]]) -> str:
        if not stats:
            return f"non-finite {label}"
        parts = [f"{key}: {nan_count} NaNs, {inf_count} Infs" for key, (nan_count, inf_count) in stats.items()]
        return f"non-finite {label} ({'; '.join(parts)})"

    @staticmethod
    def is_numeric_exception(exc: Exception) -> bool:
        message = str(exc).lower()
        return "non-finite" in message or "nan" in message or "inf" in message

    def has_bad_gradients(self) -> bool:
        if not self.check_nan_grads:
            return False
        for param in self.model.parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                return True
        return False

    def check_unused_parameters(self) -> None:
        if not bool(getattr(self.training_cfg, "check_unused_params", False)):
            return
        unused = [name for name, param in self.model.named_parameters() if param.requires_grad and param.grad is None]
        if unused:
            preview = ", ".join(unused[:10])
            suffix = "" if len(unused) <= 10 else f", ... ({len(unused)} total)"
            raise RuntimeError(f"Unused trainable parameters detected: {preview}{suffix}")

    def step_scheduler(self, train_result: EpochResult, val_result: EpochResult | None) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            metric_name = getattr(self.training_cfg, "inference_earlystop_metric", None)
            metric = None
            if metric_name and val_result is not None:
                metric = val_result.metrics.get(metric_name)
            if metric is None:
                metric = val_result.loss if val_result is not None else train_result.loss
            self.scheduler.step(metric)
        else:
            self.scheduler.step()
