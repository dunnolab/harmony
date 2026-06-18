from __future__ import annotations

from pathlib import Path
from typing import Iterable

from docking_model.config import load_config
from docking_model.config.serialization import save_config
from docking_model.data.preprocessing import ensure_preprocessed_cache
from docking_model.losses import DockingLoss
from docking_model.runtime.engine import DockingEngine
from docking_model.runtime.factory import (
    build_cached_loader,
    build_optimizer,
    build_sampler,
    build_scheduler,
    build_score_model,
    build_t_to_sigma,
    build_transform,
    build_validation_inference_loader,
    select_device,
)
from docking_model.runtime.distributed import (
    barrier,
    cleanup_distributed,
    is_main_process,
    setup_distributed,
    wrap_model_for_distributed,
)
from docking_model.runtime.loggers import build_experiment_logger
from docking_model.runtime.seeding import seed_everything


def run_training(
    config_path: str,
    model=None,
    loss_fn=None,
    train_transform=None,
    val_transform=None,
    sampler=None,
    inference_transform=None,
    overrides: Iterable[str] | None = None,
):
    cfg = load_config(config_path, overrides=overrides)
    distributed = setup_distributed()
    if distributed and cfg.training.strategy == "none":
        raise ValueError("WORLD_SIZE > 1 requires training.strategy to be 'auto' or 'ddp'.")
    seed_everything(cfg.seed, workers=cfg.data.num_workers > 0, verbose=is_main_process())

    logger = None
    try:
        if is_main_process():
            ensure_preprocessed_cache(cfg)
        barrier()

        logger = build_experiment_logger(cfg, job_type="train")
        if is_main_process():
            save_model_parameters(cfg)
        if cfg.data.split_train is None:
            raise ValueError("data.split_train is required for training.")

        model = model or build_score_model(cfg)
        loss_fn = loss_fn or DockingLoss(args=cfg.loss, t_to_sigma=build_t_to_sigma(cfg))
        train_transform = train_transform or build_transform(cfg, mode="train")
        val_transform = val_transform or build_transform(cfg, mode="val")
        inference_transform = inference_transform or build_transform(cfg, mode="inference")
        sampler = sampler or build_sampler(cfg)

        device = select_device(cfg.training.device)
        model.to(device)
        model = wrap_model_for_distributed(
            model,
            device,
            find_unused_parameters=bool(cfg.training.find_unused_parameters),
        )

        train_loader = build_cached_loader(
            cfg=cfg,
            split_path=cfg.data.split_train,
            transform=train_transform,
            shuffle=True,
        )
        val_loader = None
        if cfg.data.split_val is not None:
            val_loader = build_cached_loader(
                cfg=cfg,
                split_path=cfg.data.split_val,
                transform=val_transform,
                shuffle=False,
                multiplicity=min(cfg.data.multiplicity, 5),
            )

        inference_loader = None
        run_val_inference = cfg.data.run_val_inference or (
            cfg.training.val_inference_freq is not None and cfg.training.val_inference_freq > 0
        )
        if run_val_inference:
            if val_loader is None:
                raise ValueError("Validation inference requires data.split_val.")
            inference_loader = build_validation_inference_loader(
                cfg=cfg,
                validation_dataset=val_loader.dataset,
                transform=inference_transform,
            )

        optimizer = build_optimizer(model, cfg)
        scheduler = build_scheduler(optimizer, cfg)
        engine = DockingEngine(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            sampler=sampler,
            gradient_clip_norm=cfg.training.optimizer.gradient_clip_norm,
            scheduler=scheduler,
            training_cfg=cfg.training,
            logger=logger,
        )
        return engine.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            inference_loader=inference_loader,
            epochs=cfg.training.epochs,
            checkpoint_dir=cfg.training.output_dir,
        )
    finally:
        if logger is not None and cfg.logger.finish_on_exit:
            logger.finish()
        cleanup_distributed()


def save_model_parameters(cfg) -> None:
    source_path = getattr(cfg, "source_path", None)
    if source_path is None:
        return
    output_dir = Path(cfg.training.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, output_dir / "model_parameters.yml")
