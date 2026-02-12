"""Training entrypoint for grave border detection."""

import logging
from pathlib import Path
from typing import Any, cast

import hydra
import lightning as L
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf

from grave_border_detection.callbacks import (
    CheckpointLoggerCallback,
    FullCemeteryVisualizationCallback,
    ImageLoggerCallback,
)
from grave_border_detection.data.datamodule import GraveDataModule
from grave_border_detection.models.segmentation import SegmentationModel
from grave_border_detection.utils.dataset_hash import compute_dataset_id, get_dataset_summary

log = logging.getLogger(__name__)


def setup_callbacks(cfg: DictConfig, data_root: str) -> list[L.Callback]:
    """Set up training callbacks.

    Args:
        cfg: Hydra config.
        data_root: Absolute path to data root directory.

    Returns:
        List of Lightning callbacks.
    """
    callbacks: list[L.Callback] = []

    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.training.checkpoint_dir,
        filename="best",  # Cleaner name, score is in MLflow metrics
        monitor="val/dice",
        mode="max",
        save_top_k=1,  # Just keep the best
        save_last=True,
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if cfg.training.early_stopping.enabled:
        early_stop_callback = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            verbose=True,
        )
        callbacks.append(early_stop_callback)

    # Image logging (per-epoch tile samples)
    log_images_every = cfg.training.logging.get("log_images_every_n_epochs", 5)
    image_logger = ImageLoggerCallback(
        num_samples=4,
        log_every_n_epochs=log_images_every,
    )
    callbacks.append(image_logger)

    # Full cemetery visualization (after test phase)
    test_cemeteries = list(cfg.data.get("test_cemeteries", []))
    if test_cemeteries:
        dem_norm_method = cfg.data.dem.normalization.method
        full_viz_callback = FullCemeteryVisualizationCallback(
            data_root=data_root,
            test_cemeteries=test_cemeteries,
            tile_size=cfg.data.tiling.tile_size,
            overlap=cfg.data.tiling.overlap,
            use_dem=cfg.data.use_dem,
            dem_normalization_method=dem_norm_method,
        )
        callbacks.append(full_viz_callback)

    # Checkpoint logging to MLflow (logs to checkpoints/ folder)
    callbacks.append(CheckpointLoggerCallback())

    return callbacks


def setup_logger(cfg: DictConfig) -> MLFlowLogger | None:
    """Set up MLflow logger.

    Args:
        cfg: Hydra config.

    Returns:
        MLFlowLogger or None if disabled.
    """
    if not cfg.training.get("use_mlflow", True):
        return None

    experiment_name = cfg.get("experiment_name", "grave_border_detection")

    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=cfg.training.get("mlflow_tracking_uri", "mlruns"),
        log_model=False,  # We log checkpoints manually to checkpoints/ folder
    )


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def train(cfg: DictConfig) -> float | None:
    """Train the grave border detection model.

    Args:
        cfg: Hydra configuration.

    Returns:
        Best validation Dice score, or None if fast_dev_run.
    """
    # Log configuration
    log.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Extract DEM normalization config
    dem_norm_method = cfg.data.dem.normalization.method
    dem_norm_params = dict(cfg.data.dem.normalization.get(dem_norm_method, {}))

    # Use absolute path since Hydra may change cwd
    data_root = to_absolute_path(cfg.data.root)

    # Create data module
    data_module = GraveDataModule(
        data_root=data_root,
        train_cemeteries=list(cfg.data.train_cemeteries),
        val_cemeteries=list(cfg.data.val_cemeteries),
        test_cemeteries=list(cfg.data.get("test_cemeteries", [])),
        tile_size=cfg.data.tiling.tile_size,
        overlap=cfg.data.tiling.overlap,
        min_mask_coverage=cfg.data.tiling.get("min_mask_coverage", 0.0),
        use_dem=cfg.data.use_dem,
        dem_normalization_method=dem_norm_method,
        dem_normalization_params=dem_norm_params,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )

    # Create model (use data_module.input_channels to ensure consistency)
    in_channels = data_module.input_channels
    log.info(f"Model input channels: {in_channels} (from data module)")
    model = SegmentationModel(
        architecture=cfg.model.architecture,
        encoder_name=cfg.model.encoder_name,
        encoder_weights=cfg.model.encoder_weights,
        in_channels=in_channels,
        classes=cfg.model.classes,
        lr=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        bce_weight=cfg.training.loss.bce_weight,
        dice_weight=cfg.training.loss.dice_weight,
        max_epochs=cfg.training.max_epochs,
    )

    # Set up callbacks and logger
    callbacks = setup_callbacks(cfg, data_root)
    logger = setup_logger(cfg)

    # Compute dataset versioning
    data_root_path = Path(data_root)
    dataset_id = compute_dataset_id(
        data_root=data_root_path,
        train_cemeteries=list(cfg.data.train_cemeteries),
        val_cemeteries=list(cfg.data.val_cemeteries),
        test_cemeteries=list(cfg.data.get("test_cemeteries", [])),
    )
    dataset_summary = get_dataset_summary(data_root_path)
    log.info(f"Dataset ID: {dataset_id}")

    # Log hyperparameters to MLflow (including dataset versioning)
    if logger is not None:
        config_dict = cast("dict[str, Any]", OmegaConf.to_container(cfg, resolve=True))
        config_dict["dataset_id"] = dataset_id
        config_dict["dataset_summary"] = dataset_summary
        logger.log_hyperparams(config_dict)

        # Log dataset_id as a tag for prominent visibility in MLflow UI
        if logger.run_id:
            logger.experiment.set_tag(logger.run_id, "dataset_id", dataset_id)

    # Determine accelerator
    accelerator = cfg.training.get("accelerator", "auto")

    # Create trainer
    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=cfg.training.get("devices", "auto"),
        callbacks=callbacks,
        logger=logger,
        fast_dev_run=cfg.training.get("fast_dev_run", False),
        log_every_n_steps=cfg.training.get("log_every_n_steps", 10),
        deterministic=cfg.get("deterministic", False),
    )

    # Train
    log.info("Starting training...")
    trainer.fit(model, data_module)

    # Get best validation score
    best_score: float | None = None
    if not cfg.training.get("fast_dev_run", False):
        # Get best model score from the checkpoint callback
        checkpoint_callbacks = [c for c in callbacks if isinstance(c, ModelCheckpoint)]
        if checkpoint_callbacks:
            score = checkpoint_callbacks[0].best_model_score
            if score is not None:
                best_score = float(score)
            log.info("Best validation Dice: %.4f", best_score or 0.0)

    # Test if test data available
    if cfg.data.get("test_cemeteries") and not cfg.training.get("fast_dev_run", False):
        log.info("Running test evaluation...")
        trainer.test(model, data_module)

    return best_score


if __name__ == "__main__":
    train()
