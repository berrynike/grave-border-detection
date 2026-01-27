"""Hyperparameter optimization with Optuna.

This module provides HPO functionality with:
- SQLite storage for resumability (can restart interrupted studies)
- Pruning to early-stop unpromising trials
- MLflow integration with parent-child runs for organization
- Configurable search spaces via Hydra configs

MLflow organization (best practice for HPO):
- Parent run = entire HPO study (dataset, study config, best params)
- Child runs = individual trials (nested under parent)

Usage:
    # Run HPO with default config
    uv run python -m grave_border_detection.hpo

    # Run pilot study (smaller search space)
    uv run python -m grave_border_detection.hpo +hpo=pilot

    # Resume interrupted study (just run same command again)
    uv run python -m grave_border_detection.hpo

    # View results in dashboard
    optuna-dashboard sqlite:///hpo_studies.db
"""

import logging
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import mlflow
import optuna
from hydra.utils import to_absolute_path
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import DictConfig, OmegaConf
from optuna.integration import PyTorchLightningPruningCallback

from grave_border_detection.callbacks import ImageLoggerCallback
from grave_border_detection.data.datamodule import GraveDataModule
from grave_border_detection.models.segmentation import SegmentationModel
from grave_border_detection.utils.dataset_hash import compute_dataset_id

log = logging.getLogger(__name__)


def sample_hyperparameters(trial: optuna.Trial, search_space: DictConfig) -> dict[str, Any]:
    """Sample hyperparameters from the configured search space.

    Args:
        trial: Optuna trial object.
        search_space: Search space configuration from Hydra.

    Returns:
        Dictionary of sampled hyperparameters.
    """
    params: dict[str, Any] = {}

    for name, config in search_space.items():
        param_name = str(name)
        param_type = config.type

        if param_type == "float":
            params[param_name] = trial.suggest_float(
                param_name,
                config.low,
                config.high,
                log=config.get("log", False),
            )
        elif param_type == "int":
            params[param_name] = trial.suggest_int(
                param_name,
                config.low,
                config.high,
                step=config.get("step", 1),
            )
        elif param_type == "categorical":
            params[param_name] = trial.suggest_categorical(param_name, config.choices)
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")

    return params


def create_objective(
    cfg: DictConfig,
    hpo_cfg: DictConfig,
    parent_run_id: str,
) -> Any:
    """Create Optuna objective function with MLflow child runs.

    Args:
        cfg: Base Hydra config.
        hpo_cfg: HPO-specific config.
        parent_run_id: MLflow parent run ID for nesting child runs.

    Returns:
        Objective function for Optuna.
    """
    study_name = str(hpo_cfg.get("study_name", "hpo"))

    def objective(trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = sample_hyperparameters(trial, hpo_cfg.search_space)
        log.info(f"Trial {trial.number} params: {params}")

        # Extract sampled values with defaults from base config
        lr = params.get("lr", cfg.training.optimizer.lr)
        weight_decay = params.get("weight_decay", cfg.training.optimizer.weight_decay)
        bce_weight = params.get("bce_weight", cfg.training.loss.bce_weight)
        max_epochs = params.get("max_epochs", cfg.training.max_epochs)
        encoder_name = params.get("encoder_name", cfg.model.encoder_name)

        # Create data module (use absolute path since Hydra may change cwd)
        data_root = to_absolute_path(cfg.data.root)

        # Use smaller batch size for HPO (dataset may be small)
        hpo_batch_size = min(cfg.data.batch_size, 2)

        data_module = GraveDataModule(
            data_root=data_root,
            train_cemeteries=list(cfg.data.train_cemeteries),
            val_cemeteries=list(cfg.data.val_cemeteries),
            test_cemeteries=[],  # No test during HPO
            tile_size=cfg.data.tiling.tile_size,
            overlap=cfg.data.tiling.overlap,
            min_mask_coverage=cfg.data.tiling.get("min_mask_coverage", 0.0),
            use_dem=cfg.data.use_dem,
            batch_size=hpo_batch_size,
            num_workers=cfg.data.num_workers,
        )

        # Create model with sampled hyperparameters
        model = SegmentationModel(
            architecture=cfg.model.architecture,
            encoder_name=encoder_name,
            encoder_weights=cfg.model.encoder_weights,
            in_channels=cfg.model.in_channels,
            classes=cfg.model.classes,
            lr=lr,
            weight_decay=weight_decay,
            bce_weight=bce_weight,
            dice_weight=1.0 - bce_weight,
            max_epochs=max_epochs,
        )

        # Callbacks
        callbacks: list[L.Callback] = [
            PyTorchLightningPruningCallback(trial, monitor="val/dice"),
            EarlyStopping(monitor="val/dice", patience=10, mode="max"),
            ImageLoggerCallback(num_samples=2, log_every_n_epochs=5),
        ]

        # Create child run under parent (nested run pattern)
        # This groups all trials under the parent HPO run
        mlflow_logger = MLFlowLogger(
            experiment_name=cfg.experiment_name,
            tracking_uri=cfg.training.get("mlflow_tracking_uri", "mlruns"),
            run_name=f"trial_{trial.number:03d}",
            tags={
                "mlflow.parentRunId": parent_run_id,
                "hpo_study": study_name,
                "trial_number": str(trial.number),
            },
        )

        # Log trial params to MLflow
        mlflow_logger.log_hyperparams(
            {
                "trial_number": trial.number,
                "lr": lr,
                "weight_decay": weight_decay,
                "bce_weight": bce_weight,
                "dice_weight": 1.0 - bce_weight,
                "max_epochs": max_epochs,
                "encoder_name": encoder_name,
            }
        )

        # Trainer
        trainer = L.Trainer(
            max_epochs=max_epochs,
            accelerator=cfg.training.get("accelerator", "auto"),
            devices=cfg.training.get("devices", "auto"),
            callbacks=callbacks,
            logger=mlflow_logger,
            enable_progress_bar=True,
            enable_checkpointing=False,  # Don't checkpoint during HPO
            log_every_n_steps=cfg.training.get("log_every_n_steps", 10),
        )

        # Train
        try:
            trainer.fit(model, data_module)
        except optuna.TrialPruned:
            raise  # Re-raise pruning exception

        # Return best validation dice (or current if no best)
        val_dice = trainer.callback_metrics.get("val/dice")
        if val_dice is None:
            log.warning("No val/dice metric found, returning 0.0")
            return 0.0

        return float(val_dice.item())

    return objective


@hydra.main(version_base="1.3", config_path="../../configs", config_name="config")
def run_hpo(cfg: DictConfig) -> None:
    """Run hyperparameter optimization.

    Args:
        cfg: Hydra configuration.
    """
    log.info("Starting HPO...")
    log.info(f"Base config:\n{OmegaConf.to_yaml(cfg)}")

    # Load HPO config (default or specified via +hpo=pilot)
    hpo_cfg = cfg.get("hpo")
    if hpo_cfg is None:
        hpo_config_path = Path(__file__).parent.parent.parent / "configs" / "hpo" / "default.yaml"
        hpo_cfg = OmegaConf.load(hpo_config_path)
        log.info(f"Loaded default HPO config from {hpo_config_path}")

    log.info(f"HPO config:\n{OmegaConf.to_yaml(hpo_cfg)}")

    # Get HPO settings
    storage = hpo_cfg.get("storage", "sqlite:///hpo_studies.db")
    study_name = hpo_cfg.get("study_name", "grave_border_hpo")
    n_trials = hpo_cfg.get("n_trials", 50)
    direction = hpo_cfg.get("direction", "maximize")
    timeout = hpo_cfg.get("timeout")

    # Create pruner
    pruner_cfg = hpo_cfg.get("pruner", {})
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=pruner_cfg.get("n_startup_trials", 5),
        n_warmup_steps=pruner_cfg.get("n_warmup_steps", 10),
        interval_steps=pruner_cfg.get("interval_steps", 1),
    )

    # Create or load study (resumable via SQLite)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction=direction,
        load_if_exists=True,
        pruner=pruner,
    )

    n_existing = len(study.trials)
    if n_existing > 0:
        log.info(f"Resuming study '{study_name}' with {n_existing} existing trials")
        log.info(f"Best value so far: {study.best_value:.4f}")
        log.info(f"Best params so far: {study.best_params}")

    # Calculate remaining trials
    n_remaining = max(0, n_trials - n_existing)
    if n_remaining == 0:
        log.info(f"Study already has {n_existing} trials (target: {n_trials}). Nothing to do.")
        _print_results(study, str(storage))
        return

    log.info(f"Running {n_remaining} more trials (total target: {n_trials})")

    # Resolve data root for dataset tracking
    data_root = to_absolute_path(cfg.data.root)

    # Compute dataset ID
    dataset_id = compute_dataset_id(
        data_root=Path(data_root),
        train_cemeteries=list(cfg.data.train_cemeteries),
        val_cemeteries=list(cfg.data.val_cemeteries),
    )

    # Set up MLflow tracking
    tracking_uri = cfg.training.get("mlflow_tracking_uri", "mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)

    # Create parent run for the entire HPO study
    # Child runs (trials) will be nested under this
    with mlflow.start_run(run_name=f"hpo_{study_name}") as parent_run:
        parent_run_id = parent_run.info.run_id

        # Tag parent run
        mlflow.set_tags(
            {
                "run_type": "hpo_parent",
                "hpo_study": study_name,
                "n_trials": str(n_trials),
            }
        )

        # Log HPO config and dataset info to parent
        # For image data, we track cemeteries and a content hash (dataset_id)
        # rather than using mlflow.data which is designed for tabular data
        mlflow.log_params(
            {
                "hpo_study_name": study_name,
                "hpo_n_trials": n_trials,
                "hpo_direction": direction,
                "dataset_id": dataset_id,
                "data_root": data_root,
                "train_cemeteries": ",".join(cfg.data.train_cemeteries),
                "val_cemeteries": ",".join(cfg.data.val_cemeteries),
            }
        )

        # Run optimization with child runs
        study.optimize(
            create_objective(cfg, hpo_cfg, parent_run_id),
            n_trials=n_remaining,
            timeout=timeout,
            show_progress_bar=True,
        )

        # Log best results to parent run
        if study.best_trial:
            mlflow.log_metric("best_val_dice", study.best_value)
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.set_tag("best_trial_number", str(study.best_trial.number))

    _print_results(study, str(storage))


def _print_results(study: optuna.Study, storage: str) -> None:
    """Print HPO results summary.

    Args:
        study: Completed Optuna study.
        storage: Storage URL for the study.
    """
    log.info("=" * 60)
    log.info("HPO Complete!")
    log.info(f"Study name: {study.study_name}")
    log.info(f"Total trials: {len(study.trials)}")

    # Count trial states
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    n_failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])

    log.info(f"  Complete: {n_complete}")
    log.info(f"  Pruned: {n_pruned}")
    log.info(f"  Failed: {n_failed}")

    if study.best_trial:
        log.info("")
        log.info(f"Best trial: #{study.best_trial.number}")
        log.info(f"Best value (val/dice): {study.best_value:.4f}")
        log.info("Best hyperparameters:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                log.info(f"  {key}: {value:.6g}")
            else:
                log.info(f"  {key}: {value}")

    log.info("=" * 60)
    log.info(f"View results: optuna-dashboard {storage}")


if __name__ == "__main__":
    run_hpo()
