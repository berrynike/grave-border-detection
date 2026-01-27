"""Callback for logging model checkpoints to MLflow artifacts."""

import logging
from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

log = logging.getLogger(__name__)


class CheckpointLoggerCallback(L.Callback):
    """Log model checkpoints to MLflow checkpoints/ artifact folder.

    Logs checkpoints at end of training to keep artifacts organized:
        checkpoints/
            best.ckpt        # Best model by monitored metric
            last.ckpt        # Final model state
    """

    def on_fit_end(self, trainer: L.Trainer, _pl_module: L.LightningModule) -> None:
        """Log checkpoints to MLflow after training completes."""
        mlflow_logger = self._get_mlflow_logger(trainer)
        if mlflow_logger is None:
            return

        # Find the ModelCheckpoint callback
        checkpoint_callback = self._get_checkpoint_callback(trainer)
        if checkpoint_callback is None:
            log.warning("No ModelCheckpoint callback found, skipping checkpoint logging")
            return

        run_id = mlflow_logger.run_id
        if not run_id:
            log.warning("No MLflow run_id, skipping checkpoint logging")
            return

        from mlflow.tracking import MlflowClient

        client = MlflowClient()

        # Log best checkpoint
        best_path = checkpoint_callback.best_model_path
        if best_path and Path(best_path).exists():
            client.log_artifact(run_id, best_path, artifact_path="checkpoints")
            log.info("Logged best checkpoint to MLflow (checkpoints/)")

        # Log last checkpoint
        last_path = checkpoint_callback.last_model_path
        if last_path and Path(last_path).exists() and last_path != best_path:
            client.log_artifact(run_id, last_path, artifact_path="checkpoints")
            log.info("Logged last checkpoint to MLflow (checkpoints/)")

    def _get_mlflow_logger(self, trainer: L.Trainer) -> MLFlowLogger | None:
        """Get MLflow logger from trainer."""
        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        return None

    def _get_checkpoint_callback(self, trainer: L.Trainer) -> ModelCheckpoint | None:
        """Get ModelCheckpoint callback from trainer."""
        # Use trainer's built-in property which returns the first ModelCheckpoint
        callback = trainer.checkpoint_callback
        if isinstance(callback, ModelCheckpoint):
            return callback
        return None
