"""Callback for logging full cemetery visualizations after test phase."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import lightning as L
from lightning.pytorch.loggers import MLFlowLogger

if TYPE_CHECKING:
    import numpy as np

from grave_border_detection.inference import (
    create_full_visualization,
    predict_full_cemetery,
)
from grave_border_detection.postprocessing import (
    polygons_to_geopackage,
    prediction_to_polygons,
)

log = logging.getLogger(__name__)


class FullCemeteryVisualizationCallback(L.Callback):
    """Log full cemetery visualizations after test phase.

    Reconstructs full cemetery predictions and logs them to MLflow.
    """

    def __init__(
        self,
        data_root: str | Path,
        test_cemeteries: list[str],
        tile_size: int = 512,
        overlap: float = 0.15,
        use_dem: bool = False,
    ) -> None:
        """Initialize callback.

        Args:
            data_root: Root directory containing orthophotos/, masks/, dems/.
            test_cemeteries: List of cemetery IDs to visualize.
            tile_size: Tile size for inference.
            overlap: Overlap between tiles.
            use_dem: Whether to use DEM channel.
        """
        super().__init__()
        self.data_root = Path(data_root)
        self.test_cemeteries = test_cemeteries
        self.tile_size = tile_size
        self.overlap = overlap
        self.use_dem = use_dem

    def on_test_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
    ) -> None:
        """Generate and log full cemetery visualizations after test."""
        log.info("Generating full cemetery visualizations...")

        # Get MLflow logger
        mlflow_logger = self._get_mlflow_logger(trainer)

        # Get device
        device = next(pl_module.parameters()).device

        for cemetery_id in self.test_cemeteries:
            log.info(f"Processing cemetery: {cemetery_id}")

            # Build paths - try multiple naming conventions
            ortho_path = self.data_root / "orthophotos" / f"{cemetery_id}_ortho.tif"
            if not ortho_path.exists():
                ortho_path = self.data_root / "orthophotos" / f"{cemetery_id}.tif"

            mask_path = self.data_root / "masks" / f"{cemetery_id}_mask.tif"
            if not mask_path.exists():
                mask_path = self.data_root / "masks" / f"{cemetery_id}.tif"

            dem_path = None
            if self.use_dem:
                dem_path = self.data_root / "dems" / f"{cemetery_id}_dem.tif"
                if not dem_path.exists():
                    dem_path = self.data_root / "dems" / f"{cemetery_id}.tif"

            if not ortho_path.exists():
                log.warning(f"Orthophoto not found: {ortho_path}")
                continue

            # Run full inference
            result = predict_full_cemetery(
                model=pl_module,
                orthophoto_path=ortho_path,
                mask_path=mask_path if mask_path.exists() else None,
                dem_path=dem_path,
                tile_size=self.tile_size,
                overlap=self.overlap,
                device=device,
            )

            # Create visualization
            viz = create_full_visualization(
                rgb=result["rgb"],
                prediction=result["prediction"],
                mask=result.get("mask"),
            )

            # Extract polygons and export to GeoPackage
            polygons = prediction_to_polygons(
                result["prediction"],
                threshold=0.5,
                min_area=100,
                transform=result.get("transform"),
                simplify_tolerance=1.0,
            )

            if polygons:
                gpkg_path = Path("outputs") / f"{cemetery_id}_predictions.gpkg"
                crs = str(result.get("crs")) if result.get("crs") else None
                polygons_to_geopackage(polygons, gpkg_path, crs=crs)
                log.info(f"Exported {len(polygons)} polygons to {gpkg_path}")

                # Also log GeoPackage to MLflow
                if mlflow_logger is not None:
                    self._log_file_to_mlflow(mlflow_logger, gpkg_path, "predictions")

            # Log to MLflow
            if mlflow_logger is not None:
                self._log_image_to_mlflow(
                    mlflow_logger,
                    viz,
                    f"full_cemetery_{cemetery_id}",
                )
            else:
                # Save locally if no MLflow
                output_path = Path("outputs") / f"full_cemetery_{cemetery_id}.png"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                self._save_image(viz, output_path)
                log.info(f"Saved visualization to {output_path}")

        log.info("Full cemetery visualizations complete.")

    def _get_mlflow_logger(self, trainer: L.Trainer) -> MLFlowLogger | None:
        """Get MLflow logger from trainer."""
        if trainer.logger is None:
            return None
        if isinstance(trainer.logger, MLFlowLogger):
            return trainer.logger
        return None

    def _log_image_to_mlflow(
        self,
        logger: MLFlowLogger,
        image: "np.ndarray",
        name: str,
    ) -> None:
        """Log image array to MLflow."""
        try:
            import tempfile

            import mlflow
            from PIL import Image

            img_pil = Image.fromarray(image)

            with tempfile.TemporaryDirectory() as tmpdir:
                img_path = Path(tmpdir) / f"{name}.png"
                img_pil.save(img_path)

                run_id = logger.run_id
                if run_id:
                    mlflow.log_artifact(str(img_path), artifact_path="full_cemetery_predictions")
                    log.info(f"Logged {name} to MLflow")

        except Exception as e:
            log.warning(f"Failed to log image to MLflow: {e}")

    def _save_image(self, image: "np.ndarray", path: Path) -> None:
        """Save image to disk."""
        from PIL import Image

        img_pil = Image.fromarray(image)
        img_pil.save(path)

    def _log_file_to_mlflow(
        self,
        logger: MLFlowLogger,
        file_path: Path,
        artifact_path: str,
    ) -> None:
        """Log a file to MLflow artifacts."""
        try:
            import mlflow

            run_id = logger.run_id
            if run_id and file_path.exists():
                mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
                log.info(f"Logged {file_path.name} to MLflow")
        except Exception as e:
            log.warning(f"Failed to log file to MLflow: {e}")
