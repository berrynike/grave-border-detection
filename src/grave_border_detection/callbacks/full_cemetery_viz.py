"""Callback for logging full cemetery visualizations after test phase.

Logs test results to `final/` artifact folder:
- {cemetery_id}_combined.png: Full cemetery visualization
- {cemetery_id}_predictions.gpkg: Vectorized predictions
"""

import logging
import tempfile
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

        # Use temp directory for all outputs (cleaned up after logging to MLflow)
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

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
                    gpkg_path = tmpdir_path / f"{cemetery_id}_predictions.gpkg"
                    crs = str(result.get("crs")) if result.get("crs") else None
                    polygons_to_geopackage(polygons, gpkg_path, crs=crs)
                    log.info(f"Exported {len(polygons)} polygons")

                    # Log GeoPackage to MLflow final/ folder
                    if mlflow_logger is not None:
                        self._log_file_to_mlflow(mlflow_logger, gpkg_path, "final")

                # Log visualization to MLflow final/ folder
                if mlflow_logger is not None:
                    self._log_image_to_mlflow(
                        mlflow_logger,
                        viz,
                        f"{cemetery_id}_combined",
                        tmpdir_path,
                    )
                else:
                    # Save locally only if no MLflow (fallback)
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
        tmpdir: Path,
    ) -> None:
        """Log image array to MLflow final/ folder.

        Args:
            logger: MLflow logger instance.
            image: Image array (H, W, 3) in uint8 format.
            name: Image name (without extension).
            tmpdir: Temporary directory for intermediate files.
        """
        from mlflow.tracking import MlflowClient
        from PIL import Image

        run_id = logger.run_id
        if not run_id:
            log.warning("No MLflow run_id, skipping image logging")
            return

        # Save to temp file then log as artifact
        img_path = tmpdir / f"{name}.png"
        img_pil = Image.fromarray(image)
        img_pil.save(img_path)

        client = MlflowClient()
        client.log_artifact(run_id, str(img_path), artifact_path="final")
        log.info(f"Logged {name} to MLflow (final/)")

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
        from mlflow.tracking import MlflowClient

        run_id = logger.run_id
        if not run_id:
            log.warning("No MLflow run_id, skipping file logging")
            return

        if not file_path.exists():
            log.warning(f"File not found: {file_path}")
            return

        # Log to MLflow using client (doesn't require active run context)
        client = MlflowClient()
        client.log_artifact(run_id, str(file_path), artifact_path=artifact_path)
        log.info(f"Logged {file_path.name} to MLflow")
