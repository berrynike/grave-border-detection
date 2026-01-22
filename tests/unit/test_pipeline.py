"""Smoke tests for the training pipeline."""

from pathlib import Path

import torch


def test_synthetic_data_generates(tmp_path: Path) -> None:
    """Test that synthetic data generator runs."""
    from grave_border_detection.data.synthetic import generate_synthetic_dataset

    result = generate_synthetic_dataset(
        output_dir=tmp_path,
        num_cemeteries=1,
        image_size=(256, 256),
        graves_per_cemetery=3,
    )

    assert result["orthophotos"].exists()
    assert result["masks"].exists()


def test_dataloader_produces_batches(tmp_path: Path) -> None:
    """Test that we can load data into batches."""
    from grave_border_detection.data.datamodule import GraveDataModule
    from grave_border_detection.data.synthetic import generate_synthetic_dataset

    # Generate data
    generate_synthetic_dataset(
        output_dir=tmp_path,
        num_cemeteries=2,
        image_size=(256, 256),
        graves_per_cemetery=3,
    )

    # Load into datamodule
    dm = GraveDataModule(
        data_root=tmp_path,
        train_cemeteries=["synthetic_01"],
        val_cemeteries=["synthetic_02"],
        tile_size=128,
        batch_size=2,
        num_workers=0,
        use_dem=False,
    )
    dm.setup("fit")

    # Get a batch
    batch = next(iter(dm.train_dataloader()))
    images, masks = batch

    assert images.shape == (2, 3, 128, 128)
    assert masks.shape == (2, 1, 128, 128)


def test_model_forward_pass() -> None:
    """Test that model produces correct output shape."""
    from grave_border_detection.models.segmentation import SegmentationModel

    model = SegmentationModel(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
    )

    x = torch.randn(2, 3, 128, 128)
    y = model(x)

    assert y.shape == (2, 1, 128, 128)


def test_training_loop_runs(tmp_path: Path) -> None:
    """Test that training runs for one step without error."""
    import lightning as L

    from grave_border_detection.data.datamodule import GraveDataModule
    from grave_border_detection.data.synthetic import generate_synthetic_dataset
    from grave_border_detection.models.segmentation import SegmentationModel

    # Generate data
    generate_synthetic_dataset(
        output_dir=tmp_path,
        num_cemeteries=2,
        image_size=(256, 256),
        graves_per_cemetery=3,
    )

    # Setup
    dm = GraveDataModule(
        data_root=tmp_path,
        train_cemeteries=["synthetic_01"],
        val_cemeteries=["synthetic_02"],
        tile_size=128,
        batch_size=2,
        num_workers=0,
        use_dem=False,
    )

    model = SegmentationModel(
        architecture="Unet",
        encoder_name="resnet18",
        encoder_weights=None,
        in_channels=3,
    )

    trainer = L.Trainer(
        max_epochs=1,
        accelerator="cpu",
        fast_dev_run=True,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    # This should not raise
    trainer.fit(model, dm)
