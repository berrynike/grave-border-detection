"""Smoke tests for Hydra configuration."""

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_configs_load() -> None:
    """Test that all config combinations load without error."""
    config_dir = str(Path("configs").absolute())

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        # Default config (synthetic with DEM - 4 channels)
        cfg = compose(config_name="config")
        assert cfg.data.batch_size > 0
        assert cfg.data.input_channels == 4  # Default is RGB+DEM

        # Debug experiment
        cfg = compose(config_name="config", overrides=["experiment=debug"])
        assert cfg.training.max_epochs == 3

        # Synthetic RGB-only (3 channels)
        cfg = compose(
            config_name="config",
            overrides=["data=synthetic", "data.use_dem=false", "data.input_channels=3"],
        )
        assert cfg.data.input_channels == 3
        assert cfg.data.use_dem is False

        # Real data with DEM (4 channels)
        cfg = compose(config_name="config", overrides=["data=real"])
        assert cfg.data.input_channels == 4
        assert cfg.data.use_dem is True
