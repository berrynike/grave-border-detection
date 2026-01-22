"""Smoke tests for Hydra configuration."""

from pathlib import Path

from hydra import compose, initialize_config_dir


def test_configs_load() -> None:
    """Test that all config combinations load without error."""
    config_dir = str(Path("configs").absolute())

    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        # Default config
        cfg = compose(config_name="config")
        assert cfg.data.batch_size > 0

        # Experiments
        cfg = compose(config_name="config", overrides=["experiment=debug"])
        assert cfg.training.max_epochs == 3

        cfg = compose(config_name="config", overrides=["experiment=baseline_rgb"])
        assert cfg.data.input_channels == 3
