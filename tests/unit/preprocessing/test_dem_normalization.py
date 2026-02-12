"""Tests for DEM normalization methods."""

import numpy as np
import pytest
from numpy.typing import NDArray

from grave_border_detection.preprocessing.dem_normalization import (
    LocalHeightConfig,
    NormalizationMethod,
    PercentileClipConfig,
    normalize_dem,
    normalize_local_height,
    normalize_percentile_clip,
    normalize_robust_zscore,
    normalize_slope,
    normalize_zscore,
)


@pytest.fixture
def sample_dem() -> NDArray[np.float32]:
    """Create sample DEM with building and grave features."""
    # Base elevation ~440m
    dem = np.full((64, 64), 440.0, dtype=np.float32)

    # Add terrain slope (1m across image)
    for i in range(64):
        dem[i, :] += i * (1.0 / 64)

    # Add "building" in corner (10m tall)
    dem[5:15, 5:15] = 450.0

    # Add subtle "grave" (30cm raised)
    dem[38:44, 38:46] = 440.3

    return dem


@pytest.fixture
def flat_dem() -> NDArray[np.float32]:
    """Create flat DEM for edge case testing."""
    return np.full((32, 32), 440.0, dtype=np.float32)


class TestZScore:
    """Tests for z-score normalization."""

    def test_mean_zero(self, sample_dem: NDArray[np.float32]) -> None:
        """Z-score output should have mean ~0."""
        result = normalize_zscore(sample_dem)
        assert result.mean() == pytest.approx(0.0, abs=1e-5)

    def test_std_one(self, sample_dem: NDArray[np.float32]) -> None:
        """Z-score output should have std ~1."""
        result = normalize_zscore(sample_dem)
        assert result.std() == pytest.approx(1.0, abs=1e-5)

    def test_shape_preserved(self, sample_dem: NDArray[np.float32]) -> None:
        """Output shape should match input."""
        result = normalize_zscore(sample_dem)
        assert result.shape == sample_dem.shape

    def test_dtype_float32(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should be float32."""
        result = normalize_zscore(sample_dem)
        assert result.dtype == np.float32


class TestPercentileClip:
    """Tests for percentile clipping normalization."""

    def test_output_range(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should be in [0, 1] range."""
        result = normalize_percentile_clip(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_custom_percentiles(self, sample_dem: NDArray[np.float32]) -> None:
        """Custom percentiles should work."""
        config = PercentileClipConfig(lower=5.0, upper=95.0)
        result = normalize_percentile_clip(sample_dem, config)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_shape_preserved(self, sample_dem: NDArray[np.float32]) -> None:
        """Output shape should match input."""
        result = normalize_percentile_clip(sample_dem)
        assert result.shape == sample_dem.shape


class TestLocalHeight:
    """Tests for local height above ground normalization."""

    def test_output_range(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should be in [0, 1] range."""
        result = normalize_local_height(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_building_clipped(self, sample_dem: NDArray[np.float32]) -> None:
        """Building should be clipped to max value."""
        config = LocalHeightConfig(max_height=1.0, window_size=16)
        result = normalize_local_height(sample_dem, config)
        # Building (10m above ground) should be clipped to 1.0
        assert result[10, 10] == pytest.approx(1.0, abs=0.1)

    def test_different_max_height(self, sample_dem: NDArray[np.float32]) -> None:
        """Different max_height values should produce different scaling."""
        config_small = LocalHeightConfig(max_height=0.5, window_size=16)
        config_large = LocalHeightConfig(max_height=2.0, window_size=16)
        result_small = normalize_local_height(sample_dem, config_small)
        result_large = normalize_local_height(sample_dem, config_large)
        # The building (10m tall) is clipped in both cases, but intermediate
        # heights near the building edge will differ due to different scaling.
        # For pixels with height_above_ground between 0.5 and 2.0, the small
        # config clips to 1.0 while large config gives a fractional value.
        # This means result_small should have more 1.0 values OR the same
        # but never fewer, and the overall distributions should differ.
        assert not np.allclose(result_small, result_large)

    def test_ground_near_zero(self, sample_dem: NDArray[np.float32]) -> None:
        """Flat ground away from features should be near 0."""
        config = LocalHeightConfig(window_size=16)
        result = normalize_local_height(sample_dem, config)
        # Flat ground area should be near 0
        assert result[55, 55] < 0.2

    def test_shape_preserved(self, sample_dem: NDArray[np.float32]) -> None:
        """Output shape should match input."""
        result = normalize_local_height(sample_dem)
        assert result.shape == sample_dem.shape


class TestSlope:
    """Tests for slope normalization."""

    def test_output_range(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should be in [0, 1] range."""
        result = normalize_slope(sample_dem)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_edges_detected(self, sample_dem: NDArray[np.float32]) -> None:
        """Edges should have higher slope than flat areas."""
        result = normalize_slope(sample_dem)
        # Edge of building should have higher slope than interior
        edge_slope = result[5, 10]  # building edge
        interior_slope = result[10, 10]  # building interior (flat roof)
        assert edge_slope > interior_slope

    def test_flat_dem_low_slope(self, flat_dem: NDArray[np.float32]) -> None:
        """Flat DEM should have very low slope values."""
        result = normalize_slope(flat_dem)
        # All values should be near 0 for flat terrain
        assert result.max() < 0.1

    def test_shape_preserved(self, sample_dem: NDArray[np.float32]) -> None:
        """Output shape should match input."""
        result = normalize_slope(sample_dem)
        assert result.shape == sample_dem.shape


class TestRobustZScore:
    """Tests for robust z-score normalization."""

    def test_shape_preserved(self, sample_dem: NDArray[np.float32]) -> None:
        """Output shape should match input."""
        result = normalize_robust_zscore(sample_dem)
        assert result.shape == sample_dem.shape

    def test_dtype_float32(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should be float32."""
        result = normalize_robust_zscore(sample_dem)
        assert result.dtype == np.float32

    def test_finite_values(self, sample_dem: NDArray[np.float32]) -> None:
        """Output should have no NaN or Inf values."""
        result = normalize_robust_zscore(sample_dem)
        assert np.isfinite(result).all()


class TestNormalizeDem:
    """Tests for the main normalize_dem dispatcher function."""

    @pytest.mark.parametrize("method", list(NormalizationMethod))
    def test_all_methods_work(
        self, sample_dem: NDArray[np.float32], method: NormalizationMethod
    ) -> None:
        """All normalization methods should run without error."""
        result = normalize_dem(sample_dem, method)
        assert result.shape == sample_dem.shape
        assert np.isfinite(result).all()

    def test_string_method_name(self, sample_dem: NDArray[np.float32]) -> None:
        """Should accept string method names."""
        result = normalize_dem(sample_dem, "zscore")
        assert result.shape == sample_dem.shape

    def test_unknown_method_raises(self, sample_dem: NDArray[np.float32]) -> None:
        """Unknown method should raise ValueError."""
        with pytest.raises(ValueError, match="is not a valid NormalizationMethod"):
            normalize_dem(sample_dem, "invalid_method")  # type: ignore[arg-type]

    def test_kwargs_passed(self, sample_dem: NDArray[np.float32]) -> None:
        """Method-specific kwargs should be passed through."""
        result = normalize_dem(
            sample_dem,
            "local_height",
            window_size=16,
            max_height=1.0,
        )
        assert result.shape == sample_dem.shape

    def test_local_height_params(self, sample_dem: NDArray[np.float32]) -> None:
        """Local height parameters should affect output."""
        result_small_window = normalize_dem(
            sample_dem, "local_height", window_size=8, max_height=0.5
        )
        result_large_window = normalize_dem(
            sample_dem, "local_height", window_size=32, max_height=0.5
        )
        # Different window sizes should produce different results
        assert not np.allclose(result_small_window, result_large_window)
