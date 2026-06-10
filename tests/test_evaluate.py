import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from elapid.evaluate import boyce_index, continuous_boyce_index, get_intervals


# Test Case 1: Normal case with random data
def test_normal_case():
    np.random.seed(0)
    ypred_background = np.random.rand(1000)
    ypred_observed = np.random.choice(ypred_background, size=100, replace=False)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10, to_plot=False)
    # Check outputs
    assert isinstance(f_scores, np.ndarray)
    assert isinstance(corr, float)
    assert isinstance(intervals, np.ndarray)
    assert len(f_scores) == 10
    assert not np.any(np.isnan(f_scores))
    assert -1 <= corr <= 1


# Test Case 2: Empty background array
def test_empty_background():
    ypred_background = np.array([])
    ypred_observed = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    assert "'ypred_background' must be a non-empty one-dimensional array." in str(exc_info.value)


# Test Case 3: Empty observed array
def test_empty_observed():
    ypred_background = np.random.rand(1000)
    ypred_observed = np.array([])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    assert "'ypred_observed' must be a non-empty one-dimensional array." in str(exc_info.value)


# Test Case 4: Observed containing NaNs
def test_observed_with_nans(recwarn):
    ypred_background = np.random.rand(1000)
    ypred_observed = np.random.choice(ypred_background, size=100, replace=False)
    ypred_observed[::10] = np.nan
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    # warning should be issued
    w = recwarn.pop(UserWarning)
    assert "'ypred_observed' contains NaN values, which will be ignored." in str(w.message)
    # outputs still valid
    assert len(f_scores) == 10
    assert np.isnan(corr) or -1 <= corr <= 1


# Test Case 5: Invalid bins value
def test_invalid_bins():
    ypred_background = np.random.rand(1000)
    ypred_observed = np.random.choice(ypred_background, size=100, replace=False)
    with pytest.raises(ValueError):
        continuous_boyce_index(ypred_observed, ypred_background, bins=1)


# Test Case 6: Custom float bin width
def test_custom_bin_width():
    ypred_background = np.random.rand(1000)
    ypred_observed = np.random.choice(ypred_background, size=100, replace=False)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=0.1)
    # number of bins should be ceil(range/0.1)
    expected_nbins = int(np.ceil((ypred_background.max() - ypred_background.min()) / 0.1))
    assert len(f_scores) == expected_nbins
    assert -1 <= corr <= 1 or np.isnan(corr)


# Test Case 7: Background containing NaNs
def test_background_with_nans(recwarn):
    ypred_background = np.random.rand(1000)
    ypred_background[::50] = np.nan
    ypred_observed = np.random.choice(ypred_background[~np.isnan(ypred_background)], size=100, replace=False)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    w = recwarn.pop(UserWarning)
    assert "'ypred_background' contains NaN values, which will be ignored." in str(w.message)
    assert len(f_scores) == 10


# Test Case 8: Observed outside background range
def test_observed_outside_range():
    ypred_background = np.random.rand(1000)
    ypred_observed = np.array([ypred_background.max() + 0.5, ypred_background.max() + 1.0])
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    assert len(f_scores) == 10
    # all f_scores should be zero or nan
    assert all((np.isnan(f) or f == 0) for f in f_scores)
    assert np.isnan(corr) or -1 <= corr <= 1


# Test Case 9: Large dataset performance
def test_large_dataset():
    ypred_background = np.random.rand(1000000)
    ypred_observed = np.random.choice(ypred_background, size=10000, replace=False)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=20)
    assert len(f_scores) == 20
    assert -1 <= corr <= 1


# Test Case 10: Pandas Series inputs
def test_with_pandas_series():
    np.random.seed(0)
    ypred_background = pd.Series(np.random.rand(1000))
    ypred_observed = ypred_background.sample(n=100)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    assert len(f_scores) == 10
    assert -1 <= corr <= 1


# Test Case 11: GeoPandas GeoSeries inputs
def test_with_geopandas_geoseries():
    np.random.seed(0)
    num_points = 500
    coords = np.random.rand(num_points, 2)
    suitability = np.random.rand(num_points)
    geometry = [Point(xy) for xy in coords]
    gdf = gpd.GeoDataFrame({"suitability": suitability}, geometry=geometry)
    ypred_background = gdf["suitability"]
    ypred_observed = ypred_background.sample(n=50)
    f_scores, corr, intervals = continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    assert len(f_scores) == 10
    assert -1 <= corr <= 1


# Test Case 12: Both with NaNs leading to empty after removal
def test_empty_after_nan_removal():
    ypred_background = np.array([np.nan, np.nan])
    ypred_observed = np.array([np.nan, np.nan])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(ypred_observed, ypred_background, bins=10)
    # after dropping NaNs, arrays empty
    assert "must be a non-empty one-dimensional array" in str(exc_info.value)
