import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import Point

from elapid.evaluate import boyce_index, continuous_boyce_index


# Test Case 1: Normal case with random data
def test_normal_case():
    np.random.seed(0)
    ypred = np.random.rand(1000)
    yobs = np.random.choice(ypred, size=100, replace=False)
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "Spearman.cor" in results
    assert "F.ratio" in results
    spearman_cor = results["Spearman.cor"]
    f_ratio = results["F.ratio"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    assert len(f_ratio) == 10
    assert not np.any(np.isnan(f_ratio))
    assert np.all(f_ratio >= 0)


# Test Case 2: Edge case with empty 'ypred' array
def test_empty_ypred():
    ypred = np.array([])
    yobs = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "'ypred' or 'yobs' arrays cannot be empty." in str(exc_info.value)


# Test Case 3: Edge case with empty 'yobs' array
def test_empty_yobs():
    ypred = np.random.rand(1000)
    yobs = np.array([])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "'ypred' or 'yobs' arrays cannot be empty." in str(exc_info.value)


# Test Case 4: 'yobs' containing NaNs
def test_yobs_with_nans(recwarn):
    ypred = np.random.rand(1000)
    yobs = np.random.choice(ypred, size=100, replace=False)
    yobs[::10] = np.nan  # Introduce NaNs into 'yobs'
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    # Check for warnings
    w = recwarn.pop(UserWarning)
    assert "'yobs' contains NaN values, which will be ignored." in str(w.message)
    # Ensure function outputs are as expected
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    if not np.isnan(spearman_cor):
        assert -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) == 10


# Test Case 5: Invalid 'nbins' value (negative number)
def test_invalid_nbins():
    ypred = np.random.rand(1000)
    yobs = np.random.choice(ypred, size=100, replace=False)
    with pytest.raises(ValueError):
        continuous_boyce_index(yobs, ypred, nbins=-5, to_plot=False)


# Test Case 6: Custom 'bin_size' value
def test_custom_bin_size():
    ypred = np.random.rand(1000)
    yobs = np.random.choice(ypred, size=100, replace=False)
    results = continuous_boyce_index(yobs, ypred, bin_size=0.1, to_plot=False)
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) > 0


# Test Case 7: 'ypred' containing NaNs
def test_ypred_with_nans(recwarn):
    ypred = np.random.rand(1000)
    ypred[::50] = np.nan  # Introduce NaNs into 'ypred'
    yobs = np.random.choice(ypred[~np.isnan(ypred)], size=100, replace=False)
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    # Check for warnings
    w = recwarn.pop(UserWarning)
    assert "'ypred' contains NaN values, which will be ignored." in str(w.message)
    # Ensure function outputs are as expected
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) == 10


# Test Case 8: 'yobs' values outside the range of 'ypred'
def test_yobs_outside_ypred_range():
    ypred = np.random.rand(1000)
    yobs = np.array([1.5, 2.0, 2.5])  # Values outside the range [0, 1]
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    # Spearman correlation may be NaN due to insufficient valid data
    assert np.isnan(spearman_cor) or -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) == 10


# Test Case 9: Large dataset
def test_large_dataset():
    ypred = np.random.rand(1000000)
    yobs = np.random.choice(ypred, size=10000, replace=False)
    results = continuous_boyce_index(yobs, ypred, nbins=20, to_plot=False)
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) == 20


# Test Case 10: Using Pandas Series
def test_with_pandas_series():
    np.random.seed(0)
    ypred = pd.Series(np.random.rand(1000))
    yobs = ypred.sample(n=100, replace=False)
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1


# Test Case 11: Using GeoPandas GeoSeries
def test_with_geopandas_geoseries():
    np.random.seed(0)
    num_points = 1000
    x = np.random.uniform(-180, 180, num_points)
    y = np.random.uniform(-90, 90, num_points)
    suitability = np.random.rand(num_points)
    geometry = [Point(xy) for xy in zip(x, y)]
    gdf = gpd.GeoDataFrame({"suitability": suitability}, geometry=geometry)

    ypred = gdf["suitability"]  # This is a Pandas Series
    yobs = ypred.sample(n=100, replace=False)
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1


# Test Case 12: Both 'ypred' and 'yobs' containing NaNs
def test_both_ypred_yobs_with_nans(recwarn):
    ypred = np.random.rand(1000)
    ypred[::50] = np.nan  # Introduce NaNs into 'ypred'
    yobs = np.random.choice(ypred, size=100, replace=False)
    yobs[::10] = np.nan  # Introduce NaNs into 'yobs'
    results = continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    # Check for multiple warnings
    warnings_list = [str(w.message) for w in recwarn.list]
    assert "'ypred' contains NaN values, which will be ignored." in warnings_list
    assert "'yobs' contains NaN values, which will be ignored." in warnings_list
    # Ensure function outputs are as expected
    assert "Spearman.cor" in results
    spearman_cor = results["Spearman.cor"]
    if not np.isnan(spearman_cor):
        assert -1 <= spearman_cor <= 1
    f_ratio = results["F.ratio"]
    assert len(f_ratio) == 10


# Test Case 13: Empty arrays after removing NaNs
def test_empty_arrays_after_nan_removal():
    ypred = np.array([np.nan, np.nan])
    yobs = np.array([np.nan, np.nan])
    with pytest.raises(ValueError) as exc_info:
        continuous_boyce_index(yobs, ypred, nbins=10, to_plot=False)
    assert "'ypred' or 'yobs' arrays cannot be empty." in str(exc_info.value)
