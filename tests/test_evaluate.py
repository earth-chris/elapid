import numpy as np
import pytest
import matplotlib.pyplot as plt
from elapid.evaluate import boycei, boyce_index
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point



# Test Case 1: Normal case with random data
def test_normal_case():
    np.random.seed(0)
    fit = np.random.rand(1000)
    obs = np.random.choice(fit, size=100, replace=False)
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    assert 'Spearman.cor' in results
    assert 'F.ratio' in results
    spearman_cor = results['Spearman.cor']
    f_ratio = results['F.ratio']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    assert len(f_ratio) == 10
    assert not np.any(np.isnan(f_ratio))
    assert np.all(f_ratio >= 0)

# Test Case 2: Edge case with empty 'fit' array
def test_empty_fit():
    fit = np.array([])
    obs = np.array([0.5, 0.6, 0.7])
    with pytest.raises(ValueError):
        boyce_index(fit, obs, nclass=10, PEplot=False)

# Test Case 3: Edge case with empty 'obs' array
def test_empty_obs():
    fit = np.random.rand(1000)
    obs = np.array([])
    with pytest.raises(ValueError) as exc_info:
        boyce_index(fit, obs, nclass=10, PEplot=False)
    assert "After removing NaNs, 'fit' or 'obs' arrays cannot be empty." in str(exc_info.value)

# Test Case 4: 'obs' containing NaNs
def test_obs_with_nans():
    fit = np.random.rand(1000)
    obs = np.random.choice(fit, size=100, replace=False)
    obs[::10] = np.nan  # Introduce NaNs into 'obs'
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    spearman_cor = results['Spearman.cor']
    assert 'Spearman.cor' in results
    if not np.isnan(spearman_cor):
        assert -1 <= spearman_cor <= 1
    f_ratio = results['F.ratio']
    assert len(f_ratio) == 10

# Test Case 5: Invalid 'nclass' value (negative number)
def test_invalid_nclass():
    fit = np.random.rand(1000)
    obs = np.random.choice(fit, size=100, replace=False)
    with pytest.raises(ValueError):
        boyce_index(fit, obs, nclass=-5, PEplot=False)

# Test Case 6: Custom 'window' value
def test_custom_window():
    fit = np.random.rand(1000)
    obs = np.random.choice(fit, size=100, replace=False)
    results = boyce_index(fit, obs, window=0.1, PEplot=False)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results['F.ratio']
    assert len(f_ratio) > 0

# Test Case 7: 'PEplot' set to True
def test_peplot_true():
    fit = np.random.rand(1000)
    obs = np.random.choice(fit, size=100, replace=False)
    results = boyce_index(fit, obs, nclass=10, PEplot=True)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    plt.close('all')  # Close the plot to avoid display during testing

# Test Case 8: 'fit' containing NaNs
def test_fit_with_nans():
# In this code snippet:
    fit = np.random.rand(1000)
    fit[::50] = np.nan  # Introduce NaNs into 'fit'
    obs = np.random.choice(fit[~np.isnan(fit)], size=100, replace=False)
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results['F.ratio']
    assert len(f_ratio) == 10

# Test Case 9: 'obs' values outside the range of 'fit'
def test_obs_outside_fit_range():
    fit = np.random.rand(1000)
    obs = np.array([1.5, 2.0, 2.5])  # Values outside the range [0, 1]
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    spearman_cor = results['Spearman.cor']
    assert 'Spearman.cor' in results
    assert np.isnan(spearman_cor) or -1 <= spearman_cor <= 1
    f_ratio = results['F.ratio']
    assert len(f_ratio) == 10

# Test Case 10: Large dataset
def test_large_dataset():
    fit = np.random.rand(1000000)
    obs = np.random.choice(fit, size=10000, replace=False)
    results = boyce_index(fit, obs, nclass=20, PEplot=False)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1
    f_ratio = results['F.ratio']
    assert len(f_ratio) == 20


# Test Case 11: Using Pandas Series
def test_with_pandas_series():
    np.random.seed(0)
    fit = pd.Series(np.random.rand(1000))
    obs = fit.sample(n=100, replace=False)
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1

# Test Case 12: Using GeoPandas GeoSeries
def test_with_geopandas_geoseries():
    np.random.seed(0)
    num_points = 1000
    x = np.random.uniform(-180, 180, num_points)
    y = np.random.uniform(-90, 90, num_points)
    suitability = np.random.rand(num_points)
    geometry = [Point(xy) for xy in zip(x, y)]
    gdf = gpd.GeoDataFrame({'suitability': suitability}, geometry=geometry)
    
    fit = gdf['suitability']  # This is a Pandas Series
    obs = fit.sample(n=100, replace=False)
    results = boyce_index(fit, obs, nclass=10, PEplot=False)
    assert 'Spearman.cor' in results
    spearman_cor = results['Spearman.cor']
    assert not np.isnan(spearman_cor)
    assert -1 <= spearman_cor <= 1