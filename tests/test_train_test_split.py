import os

import geopandas as gpd
import numpy as np

from elapid import train_test_split

# set the test raster data paths
directory_path, script_path = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(directory_path, "data")
points = gpd.read_file(os.path.join(data_path, "test-point-samples.gpkg"))


def test_checkerboard_split():
    train, test = train_test_split.checkerboard_split(points, grid_size=1000)
    assert isinstance(train, gpd.GeoDataFrame)

    buffer = 500
    xmin, ymin, xmax, ymax = points.total_bounds
    buffered_bounds = [xmin - buffer, ymin - buffer, xmax + buffer, ymax + buffer]
    train_buffered, test_buffered = train_test_split.checkerboard_split(points, grid_size=1000, bounds=buffered_bounds)
    assert len(train_buffered) > len(train)


def test_GeographicKFold():
    n_folds = 4
    gfolds = train_test_split.GeographicKFold(n_splits=n_folds)
    counted_folds = 0
    for train_idx, test_idx in gfolds.split(points):
        train = points.iloc[train_idx]
        test = points.iloc[test_idx]
        assert len(train) > len(test)
        counted_folds += 1
    assert gfolds.get_n_splits() == n_folds == counted_folds
