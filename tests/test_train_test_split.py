import os

import geopandas as gpd
import numpy as np

from elapid import geo, train_test_split

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


def test_BufferedLeaveOneOut():
    # straight leave-one-out
    min_distance = 200
    bloo = train_test_split.BufferedLeaveOneOut(distance=min_distance)
    for train_idx, test_idx in bloo.split(points):
        train = points.iloc[train_idx]
        test = points.iloc[test_idx]
        distance = geo.nearest_point_distance(test, train)
        assert distance.min() >= min_distance
        assert len(train) > 0

    # grouped leave-one-out
    points["group"] = 0
    points.loc[30:40, "group"] = 1
    points.loc[40:, "group"] = 2
    train_idxs = []
    for train_idx, test_idx in bloo.split(points, groups="group"):
        train = points.iloc[train_idx]
        test = points.iloc[test_idx]
        distance = geo.nearest_point_distance(test, train)
        assert distance.min() >= min_distance
        assert len(train) > 0
        train_idxs.append(train_idx)
    assert len(train_idxs) == 3

    # only y==1 leave-one-out
    n_presence = 10
    points["class"] = 0
    points.loc[0 : n_presence - 1, "class"] = 1
    test_idxs = []
    for train_idx, test_idx in bloo.split(points, class_label="class"):
        train = points.iloc[train_idx]
        test = points.iloc[test_idx]
        distance = geo.nearest_point_distance(test, train)
        assert distance.min() >= min_distance
        assert len(train) > 0
        test_idxs.append(test_idx)
    print(test_idxs)
    assert len(test_idxs) == n_presence
