"""Methods for geographlically splitting data into train/test splits"""

from typing import List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator

from elapid.types import Vector


def checkerboard_split(
    points: Vector, grid_size: float, buffer: float = 0, bounds: Tuple[float, float, float, float] = None
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Create train/test splits with a spatially-gridded checkerboard.

    Args:
        points: point-format GeoSeries or GeoDataFrame
        grid_size: the height and width of each checkerboard side to split
            data using. Should match the units of the points CRS
            (i.e. grid_size=1000 is a 1km grid for UTM data)
        buffer: add an x/y buffer around the initial checkerboard bounds
        bounds: instead of deriving the checkerboard bounds from `points`,
            use this tuple of [xmin, ymin, xmax, ymax] values.

    Returns:
        (train_points, test_points) split using a checkerboard grid.
    """
    if isinstance(points, gpd.GeoSeries):
        points = points.to_frame("geometry")

    bounds = points.total_bounds if bounds is None else bounds
    xmin, ymin, xmax, ymax = bounds

    x0s = np.arange(xmin - buffer, xmax + buffer + grid_size, grid_size)
    y0s = np.arange(ymin - buffer, ymax + buffer + grid_size, grid_size)

    train_cells = []
    test_cells = []
    for idy, y0 in enumerate(y0s):
        offset = 0 if idy % 2 == 0 else 1
        for idx, x0 in enumerate(x0s):
            cell = box(x0, y0, x0 + grid_size, y0 + grid_size)
            cell_type = 0 if (idx + offset) % 2 == 0 else 1
            if cell_type == 0:
                train_cells.append(cell)
            else:
                test_cells.append(cell)

    grid_crs = points.crs
    train_grid = gpd.GeoDataFrame(geometry=train_cells, crs=grid_crs)
    test_grid = gpd.GeoDataFrame(geometry=test_cells, crs=grid_crs)
    train_points = (
        gpd.sjoin(points, train_grid, how="left", predicate="within")
        .dropna()
        .drop(columns="index_right")
        .reset_index(drop=True)
    )
    test_points = (
        gpd.sjoin(points, test_grid, how="left", predicate="within")
        .dropna()
        .drop(columns="index_right")
        .reset_index(drop=True)
    )

    return train_points, test_points


class GeographicKFold(BaseCrossValidator):
    """Compute geographically-clustered train/test folds using KMeans clustering"""

    def __init__(self, n_splits: int = 4):
        self.n_splits = n_splits

    def split(self, points: Vector) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
        """Split point data into geographically-clustered train/test folds and
            return their array indices.

        Args:
            points: point-format GeoSeries or GeoDataFrame.

        Yields:
            (train_idxs, test_idxs) the train/test splits for each geo fold.
        """
        for train, test in super().split(points):
            yield train, test

    def _iter_test_indices(self, X: Vector, y: None = None, groups: None = None):
        """The method used by the base class to split train/test data"""
        kmeans = KMeans(n_clusters=self.n_splits)
        xy = np.array(list(zip(X.geometry.x, X.geometry.y)))
        kmeans.fit(xy)
        clusters = kmeans.predict(xy)
        indices = np.arange(len(xy))
        for cluster in range(self.n_splits):
            test = clusters == cluster
            yield indices[test]

    def get_n_splits(self, X: None = None, y: None = None, groups: None = None) -> int:
        """Returns the number of splitting iterations in the cross-validator

        Args:
            X: ignored, exists for compatibility.
            y: ignored, exists for compatibility.
            groups: ignored, exists for compatibility.

        Returns:
            The number of splitting iterations in the cross-validator.
        """
        return self.n_splits
