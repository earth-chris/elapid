"""Methods for geographlically splitting data into train/test splits"""

from typing import List, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box
from sklearn.cluster import KMeans
from sklearn.model_selection import BaseCrossValidator
from sklearn.utils.validation import _num_samples

from elapid.geo import nearest_point_distance
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
        """Cluster x/y points into separate cross-validation folds.

        Args:
            n_splits: Number of geographic clusters to split the data into.
        """
        self.n_splits = n_splits

    def _iter_test_indices(self, points: Vector, y: None = None, groups: None = None):
        """Generate indices for test data samples."""
        kmeans = KMeans(n_clusters=self.n_splits)
        xy = np.array(list(zip(points.geometry.x, points.geometry.y)))
        kmeans.fit(xy)
        clusters = kmeans.predict(xy)
        indices = np.arange(len(xy))
        for cluster in range(self.n_splits):
            test = clusters == cluster
            yield indices[test]

    def split(self, points: Vector) -> Tuple[np.ndarray, np.ndarray]:
        """Split point data into geographically-clustered train/test folds and
            return their array indices.

        Args:
            points: point-format GeoSeries or GeoDataFrame.

        Yields:
            (train_idxs, test_idxs) the train/test splits for each geo fold.
        """
        for train, test in super().split(points):
            yield train, test

    def get_n_splits(self) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Returns:
            Splitting iteration count.
        """
        return self.n_splits


class BufferedLeaveOneOut(BaseCrossValidator):
    """Leave-one-out CV that excludes training points within a buffered distance."""

    def __init__(self, distance: float):
        """Buffered leave-one-out cross-validation strategy.

        Drops points from the training data based on a buffered distance
            to the left-out test point(s). Implemented from Ploton et al. 2020,
            https://www.nature.com/articles/s41467-020-18321-y

        Args:
            distance: drop training data points within this distance of test data.
        """
        self.distance = distance

    def _group_idxs(
        self, points: Vector, class_label: str = None, groups: str = None, count: bool = False
    ) -> List[int]:
        """Get test indices for grouped train/test splits."""
        if class_label is not None:
            in_class = points[class_label] == 1
            points = points.iloc[in_class]

        unique = points[groups].unique()
        if count:
            return len(unique)

        all_idxs = np.arange(len(points))
        test_idxs = []
        for group in unique:
            in_group = points[groups] == group
            test_idxs.append(all_idxs[in_group])

        return test_idxs

    def _point_idxs(self, points: Vector, class_label: str = None, count: bool = False) -> List[int]:
        """Get test indices for single point train/test splits."""
        if class_label is None:
            if count:
                return len(points)
            else:
                return range(len(points))

        else:
            in_class = points[class_label] == 1
            if count:
                return in_class.sum()
            else:
                return np.where(in_class)[0]

    def _iter_test_indices(self, points: Vector, class_label: str = None, groups: str = None, y: None = None):
        """Generate indices for test data samples."""
        if groups is None:
            test_idxs = self._point_idxs(points, class_label)

        else:
            test_idxs = self._group_idxs(points, class_label, groups)

        for indices in test_idxs:
            yield indices

    def _iter_test_masks(self, points: Vector, class_label: str = None, groups: str = None):
        """Generates boolean masks corresponding to test sets."""
        for test_index in self._iter_test_indices(points, class_label, groups):
            test_mask = np.zeros(_num_samples(points), dtype=bool)
            test_mask[test_index] = True
            yield test_mask

    def split(self, points: Vector, class_label: str = None, groups: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """Split point data into train/test folds and return their array indices.

        Default behaviour is to perform leave-one-out cross-validation, meaning
            there will be as many train/test splits as there are samples.
            To run leave-one-out splits for each y==1 sample, use the
            `class_label` parameter to define which column includes the class
            to leave out. To run a grouped leave-one-out, use the `groups`
            parameter to define which column includes unique IDs to group by.

        Args:
            points: point-format GeoSeries or GeoDataFrame.
            class_label: column to specify presence locations (y==1).
            groups: column to group train/test splits by.

        Yields:
            (train_idxs, test_idxs) the train/test splits for each fold.
        """
        n_samples = len(points)
        indices = np.arange(n_samples)
        for test_index in self._iter_test_masks(points, class_label, groups):
            train_idx = indices[np.logical_not(test_index)]
            test_idx = indices[test_index]
            train_pts = points.iloc[train_idx]
            test_pts = points.iloc[test_idx]
            distances = nearest_point_distance(test_pts, train_pts)
            in_range = distances > self.distance
            buffered_train_idx = train_idx[in_range]
            yield buffered_train_idx, test_idx

    def get_n_splits(self, points: Vector, class_label: str = None, groups: str = None) -> int:
        """Return the number of splitting iterations in the cross-validator.

        Args:
            points: point-format GeoSeries or GeoDataFrame.
            class_label: column to specify presence locations (y==1).
            groups: column to group train/test splits by.

        Returns:
            Splitting iteration count.
        """
        if groups is None:
            return self._point_idxs(points, class_label, count=True)
        else:
            return self._group_idxs(points, class_label, groups, count=True)
