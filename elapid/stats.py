"""Utilities for calculating zonal statistics"""

from typing import Callable, List

import numpy as np
from scipy import stats as scistats


class RasterStat:
    """Utility class to iterate over and apply reductions to multiband arrays"""

    name: str = None
    method: Callable = None
    dtype: str = None
    kwargs: dict = None

    def __init__(self, name: str, method: Callable, dtype: str = None, **kwargs):
        """Create a RasterStat object

        Args:
            name: the label to prepend to the output column
            method: function to reduce a 2d ndarray to an (nbands,) shape array
            dtype: the output array data type
            **kwargs: additional arguments to pass to `method`
        """
        self.name = name
        self.method = method
        self.dtype = dtype
        self.kwargs = kwargs

    def format(self, x: np.ndarray) -> np.ndarray:
        """Format the array data into an array of shape [nbands, n_valid_pixels]

        Args:
            x: ndarray of shape (nbands, nrows, ncols) or (nbands, n_valid_pixels)

        Returns:
            2d ndarray
        """
        if x.ndim == 3:
            bands, rows, cols = x.shape
            x = x.reshape((bands, rows * cols))
        return x

    def reduce(self, x: np.ndarray) -> np.ndarray:
        """Reduce an array using the objects `method` function

        Args:
            x: ndarray of shape (nbands, n_valid_pixels)

        Returns:
            ndarray of shape (nbands,)
        """
        return self.method(self.format(x), **self.kwargs)


def raster_mean(x):
    return np.nanmean(x, axis=1)


def raster_min(x):
    return np.nanmin(x, axis=1)


def raster_max(x):
    return np.nanmax(x, axis=1)


def raster_count(x):
    return x.shape[1]


def raster_sum(x):
    return np.nansum(x, axis=1)


def raster_stdv(x):
    return np.nanstd(x, axis=1)


def raster_skew(x):
    return scistats.skew(x, axis=1, nan_policy="omit")


def raster_kurtosis(x):
    return scistats.kurtosis(x, axis=1, nan_policy="omit")


def raster_mode(x):
    summary = scistats.mode(x, axis=1, nan_policy="omit")
    return summary.mode.flatten()


def raster_percentile(x, pctile):
    return np.nanpercentile(x, pctile, axis=1)


def get_raster_stats_methods(
    mean: bool = True,
    min: bool = False,
    max: bool = False,
    count: bool = False,
    sum: bool = False,
    stdv: bool = False,
    skew: bool = False,
    kurtosis: bool = False,
    mode: bool = False,
    percentiles: list = [],
    all: bool = False,
) -> List[RasterStat]:
    """Return RasterStat configs for the requested stats calculations"""
    methods = []

    if mean or all:
        methods.append(RasterStat(name="mean", method=raster_mean, dtype="float32"))

    if min or all:
        methods.append(RasterStat(name="min", method=raster_min))

    if max or all:
        methods.append(RasterStat(name="max", method=raster_max))

    if count or all:
        methods.append(RasterStat(name="count", method=raster_count, dtype="int16"))

    if sum or all:
        methods.append(RasterStat(name="sum", method=raster_sum))

    if stdv or all:
        methods.append(RasterStat(name="stdv", method=raster_stdv))

    if skew or all:
        methods.append(RasterStat(name="skew", method=raster_skew))

    if kurtosis or all:
        methods.append(RasterStat(name="kurt", method=raster_kurtosis))

    if mode or all:
        methods.append(RasterStat(name="mode", method=raster_mode))

    if len(percentiles) > 0:
        for percentile in percentiles:
            methods.append(RasterStat(name=f"{percentile}pct", method=raster_percentile, pctile=percentile))

    return methods
