"""Utilities for calculating zonal statistics"""

from typing import Callable, List

import numpy as np
from scipy import stats as scistats


class RasterStat:
    name: str = None
    method: Callable = None
    dtype: str = None
    kwargs: dict = None

    def __init__(self, name: str, method: Callable, dtype: str = None, **kwargs):
        self.name = name
        self.method = method
        self.dtype = dtype
        self.kwargs = kwargs

    def format(self, x: np.ndarray):
        if x.ndim == 3:
            bands, rows, cols = x.shape
            x = x.reshape((bands, rows * cols))
        return x

    def apply(self, x: np.ndarray):
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
    mode_stats = scistats.mode(x, axis=1, nan_policy="omit")
    return mode_stats[0]


def raster_percentile(x, pctile):
    return np.nanpercentile(x, pctile)


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
) -> List[RasterStat]:
    """Return RasterStat configs for the requested stats calculations"""
    methods = []

    if mean:
        methods.append(RasterStat(name="mean", method=raster_mean, dtype="float32"))

    if min:
        methods.append(RasterStat(name="min", method=raster_min))

    if max:
        methods.append(RasterStat(name="max", method=raster_max))

    if count:
        methods.append(RasterStat(name="count", method=raster_count, dtype="int16"))

    if sum:
        methods.append(RasterStat(name="sum", method=raster_sum))

    if stdv:
        methods.append(RasterStat(name="stdv", method=raster_stdv))

    if skew:
        methods.append(RasterStat(name="skew", method=raster_skew))

    if kurtosis:
        methods.append(RasterStat(name="kurt", method=raster_kurtosis))

    if mode:
        methods.append(RasterStat(name="mode", method=raster_mode))

    if len(percentiles) > 0:
        for percentile in percentiles:
            methods.append(RasterStat(name=f"{percentile}pct", method=raster_percentile, pctile=percentile))

    return methods
