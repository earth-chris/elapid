"""Backend helper functions that don't need to be exposed to users"""

import numpy as np
from sklearn.preprocessing import OneHotEncoder

from elapid.utils import repeat_array


def hingeval(x, mn, mx):
    """
    Computes hinge features
    """
    return np.minimum(1, np.maximum(0, (x - mn) / (mx - mn)))


def hinge(x, n_hinges=30, range=None):
    """"""
    mn = range[0] if range is not None else np.min(x)
    mx = range[1] if range is not None else np.max(x)
    k = np.linspace(mn, mx, n_hinges)

    xarr = repeat_array(x, len(k) - 1, axis=1)
    lharr = repeat_array(k[:-1], len(x), axis=0)
    rharr = repeat_array(k[1:], len(x), axis=0)

    lh = hingeval(xarr, lharr, mx)
    rh = hingeval(xarr, mn, rharr)

    return np.concatenate((lh, rh), axis=1)


def categorical(x, **kwargs):
    """"""
    ohe = OneHotEncoder(sparse=False, dtype=np.uint8)

    return ohe.fit_transform(x.reshape(-1, 1))


def threshold(x, n_thresholds=30, range=None):
    """"""
    mn = range[0] if range is not None else np.min(x)
    mx = range[1] if range is not None else np.max(x)
    k = np.linspace(mn, mx, n_thresholds + 2)[2:-2]

    xarr = repeat_array(x, len(k), axis=1)
    tarr = repeat_array(k, len(x), axis=0)

    return (xarr > tarr).astype(np.uint8)
