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


def clamp(x):
    """
    Clamps feature data to the range of features previously estimated ranges
    """
    pass


def compute_lambdas(y, weights, reg, n_lambda=200):
    """
    Computes lambda parameter values for elastic lasso fits.

    :param y: pandas series or array with binary presence/background (1/0) values
    :param weights: per-sample model weights
    :param reg: per-feature regularization coefficients
    :param n_lambda: the number of lambda values to estimate
    :return lambdas: a numpy array of lambda scores of length n_lambda
    """
    n_presence = np.sum(y)
    mean_regularization = np.mean(reg)
    total_weight = np.sum(weights)
    seed_range = np.linspace(4, 0, n_lambda)
    lambdas = 10 ** (seed_range) * mean_regularization * (n_presence / total_weight)

    return lambdas


def compute_weights(y):
    """
    Uses Maxent's weight formulation to compute per-sample model weights.

    :param y: pandas series or 1d array with binary presence/background (1/0) values
    """
    weights = np.array(y + (1 - y) * 100)

    return weights
