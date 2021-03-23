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

    :param y: pandas series or 1d array with binary presence/background (1/0) values
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


def compute_regularization(
    f, y, beta_multiplier=1.0, beta_lqp=1.0, beta_threshold=1.0, beta_hinge=1.0, beta_categorical=1.0
):
    """
    Computes variable regularization values for all feature data.

    :param f: pandas dataframe with feature transformations applied
    :param y: pandas series with binary presence/background (1/0) values
    :returns reg: a numpy array with per-feature regularization parameters
    """

    # tailor the regularization to presence-only locations
    mm = f[y == 1]
    n_points = len(mm)
    features = list(f.columns)
    n_features = len(features)
    regularization = np.zeros(n_features)

    # set the default regularization values
    q_features = len([i for i in features if "_squared" in i])
    p_features = len([i for i in features if "_x_" in i])
    if q_features > 0:
        regtable = [[0, 10, 17, 30, 100], [1.3, 0.8, 0.5, 0.25, 0.05]]
    elif p_features > 0:
        regtable = [[0, 10, 17, 30, 100], [2.6, 1.6, 0.9, 0.55, 0.05]]
    else:
        regtable = [[0, 10, 30, 100], [1, 1, 0.2, 0.05]]

    for i, feature in enumerate(features):

        if "_linear" in feature or "_squared" in feature or "_x_" in feature:
            freg = regtable
            multiplier = beta_lqp
        elif "_hinge" in feature:
            freg = [[0, 1], [0.5, 0.5]]
            multiplier = beta_hinge
        elif "_threshold" in feature:
            freg = [[0, 100], [2, 1]]
            multiplier = beta_threshold
        elif "_class" in feature:
            freg = [[0, 10, 17], [0.65, 0.5, 0.25]]
            multiplier = beta_categorical

        ap = np.interp(n_points, freg[0], freg[1])
        regularization[i] = multiplier * ap / np.sqrt(n_points)

    # increase regularization for extreme hinge values
    hinge_features = [i for i in features if "_hinge_" in i]
    hinge_reg = np.zeros(n_features)
    for hinge_feature in hinge_features:
        hinge_idx = features.index(hinge_feature)
        std = np.max([np.std(mm[hinge_feature], ddof=1), (1 / np.sqrt(n_points))])
        hinge_reg[hinge_idx] = (0.5 * std) / np.sqrt(n_points)

    # increase threshold regularization for uniform values
    threshold_features = [i for i in features if "_threshold_" in i]
    threshold_reg = np.zeros(n_features)
    for threshold_feature in threshold_features:
        threshold_idx = features.index(threshold_feature)
        all_zeros = np.all(mm[threshold_feature] == 0)
        all_ones = np.all(mm[threshold_feature] == 1)
        threshold_reg[threshold_idx] = 1 if all_zeros or all_ones else 0

    # report the max regularization value
    default_reg = 0.001 * (np.max(f, axis=0) - np.min(f, axis=0))
    variance_reg = np.std(mm, axis=0, ddof=1) * regularization
    max_reg = np.max([default_reg, variance_reg, hinge_reg, threshold_reg], axis=0)

    # and scale it
    max_reg *= beta_multiplier

    return max_reg
