"""Backend helper functions that don't need to be exposed to users"""
import multiprocessing as mp

import numpy as np

_ncpus = mp.cpu_count()

MAXENT_DEFAULTS = {
    "clamp": True,
    "beta_multiplier": 1.0,
    "beta_hinge": 1.0,
    "beta_lqp": 1.0,
    "beta_threshold": 1.0,
    "beta_categorical": 1.0,
    "feature_types": ["linear", "hinge", "product"],
    "n_hinge_features": 50,
    "n_threshold_features": 50,
    "scorer": "roc_auc",
    "tau": 0.5,
    "tolerance": 1e-7,
    "use_lambdas": "last",
}


def repeat_array(x, length=1, axis=0):
    """
    Repeats a 1D numpy array along an axis to an arbitrary length

    :param x: the n-dimensional array to repeat
    :param length: the number of times to repeat the array
    :param axis: the axis along which to repeat the array (valid values include 0 to n+1)
    :returns: an n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)
