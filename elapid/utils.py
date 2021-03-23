"""Backend helper functions that don't need to be exposed to users"""
import multiprocessing as mp

import numpy as np

_ncpus = mp.cpu_count()


def repeat_array(x, length=1, axis=0):
    """
    Repeats a 1D numpy array along an axis to an arbitrary length

    :param x: the n-dimensional array to repeat
    :param length: the number of times to repeat the array
    :param axis: the axis along which to repeat the array (valid values include 0 to n+1)
    :return: an n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)


def _validate_feature_types(features):
    """
    Ensures the feature classes passed are legitimate
    :param features: a list or string that must be in ["linear", "quadratic", "product", "hinge", "threshold", "auto"] or string "lqphta"
    :return valid_features: a list of formatted valid features
    """
    valid_list = ["linear", "quadratic", "product", "hinge", "threshold"]
    valid_string = "lqpht"
    valid_features = list()

    assert features is not None, "Features cannot be empty"

    # ensure the string features are valid, and translate to a standard feature list
    if type(features) is str:
        for feature in features:

            if feature == "a":
                return valid_list

            assert feature in valid_string, "Invalid feature passed: {}".format(feature)

            if feature == "l":
                valid_features.append("linear")
            elif feature == "q":
                valid_features.append("quadratic")
            elif feature == "p":
                valid_features.append("product")
            elif feature == "h":
                valid_features.append("hinge")
            elif feature == "t":
                valid_features.append("threshold")

    # or ensure the list features are valid
    elif type(features) is list:
        for feature in features:
            if feature == "auto":
                return valid_list

            assert feature in valid_list, "Invalid feature passed: {}".format(feature)

            valid_features.append(feature)

    return valid_features
