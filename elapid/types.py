"""Custom maxent and typing data types."""
from typing import Any, Union

import numpy as np
import pandas as pd
import pyproj

# typing
Number = Union[int, float]
ArrayLike = Union[np.array, pd.DataFrame]
CRSType = Union[pyproj.CRS, str]


# maxent feature transformations
def get_feature_types(return_string: bool = False) -> Union[list, str]:
    feature_types = "lqpht" if return_string else ["linear", "quadratic", "product", "hinge", "threshold"]
    return feature_types


def validate_feature_types(features: Union[str, list]) -> list:
    """Ensures the feature classes passed are maxent-legible

    Args:
        features: List or string that must be in ["linear", "quadratic", "product",
            "hinge", "threshold", "auto"] or string "lqphta"

    Returns:
        valid_features: List of formatted valid feature values
    """
    valid_list = get_feature_types(return_string=False)
    valid_string = get_feature_types(return_string=True)
    valid_features = list()

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


# data type checking
def validate_boolean(var: Any) -> bool:
    """Evaluates whether an argument is boolean True/False

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not boolean
    """
    assert isinstance(var, bool), "Argument must be True/False"
    return var


def validate_numeric_scalar(var: Any) -> bool:
    """Evaluates whether an argument is a single numeric value.

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not numeric.
    """
    assert isinstance(var, (int, float)), "Argument must be single numeric value"
    return var


def to_iterable(var: Any) -> list:
    """Checks and converts variables to an iterable type.

    Args:
        var: the input variable to check and convert.

    Returns:
        `var` wrapped in a list.
    """
    if not hasattr(var, "__iter__"):
        return [var]
    elif isinstance(var, (str)):
        return [var]
    else:
        return var
