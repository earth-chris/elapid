import math
import warnings
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def plot_PE_curve(x: Union[np.ndarray, List[float]], y: Union[np.ndarray, List[float]]) -> None:
    """
    Plots the Predicted/Expected (P/E) curve for the Boyce Index.

    Args:
        x (array-like): Habitat suitability values (e.g., interval midpoints).
        y (array-like): Predicted/Expected ratios corresponding to the habitat suitability intervals.

    Returns:
        None
    """
    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Habitat suitability")
    plt.ylabel("Predicted/Expected ratio")
    plt.title("Boyce Index")
    plt.show()


def get_intervals(
    bins: Union[int, float, List[float], np.ndarray, str] = "default",
    value_range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    Generates habitat suitability intervals based on a 'bins' parameter.
    Args:
        bins (int | float | list | np.ndarray | str, optional): Defines the binning strategy.
            - float: Bin width. Number of bins will be computed as ceil(range / width).
            - int: Number of bins.
            - list or ndarray: Custom bin edges. Must contain at least 2 values.
            - "default": Uses 10 equally spaced bins. Default is "default".
        value_range (tuple): The (min, max) range over which to create intervals. Default is (0, 1).

    Returns:
        np.ndarray: An (N, 2) array of bin intervals, where each row is (lower_bound, upper_bound). N = total number of bins.
    """
    mini, maxi = value_range

    if isinstance(bins, float):
        div_result = (maxi - mini) / bins
        nbins = int(np.ceil(div_result))
        if not np.isclose(div_result, nbins):
            warnings.warn(
                f"Bin size results in a non-integer number of bins. Using ceil: {div_result} -> {nbins}",
                UserWarning,
            )
        boundaries = np.linspace(mini, maxi, nbins + 1)
        intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    elif isinstance(bins, (list, np.ndarray)):
        bins = np.sort(np.array(bins))
        if len(bins) < 2:
            raise ValueError("bins list must have at least two elements.")
        intervals = np.column_stack((bins[:-1], bins[1:]))
    elif isinstance(bins, int) and bins > 1:
        boundaries = np.linspace(mini, maxi, bins + 1)
        intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    elif bins == "default":
        boundaries = np.linspace(mini, maxi, 11)
        intervals = np.column_stack((boundaries[:-1], boundaries[1:]))
    else:
        raise ValueError("Invalid `bins` value. Must be a float (width), int (count), or list of edges.")

    return intervals


def boyce_index(
    ypred_observed: np.ndarray, ypred_background: np.ndarray, interval: Union[Tuple[float, float], List[float]]
) -> float:
    """
    Calculates the Boyce index for a given interval.

    Uses the convention as defined in Hirzel et al. 2006 to compute the ratio of observed to expected frequencies.

    Args:
        ypred_observed (np.ndarray): Suitability values at observed locations (e.g., predictions at presence points).
        ypred_background (np.ndarray): Suitability values at random locations (e.g., predictions at background points).
        interval (tuple or list): Two elements representing the lower and upper bounds of the interval (i.e., habitat suitability).

    Returns:
        float: The ratio of observed to expected frequencies for the given interval.
    """
    lower, upper = interval
    yobs_bin = (ypred_observed >= lower) & (ypred_observed < upper)
    ypred_bin = (ypred_background >= lower) & (ypred_background < upper)

    # Include upper edge for last interval
    if np.isclose(upper, np.max(ypred_background)):
        yobs_bin |= ypred_observed == upper
        ypred_bin |= ypred_background == upper

    pi = np.sum(yobs_bin) / len(ypred_observed)
    ei = np.sum(ypred_bin) / len(ypred_background)

    return np.nan if ei == 0 else pi / ei


def continuous_boyce_index(
    ypred_observed: Union[np.ndarray, pd.Series, gpd.GeoSeries],
    ypred_background: Union[np.ndarray, pd.Series, gpd.GeoSeries],
    bins: Union[int, float, List[float], np.ndarray, str] = "default",
    to_plot: bool = False,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Compute the continuous Boyce index to evaluate habitat suitability models.

    Uses the convention as defined in Hirzel et al. 2006 to compute the ratio of observed to expected frequencies.

    Args:
        ypred_observed (numpy.ndarray | pd.Series | gpd.GeoSeries): Suitability values at observed locations (i.e., presence points).
        ypred_background (numpy.ndarray | pd.Series | gpd.GeoSeries): Suitability values at random/background locations.
        bins (int | float | list | np.ndarray | str, optional): Defines the binning strategy:
            - int: number of bins
            - float: bin width
            - list/ndarray: custom bin edges
            - 'default': 10 equally spaced bins over the prediction range
        to_plot (bool, optional): Whether to plot the predicted-to-expected (P/E) curve. Defaults to False.

    Returns:
        Tuple:
            - f_scores (numpy.ndarray): The Boyce index scores for each interval.
            - corr (float): Spearman correlation coefficient between the P/E ratios and the midpoints of the intervals.
            - intervals (numpy.ndarray): The intervals used for the Boyce index calculation.
    """
    if not isinstance(ypred_background, (np.ndarray, pd.Series, gpd.GeoSeries)):
        raise TypeError(
            "The 'ypred_background' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries."
        )
    if not isinstance(ypred_observed, (np.ndarray, pd.Series, gpd.GeoSeries)):
        raise TypeError("The 'ypred_observed' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries.")

    # Remove NaNs
    if np.isnan(ypred_background).any():
        warnings.warn("'ypred_background' contains NaN values, which will be ignored.", UserWarning)
        ypred_background = ypred_background[~np.isnan(ypred_background)]
    if np.isnan(ypred_observed).any():
        warnings.warn("'ypred_observed' contains NaN values, which will be ignored.", UserWarning)
        ypred_observed = ypred_observed[~np.isnan(ypred_observed)]

    ypred_background = np.asarray(ypred_background)
    ypred_observed = np.asarray(ypred_observed)

    if ypred_background.ndim != 1 or len(ypred_background) == 0:
        raise ValueError("'ypred_background' must be a non-empty one-dimensional array.")
    if ypred_observed.ndim != 1 or len(ypred_observed) == 0:
        raise ValueError("'ypred_observed' must be a non-empty one-dimensional array.")

    mini, maxi = np.min(ypred_background), np.max(ypred_background)
    intervals = get_intervals(bins, value_range=(mini, maxi))

    f_scores = np.array([boyce_index(ypred_observed, ypred_background, interval) for interval in intervals])

    valid = ~np.isnan(f_scores)
    f_valid = f_scores[valid]
    intervals_mid = np.mean(intervals[valid], axis=1)

    if np.sum(valid) <= 2:
        warnings.warn("Not enough valid intervals to compute Spearman correlation.", UserWarning)
        corr = np.nan
    else:
        corr, _ = spearmanr(f_valid, intervals_mid)

    if to_plot:
        plot_PE_curve(x=intervals_mid, y=f_valid)

    return f_scores, corr, intervals
