import math
import warnings
from typing import Dict, List, Tuple, Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# implement continuous Boyce index as describe in https://www.whoi.edu/cms/files/hirzel_etal_2006_53457.pdf (Eq.4)


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
    nbins: Union[int, List[float], np.ndarray] = 0,
    bin_size: Union[float, str] = "default",
    range: Tuple[float, float] = (0, 1),
) -> np.ndarray:
    """
    Generates habitat suitability intervals for the Boyce Index calculation.

    Calculates intervals based on the provided range and either the number of bins or bin size.

    Args:
        nbins (int or list or np.ndarray, optional): Number of classes or a list of class thresholds. Defaults to 0.
        bin_size (float or str, optional): Width of the bins. Defaults to 'default', which sets the width as 1/10th of the range.
        range (tuple or list): Two elements representing the minimum and maximum values of habitat suitability. Default : [0, 1]

    Returns:
        np.ndarray: An array of intervals, each represented by a pair of lower and upper bounds.

    Raises:
        ValueError: If invalid values are provided for nbins or bin_size.
    """
    mini, maxi = range

    if isinstance(bin_size, float):
        nbins = (maxi - mini) / bin_size
        if not nbins.is_integer():
            warnings.warn(
                f"bin_size has been adjusted to nearest appropriate size using ceil, as range/bin_size : {(maxi - mini)} / {bin_size} is not an integer.",
                UserWarning,
            )
        nbins = math.ceil(nbins)
    elif nbins == 0 and bin_size == "default":
        nbins = 10

    if isinstance(nbins, (list, np.ndarray)):
        if len(nbins) == 1:
            raise ValueError("Invalid nbins value. len(nbins) must be > 1")
        nbins.sort()
        intervals = np.column_stack((nbins[:-1], nbins[1:]))
    elif nbins > 1:
        boundary = np.linspace(mini, maxi, num=nbins + 1)
        intervals = np.column_stack((boundary[:-1], boundary[1:]))
    else:
        raise ValueError("Invalid nbins value. nbins > 1")

    return intervals


def boyce_index(yobs: np.ndarray, ypred: np.ndarray, interval: Union[Tuple[float, float], List[float]]) -> float:
    """
    Calculates the Boyce index for a given interval.

    Uses the convention as defined in Hirzel et al. 2006 to compute the ratio of observed to expected frequencies.

    Args:
        yobs (np.ndarray): Suitability values at observed locations (e.g., predictions at presence points).
        ypred (np.ndarray): Suitability values at random locations (e.g., predictions at background points).
        interval (tuple or list): Two elements representing the lower and upper bounds of the interval (i.e., habitat suitability).

    Returns:
        float: The ratio of observed to expected frequencies for the given interval.

    """
    yobs_bin = (yobs >= interval[0]) & (yobs <= interval[1])
    ypred_bin = (ypred >= interval[0]) & (ypred <= interval[1])

    pi = np.sum(yobs_bin) / len(yobs_bin)
    ei = np.sum(ypred_bin) / len(ypred_bin)

    if ei == 0:
        fi = np.nan  # Avoid division by zero
    else:
        fi = pi / ei

    return fi


def continuous_boyce_index(
    yobs: Union[np.ndarray, pd.Series, gpd.GeoSeries],
    ypred: Union[np.ndarray, pd.Series, gpd.GeoSeries],
    nbins: Union[int, List[float], np.ndarray] = 0,
    bin_size: Union[float, str] = "default",
    to_plot: bool = False,
) -> Dict[str, Union[np.ndarray, float]]:
    """
    Compute the continuous Boyce index to evaluate habitat suitability models.

    Uses the convention as defined in Hirzel et al. 2006 to compute the ratio of observed to expected frequencies.

    Args:
        yobs (numpy.ndarray | pd.Series | gpd.GeoSeries): Suitability values at observed location (i.e., predictions at presence points).
        ypred (numpy.ndarray | pd.Series | gpd.GeoSeries): Suitability values at random location (i.e., predictions at background points).
        nbins (int | list, optional): Number of classes or a list of class thresholds. Defaults to 0.
        bin_size (float | str, optional): Width of the the bin. Defaults to 'default' which sets width as 1/10th of the fit range.
        to_plot (bool, optional): Whether to plot the predicted-to-expected (P/E) curve. Defaults to False.

    Returns:
        dict: A dictionary with the following keys:
            - 'F.ratio' (numpy.ndarray): The P/E ratio for each bin.
            - 'Spearman.cor' (float): The Spearman's rank correlation coefficient between interval midpoints and F ratios.
            - 'HS' (numpy.ndarray): The habitat suitability intervals.
    """
    if not isinstance(ypred, (np.ndarray, pd.Series, gpd.GeoSeries)):
        raise TypeError("The 'ypred' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries.")
    if not isinstance(yobs, (np.ndarray, pd.Series, gpd.GeoSeries)):
        raise TypeError("The 'yobs' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries.")
    if not isinstance(nbins, (int, list, np.ndarray)):
        raise TypeError("The 'nbins' parameter must be a int, list, or 1-d NumPy array.")
    if not isinstance(bin_size, (float, str)):
        raise TypeError("The 'bin_size' parameter must be a float, or str ('default').")

    if isinstance(bin_size, float) and (isinstance(nbins, (list, np.ndarray)) or nbins > 0):
        raise ValueError(
            f"Ambiguous value provided. Provide either nbins or bin_size. Cannot provide both. Provided values for nbins, bin_size are: ({nbins, bin_size})"
        )

    # Check for NaN values and issue warnings
    if np.isnan(ypred).any():
        warnings.warn("'ypred' contains NaN values, which will be ignored.", UserWarning)
        ypred = ypred[~np.isnan(ypred)]
    if np.isnan(yobs).any():
        warnings.warn("'yobs' contains NaN values, which will be ignored.", UserWarning)
        yobs = yobs[~np.isnan(yobs)]

    ypred = np.asarray(ypred)
    yobs = np.asarray(yobs)

    if ypred.ndim != 1 or yobs.ndim != 1:
        raise ValueError("Both 'ypred' and 'yobs' must be one-dimensional arrays.")

    if len(ypred) == 0 or len(yobs) == 0:
        raise ValueError("'ypred' or 'yobs' arrays cannot be empty.")

    mini, maxi = np.min(ypred), np.max(ypred)

    intervals = get_intervals(nbins, bin_size, range=[mini, maxi])
    f_scores = np.array([boyce_index(yobs, ypred, interval) for interval in intervals])

    valid = ~np.isnan(f_scores)
    f_valid = f_scores[valid]

    intervals_mid = np.mean(intervals[valid], axis=1)
    if np.sum(valid) <= 2:
        corr = np.nan
    else:
        corr, _ = spearmanr(f_valid, intervals_mid)

    if to_plot:
        plot_PE_curve(x=intervals_mid, y=f_valid)

    results = {
        "F.ratio": f_scores,
        "Spearman.cor": corr,
        "HS": intervals,
    }

    return results
