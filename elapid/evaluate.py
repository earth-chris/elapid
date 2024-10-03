import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


# implement Boyce index as describe in https://www.whoi.edu/cms/files/hirzel_etal_2006_53457.pdf (Eq.4)


def boycei(interval, obs, fit):
    """
    Calculate the Boyce index for a given interval.
    
    Args:
        interval (tuple or list): Two elements representing the lower and upper bounds of the interval.
        obs (numpy.ndarray): Observed suitability values (i.e., predictions at presence points).
        fit (numpy.ndarray): Suitability values (e.g., from a raster), i.e., predictions at presence + background points.
    
    Returns:
        float: The ratio of observed to expected frequencies, representing the Boyce index for the given interval.
    """
    # Boolean arrays for classification
    fit_bin = (fit >= interval[0]) & (fit <= interval[1])
    obs_bin = (obs >= interval[0]) & (obs <= interval[1])

    # Compute pi and ei
    pi = np.sum(obs_bin) / len(obs_bin)
    ei = np.sum(fit_bin) / len(fit_bin)

    if ei == 0:
        fi = np.nan  # Avoid division by zero
    else:
        fi = pi / ei

    return fi


def boyce_index(fit, obs, nclass=0, window="default", res=100, PEplot=False):
    """
    Compute the Boyce index to evaluate habitat suitability models.
    
    The Boyce index evaluates how well a model predicts species presence by comparing its predictions
    to a random distribution of observed presences along the prediction gradients. It is specifically 
    designed for presence-only models and serves as an appropriate metric in such cases.
    
    It divides the probability of species presence into ranges and, for each range, calculates the predicted-to-expected ratio (F ratio).
    The final output is given by the Spearman correlation between the mid-point of the probability interval and the F ratio.
    
    Index ranges from -1 to +1:
    - Positive values: Model predictions align with actual species presence distribution.
    - Values near zero: Model performs similarly to random predictions.
    - Negative values: Model incorrectly predicts low-quality areas where species are more frequently found.
    
    This calculation is based on the continuous Boyce index (Eq. 4) as defined in Hirzel et al. 2006.

    Args:
        fit (numpy.ndarray | pd.Series | gpd.GeoSeries): Suitability values (e.g., predictions at presence + background points).
        obs (numpy.ndarray | pd.Series | gpd.GeoSeries): Observed suitability values, i.e., predictions at presence points.
        nclass (int | list, optional): Number of classes or list of class thresholds. Defaults to 0.
        window (float | str, optional): Width of the moving window. Defaults to 'default' which sets window as 1/10th of the fit range.
        res (int, optional): Resolution, i.e., number of steps if nclass=0. Defaults to 100.
        PEplot (bool, optional): Whether to plot the predicted-to-expected (P/E) curve. Defaults to False.
    
    Returns:
        dict: A dictionary with the following keys:
            - 'F.ratio' (numpy.ndarray): The P/E ratio for each bin.
            - 'Spearman.cor' (float): The Spearman's rank correlation coefficient between interval midpoints and F ratios.
            - 'HS' (numpy.ndarray): The habitat suitability intervals.
    
    Example:
        # Predicted suitability scores (e.g., predictions at presence + background points)
        predicted = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        # Observed presence suitability scores (e.g., predictions at presence points)
        observed = np.array([0.3, 0.7, 0.8, 0.9])

        # Call the boyce_index function to calculate the Boyce index and Spearman correlation
        results = boyce_index(fit=predicted, obs=observed, nclass=3)
        print(results)

        # Output:
        # {'F.ratio': array([0.625, 0.625, 1.875]),
        #  'Spearman.cor': 0.866,
        #  'HS': array([[0.1 , 0.4 ],
        #               [0.4 , 0.7 ],
        #               [0.7 , 1.  ]])}
    """
    
    
    # Check types of fit and obs
    acceptable_types = (np.ndarray, pd.Series, gpd.GeoSeries)
    if not isinstance(fit, acceptable_types):
        raise TypeError("The 'fit' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries.")
    if not isinstance(obs, acceptable_types):
        raise TypeError("The 'obs' parameter must be a NumPy array, Pandas Series, or GeoPandas GeoSeries.")

    
    # Convert inputs to NumPy arrays
    fit = np.asarray(fit)
    obs = np.asarray(obs)
    
    
    # Ensure fit and obs are one-dimensional arrays
    if fit.ndim != 1 or obs.ndim != 1:
        raise ValueError("Both 'fit' and 'obs' must be one-dimensional arrays.")


    # Remove NaNs from fit and obs
    fit = fit[~np.isnan(fit)]
    obs = obs[~np.isnan(obs)]

    if len(fit) == 0 or len(obs) == 0:
        raise ValueError("After removing NaNs, 'fit' or 'obs' arrays cannot be empty.")

    
    # Remove NaNs from fit
    fit = fit[~np.isnan(fit)]

    if window == "default":
        window = (np.max(fit) - np.min(fit)) / 10.0

    mini = np.min(fit)
    maxi = np.max(fit)

    if nclass == 0:
        vec_mov = np.linspace(mini, maxi - window, num=res+1)
        intervals = np.column_stack((vec_mov, vec_mov + window))
    elif isinstance(nclass, (list, np.ndarray)) and len(nclass) > 1:
        nclass.sort()
        if mini > nclass[0] or  maxi < nclass[-1]:
            raise ValueError(f"The range provided via nclass is: ({nclass[0], nclass[-1]}). The range computed via fit is: ({mini, maxi}). Provided range via nclass should be in range computed via (max(fit), min(fit)).")
        vec_mov = np.concatenate(([mini], nclass))
        intervals = np.column_stack((vec_mov[:-1], vec_mov[1:]))
        print(vec_mov)
        print(intervals)
    elif nclass > 0:
        vec_mov = np.linspace(mini, maxi, num=nclass + 1)
        intervals = np.column_stack((vec_mov[:-1], vec_mov[1:]))
    else:
        raise ValueError("Invalid nclass value.")


    # Apply boycei function to each interval
    f_list = []
    for inter in intervals:
        fi = boycei(inter, obs, fit)
        f_list.append(fi)
    f = np.array(f_list)


    # Remove NaNs
    valid = ~np.isnan(f)
    
    # use interval midpoints to calculate the spearmanr coeff. 
    intervals_mid = np.mean(intervals[valid], axis=1)
    if np.sum(valid) <= 2:
        corr = np.nan
    else:
        f_valid = f[valid]
        corr, _ = spearmanr(f_valid, intervals_mid)


    if PEplot:
        plt.figure()
        plt.plot(intervals_mid, f[valid], marker='o')
        plt.xlabel('Habitat suitability')
        plt.ylabel('Predicted/Expected ratio')
        plt.title('Boyce Index')
        plt.show()


    results = {
        'F.ratio': f,
        'Spearman.cor': round(corr, 3) if not np.isnan(corr) else np.nan,
        'HS': intervals,
    }

    return results

