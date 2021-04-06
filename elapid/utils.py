"""Backend helper functions that don't need to be exposed to users"""
import gzip
import multiprocessing as mp
import os
import pickle
import sys

import numpy as np
import pandas as pd
import rasterio as rio

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


def load_sample_data(name="bradypus"):
    """
    Loads example species presence/background and covariate data.

    :param name: the sample dataset to load. options currently include ["bradypus"], from the R 'maxnet' package
    :returns: (x, y), a tuple of dataframes containing covariate and response data, respectively
    """
    assert name.lower() in ["bradypus"], "Invalid sample data requested"

    package_path = os.path.realpath(__file__)
    package_dir = os.path.dirname(package_path)

    if name.lower() == "bradypus":

        file_path = os.path.join(package_dir, "data", "bradypus.csv.gz")
        df = pd.read_csv(file_path, compression="gzip").astype("int64")
        y = df["presence"].astype("int8")
        x = df[df.columns[1:]].astype({"ecoreg": "category"})
        return x, y


def save_object(obj, path, compress=True):
    """
    Writes a python object to disk for later access.

    :param obj: a python object to be saved (e.g., a MaxentModel() instance)
    :param path: the output file path
    :returns: none
    """
    obj = pickle.dumps(obj)

    if compress:
        obj = gzip.compress(obj)

    with open(path, "wb") as f:
        f.write(obj)


def load_object(path, compressed=True):
    """
    Reads a python object into memory that's been saved to disk.

    :param path: the file path of the object to load
    :param compressed: flag to specify whether the file was compressed prior to saving
    :returns: obj, the python object that has been saved (e.g., a MaxentModel() instance)
    """
    with open(path, "rb") as f:
        obj = f.read()

    if compressed:
        obj = gzip.decompress(obj)

    return pickle.loads(obj)


def create_output_raster_profile(
    raster_paths, template_idx, nodata=None, compress=None, driver="GTiff", bigtiff=True, dtype="float32"
):
    """
    Gets parameters for windowed reading/writing to output rasters.

    :param raster_paths: a list of raster paths of covariates to apply the model to
    :param template_idx: the index of the raster file to use as a template. template_idx=0 sets the first raster as template
    :param nodata: the output nodata value to set
    :param output_driver: the output raster file format (from rasterio.drivers.raster_driver_extensions())
    :param compress: str of the compression type to apply to the output file
    :param bigtiff: bool of whether to specify the output file as a bigtiff (for rasters > 2GB)
    :returns: windows, profile, an iterable and a dictionary for the window reads and the raster profile
    """
    with rio.open(raster_paths[template_idx]) as src:
        windows = src.block_windows()
        dst_profile = src.profile
        dst_profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress=compress,
            driver=driver,
        )
        if bigtiff:
            dst_profile.update(BIGTIFF="YES")

    return windows, dst_profile


def get_raster_band_indexes(raster_paths):
    """
    Gets the band indexes of multiple raster bands to handle indexing multi-source and multi-band covariates.

    :param raster_paths: a list of raster covariate paths
    :returns: nbands, band_idx, int and list of the total number of bands and the 0-based start/stop band index
    """
    nbands = 0
    band_idx = [0]
    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            nbands += src.count
            band_idx.append(band_idx[i] + src.count)

    return nbands, band_idx


def check_raster_alignment(raster_paths):
    """
    Checks whether the extent, resolution and projection of multiple rasters match exactly.

    :param raster_paths: a list of raster covariate paths
    :returns: bool indicating wither they all align
    """
    first = raster_paths[0]
    rest = raster_paths[1:]

    with rio.open(first) as src:
        res = src.res
        bounds = src.bounds
        transform = src.transform

    for path in rest:
        with rio.open(path) as src:
            if src.res != res or src.bounds != bounds or src.transform != transform:
                return False

    return True


def in_notebook():
    """
    Tests whether the module is currently running in a jupyter notebook.

    :returns: bool
    """
    return "ipykernel" in sys.modules


def get_tqdm():
    """
    Returns the appropriate tqdm progress tracking module based on the user context, as
      behavior changes inside/outside of jupyter notebooks.

    :returns: the tqdm module
    """
    if in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


def apply_model_to_raster_array(model, array, nodata, nodata_idx, transform=None):
    """
    Applies a maxent model to a (nbands, nrows, ncols) array of extracted pixel values.

    :param model: the trained model with a model.predict() function
    :param array: array of shape (nbands, nrows, ncols) with pixel values
    :param dims: a tuple of the array dimensions as (nbands, nrows, ncols)
    :param nodata: the nodata value to apply to the output array
    :param nodata_idx: array of bool values of shape (nbands, nrows, ncols) with nodata locations
    :param transform: the method for transforming maxent model output from ["raw", "exponential", "logistic", "cloglog"]
    :returns: predictions_window, an array of shape (nbands, nrows, ncols) with the predictions to write
    """
    # we'll run the computations for only good-data pixels
    nbands, nrows, ncols = array.shape
    good = ~nodata_idx.all(axis=0)
    ngood = good.sum()
    predictions_window = np.zeros((1, nrows, ncols), dtype=np.float32) + nodata
    if ngood > 0:
        covariate_array = array[:, good].transpose()
        predictions_array = model.predict(covariate_array, is_features=False, transform=transform)
        predictions_window[:, good] = predictions_array.to_numpy(dtype=np.float32).transpose()

    return predictions_window
