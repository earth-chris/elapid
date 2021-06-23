"""Backend helper and convenience functions."""

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
    """Repeats a 1D numpy array along an axis to an arbitrary length

    Args:
        x: the n-dimensional array to repeat
        length: the number of times to repeat the array
        axis: the axis along which to repeat the array (valid values include 0 to n+1)

    Returns:
        An n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)


def load_sample_data(name="bradypus"):
    """Loads example species presence/background and covariate data.

    Args:
        name: the sample dataset to load. options currently include ["bradypus"], from the R 'maxnet' package

    Returns:
        (x, y): a tuple of dataframes containing covariate and response data, respectively
    """
    assert name.lower() in ["bradypus"], "Invalid sample data requested"

    package_path = os.path.realpath(__file__)
    package_dir = os.path.dirname(package_path)

    if name.lower() == "bradypus":

        file_path = os.path.join(package_dir, "data", "bradypus.csv.gz")
        df = pd.read_csv(file_path, compression="gzip").astype("int64")
        y = df["presence"].astype("int8")
        x = df.drop(columns=["presence"]).astype({"ecoreg": "category"})
        return x, y


def save_object(obj, path, compress=True):
    """Writes a python object to disk for later access.

    Args:
        obj: a python object to be saved (e.g., a MaxentModel() instance)
        path: the output file path

    Returns:
        None
    """
    obj = pickle.dumps(obj)

    if compress:
        obj = gzip.compress(obj)

    with open(path, "wb") as f:
        f.write(obj)


def load_object(path, compressed=True):
    """Reads a python object into memory that's been saved to disk.

    Args:
        path: the file path of the object to load
        compressed: flag to specify whether the file was compressed prior to saving

    Returns:
        obj: the python object that has been saved (e.g., a MaxentModel() instance)
    """
    with open(path, "rb") as f:
        obj = f.read()

    if compressed:
        obj = gzip.decompress(obj)

    return pickle.loads(obj)


def create_output_raster_profile(
    raster_paths,
    template_idx=0,
    windowed=True,
    nodata=None,
    compress=None,
    driver="GTiff",
    bigtiff=True,
    dtype="float32",
):
    """Gets parameters for windowed reading/writing to output rasters.

    Args:
        raster_paths: a list of raster paths of covariates to apply the model to
        template_idx: the index of the raster file to use as a template. template_idx=0 sets the first raster as template
        windowed: bool to perform a block-by-block data read. slower, but reduces memory use.
        nodata: the output nodata value to set
        output_driver: the output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: str of the compression type to apply to the output file
        bigtiff: bool of whether to specify the output file as a bigtiff (for rasters > 2GB)

    Returns:
        (windows, profile): an iterable and a dictionary for the window reads and the raster profile
    """
    with rio.open(raster_paths[template_idx]) as src:
        if windowed:
            windows = src.block_windows()
        else:
            idx = (0, 0)
            window = rio.windows.Window(0, 0, src.width, src.height)
            windows = iter([(idx, window)])

        dst_profile = src.profile
        dst_profile.update(
            count=1,
            dtype=dtype,
            nodata=nodata,
            compress=compress,
            driver=driver,
        )
        if bigtiff and driver == "GTiff":
            dst_profile.update(BIGTIFF="YES")

    return windows, dst_profile


def get_raster_band_indexes(raster_paths):
    """Counts the number raster bands to index multi-source, multi-band covariates.

    Args:
        raster_paths: a list of raster paths

    Returns:
        (nbands, band_idx): int and list of the total number of bands and the 0-based start/stop
            band index for each path
    """
    nbands = 0
    band_idx = [0]
    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            nbands += src.count
            band_idx.append(band_idx[i] + src.count)

    return nbands, band_idx


def check_raster_alignment(raster_paths):
    """Checks whether the extent, resolution and projection of multiple rasters match exactly.

    Args:
        raster_paths: a list of raster covariate paths

    Returns:
        Boolean: indicates whether all rasters align
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
    """Tests whether the module is currently running in a jupyter notebook.

    Args:
        None

    Returns:
        Bool
    """
    return "ipykernel" in sys.modules


def get_tqdm():
    """Returns the appropriate tqdm progress tracking module

    Determines the appropriate tqdm based on the user context, as
    behavior changes inside/outside of jupyter notebooks.

    Args:
        None

    Returns:
        tqdm: the context-specific tqdm module
    """
    if in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


class NoDataException(Exception):
    pass


def n_digits(number):
    """Counts the number of significant integer digits of a number.

    Args:
        number: a float or int

    Returns:
        order: integer of the number of digits required to represent a number
    """
    if number == 0:
        order = 1
    else:
        order = np.floor(np.log10(number)).astype(int) + 1

    return order


def count_raster_bands(raster_paths):
    """Returns the total number of bands from a list of rasters.

    Args:
        raster_paths: List of raster data file paths.

    Returns:
        n_bands: Int of the band count.
    """
    n_bands = 0
    for path in raster_paths:
        with rio.open(path) as src:
            n_bands += src.count

    return n_bands


def make_band_labels(n_bands):
    """Creates a list of band names to assign as dataframe columns.

    Args:
        n_bands: Int of the number of raster bands to create labels for.

    Returns:
        labels: List of column labels.
    """
    n_zeros = n_digits(n_bands)
    labels = ["band_{band_number:0{n_zeros}d}".format(band_number=i + 1, n_zeros=n_zeros) for i in range(n_bands)]

    return labels
