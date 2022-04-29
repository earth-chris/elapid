"""Backend helper and convenience functions."""

import gzip
import multiprocessing as mp
import os
import pickle
import sys
from typing import Any, Callable, Dict, Iterable, Tuple, Union

import numpy as np
import pandas as pd
import rasterio as rio

from elapid.types import Number

n_cpus = mp.cpu_count()


class NoDataException(Exception):
    pass


def repeat_array(x: np.array, length: int = 1, axis: int = 0) -> np.ndarray:
    """Repeats a 1D numpy array along an axis to an arbitrary length

    Args:
        x: the n-dimensional array to repeat
        length: the number of times to repeat the array
        axis: the axis along which to repeat the array (valid values include 0 to n+1)

    Returns:
        An n+1 dimensional numpy array
    """
    return np.expand_dims(x, axis=axis).repeat(length, axis=axis)


def load_sample_data(name: str = "bradypus") -> Tuple[pd.DataFrame, pd.DataFrame]:
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


def save_object(obj: object, path: str, compress: bool = True) -> None:
    """Writes a python object to disk for later access.

    Args:
        obj: a python object or variable to be saved (e.g., a MaxentModel() instance)
        path: the output file path
    """
    obj = pickle.dumps(obj)

    if compress:
        obj = gzip.compress(obj)

    with open(path, "wb") as f:
        f.write(obj)


def load_object(path: str, compressed: bool = True) -> Any:
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
    raster_paths: list,
    template_idx: int = 0,
    windowed: bool = True,
    nodata: Number = None,
    count: int = 1,
    compress: str = None,
    driver: str = "GTiff",
    bigtiff: bool = True,
    dtype: str = "float32",
) -> Tuple[Iterable, Dict]:
    """Gets parameters for windowed reading/writing to output rasters.

    Args:
        raster_paths: raster paths of covariates to apply the model to
        template_idx: index of the raster file to use as a template. template_idx=0 sets the first raster as template
        windowed: perform a block-by-block data read. slower, but reduces memory use
        nodata: output nodata value
        count: number of bands in the prediction output
        output_driver: output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: compression type to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        dtype: rasterio data type string

    Returns:
        (windows, profile): an iterable and a dictionary for the window reads and the raster profile
    """
    with rio.open(raster_paths[template_idx]) as src:
        if windowed:
            windows = [window for _, window in src.block_windows()]
        else:
            windows = [rio.windows.Window(0, 0, src.width, src.height)]

        dst_profile = src.profile.copy()
        dst_profile.update(
            count=count,
            dtype=dtype,
            nodata=nodata,
            compress=compress,
            driver=driver,
        )
        if bigtiff and driver == "GTiff":
            dst_profile.update(BIGTIFF="YES")

    return windows, dst_profile


def get_raster_band_indexes(raster_paths: list) -> Tuple[int, list]:
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


def check_raster_alignment(raster_paths: list) -> bool:
    """Checks whether the extent, resolution and projection of multiple rasters match exactly.

    Args:
        raster_paths: a list of raster covariate paths

    Returns:
        whether all rasters align
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


def in_notebook() -> bool:
    """Evaluate whether the module is currently running in a jupyter notebook."""
    return "ipykernel" in sys.modules


def get_tqdm() -> Callable:
    """Returns a context-appropriate tqdm progress tracking function.

    Determines the appropriate tqdm based on the user context, as
        behavior changes inside/outside of jupyter notebooks.

    Returns:
        tqdm: the context-specific tqdm module
    """
    if in_notebook():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm

    return tqdm


def n_digits(number: Number) -> int:
    """Counts the number of significant integer digits of a number.

    Args:
        number: the number to evaluate.

    Returns:
        order: number of digits required to represent a number
    """
    if number == 0:
        order = 1
    else:
        order = np.floor(np.log10(number)).astype(int) + 1

    return order


def count_raster_bands(raster_paths: list) -> int:
    """Returns the total number of bands from a list of rasters.

    Args:
        raster_paths: List of raster data file paths.

    Returns:
        n_bands: total band count.
    """
    n_bands = 0
    for path in raster_paths:
        with rio.open(path) as src:
            n_bands += src.count

    return n_bands


def make_band_labels(n_bands: int) -> list:
    """Creates a list of band names to assign as dataframe columns.

    Args:
        n_bands: total number of raster bands to create labels for.

    Returns:
        labels: list of column names.
    """
    n_zeros = n_digits(n_bands)
    labels = ["b{band_number:0{n_zeros}d}".format(band_number=i + 1, n_zeros=n_zeros) for i in range(n_bands)]

    return labels


def format_band_labels(raster_paths: list, labels: list = None):
    """Verifies whether a list of band labels matches the band count,
        or creates labels when none are passed.

    Args:
        raster_paths:
    """
    n_bands = count_raster_bands(raster_paths)

    if labels is None:
        labels = make_band_labels(n_bands)

    n_labels = len(labels)
    assert n_labels == n_bands, f"number of band labels ({n_labels}) != n_bands ({n_bands})"

    return labels
