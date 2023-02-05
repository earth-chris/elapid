"""Unit tests for the elapid/utils.py module"""

import os
import shutil
import tempfile
from copy import copy
from glob import glob

import numpy as np
import rasterio as rio

from elapid import utils

# set the test raster data paths
directory_path, script_path = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(directory_path, "data")
raster_1b = os.path.join(data_path, "test-raster-1band.tif")
raster_2b = os.path.join(data_path, "test-raster-2bands.tif")
raster_1b_offset = os.path.join(data_path, "test-raster-1band-offset.tif")
with rio.open(raster_1b, "r") as src:
    raster_1b_profile = copy(src.profile)


def test_repeat_array():
    n_elements = 10
    x = np.zeros(n_elements)

    # repeating row-wise
    r1 = utils.repeat_array(x, length=1, axis=0)
    r2 = utils.repeat_array(x, length=2, axis=0)
    assert r1.shape == (1, n_elements)
    assert r2.shape == (2, n_elements)

    # repeating column-wise
    c1 = utils.repeat_array(x, length=1, axis=1)
    c2 = utils.repeat_array(x, length=2, axis=1)
    assert c1.shape == (n_elements, 1)
    assert c2.shape == (n_elements, 2)


def test_load_sample_data():
    x, y = utils.load_sample_data(name="bradypus")
    assert x.shape == (1116, 14), "Dataframe not verified dimensions"
    assert len(x) == len(y), "Nuber of x/y rows must match"
    assert y.min() == 0, "y data should only contain 0/1"
    assert y.max() == 1, "y data should only contain 0/1"

    # data verified from first row
    first_record = np.array([76, 104, 10, 2, 121, 46, 84, 41, 54, 3, 192, 266, 337, 279])
    diff = x.iloc[0].to_numpy() - first_record
    assert diff.sum() == 0, "First row of bradypus data incorrectly read"


def test_download_sample_data():
    outdir = "tmp-download"
    utils.download_sample_data(outdir)
    assert os.path.exists(outdir)
    fnames = glob(os.path.join(outdir, "*"))
    for fname in fnames:
        assert os.path.getsize(fname) > 0
    shutil.rmtree(outdir)


def test_save_object():
    obj = np.zeros(10)
    try:
        with tempfile.NamedTemporaryFile() as tf:
            temp_name = tf.name
            utils.save_object(obj, temp_name, compress=False)
            uncompressed_size = os.path.getsize(temp_name)
            assert uncompressed_size > 0, "Saved file should be greater than zero bytes"
            utils.save_object(obj, temp_name, compress=True)
            compressed_size = os.path.getsize(temp_name)
            assert compressed_size < uncompressed_size, "Compressed size should be smaller than uncompressed"
    except PermissionError:
        pass


def test_load_object():
    n_elements = 10
    obj = np.zeros(n_elements)
    obj[-1] = n_elements
    compress = False
    try:
        with tempfile.NamedTemporaryFile() as tf:
            temp_name = tf.name
            utils.save_object(obj, temp_name, compress=compress)
            loaded_obj = utils.load_object(temp_name, compressed=compress)
        assert len(loaded_obj) == n_elements, "Loaded object doesn't match shape of saved object"
        assert loaded_obj[-1] == n_elements, "Loaded object doesn't match data content of saved object"
    except PermissionError:
        pass


def test_create_output_raster_profile():
    raster_paths = [raster_1b, raster_2b]
    nodata = -9999
    windows, output_profile = utils.create_output_raster_profile(raster_paths, template_idx=0, nodata=nodata)

    # window check
    nwindows = len(list(windows))
    assert nwindows == 1

    # profile check
    assert raster_1b_profile["width"] == output_profile["width"]
    assert raster_1b_profile["nodata"] != output_profile["nodata"]


def test_get_raster_band_indexes():
    raster_paths = [raster_1b, raster_2b]
    nbands, index = utils.get_raster_band_indexes(raster_paths)
    assert nbands == 3
    assert index == [0, 1, 3]


def test_check_raster_alignment():
    # fail on misaligned
    raster_paths = [raster_1b, raster_1b_offset]
    aligned = utils.check_raster_alignment(raster_paths)
    assert aligned is False

    # succeed on aligned
    raster_paths = [raster_1b, raster_2b]
    aligned = utils.check_raster_alignment(raster_paths)
    assert aligned is True


def test_in_notebook():
    assert utils.in_notebook() is False


def test_get_tqdm():
    tqdm = utils.get_tqdm()
    methods = dir(tqdm)

    assert "monitor_interval" in methods
    assert "tqdm_notebook" not in methods, "Returned tqdm should not be the base module"


def test_n_digits():
    assert utils.n_digits(1) == 1
    assert utils.n_digits(11) == 2
    assert utils.n_digits(111) == 3


def test_count_raster_bands():
    list_2b = [raster_1b, raster_1b]
    list_3b = [raster_1b, raster_2b]

    assert utils.count_raster_bands(list_2b) == 2
    assert utils.count_raster_bands(list_3b) == 3


def test_make_band_labels():
    n_bands = 1
    labels = utils.make_band_labels(n_bands)
    assert len(labels) == n_bands

    n_bands = 10
    labels = utils.make_band_labels(n_bands)
    assert len(labels) == n_bands
