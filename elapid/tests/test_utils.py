"""Unit test suite for the elapid/utils.py module"""

import os
import tempfile

import numpy as np

from elapid import utils


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
    first_record = np.array([76,104,10,2,121,46,84,41,54,3,192,266,337,279])
    diff = x.iloc[0].to_numpy() - first_record
    assert diff.sum() == 0, "First row of bradypus data incorrectly read"


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

    with tempfile.NamedTemporaryFile() as tf:
        temp_name = tf.name
        utils.save_object(obj, temp_name, compress=compress)
        loaded_obj = utils.load_object(temp_name, compressed=compress)

    assert len(loaded_obj) == n_elements, "Loaded object doesn't match shape of saved object"
    assert loaded_obj[-1] == n_elements, "Loaded object doesn't match data content of saved object"


# TODO
def test_create_output_raster_profile():
    pass


# TODO
def test_get_raster_band_indexes():
    pass


# TODO
def test_check_raster_alignment():
    pass


def test_in_notebook():
    assert utils.in_notebook() == False


def test_get_tqdm():
    tqdm = utils.get_tqdm()
    methods = dir(tqdm)

    assert "monitor_interval" in methods
    assert "tqdm_notebook" not in methods, "Returned tqdm should not be the module"
