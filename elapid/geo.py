"""Geospatial data operations like reading/writing/indexing raster and vector data."""

from multiprocessing import Pool
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from shapely.geometry import MultiPoint, Point
from sklearn.base import BaseEstimator

from elapid.types import CRSType, to_iterable
from elapid.utils import (
    NoDataException,
    check_raster_alignment,
    count_raster_bands,
    create_output_raster_profile,
    get_raster_band_indexes,
    get_tqdm,
    make_band_labels,
    n_digits,
)

tqdm = get_tqdm()
tqdm_opts = {"desc": "Geometry", "leave": False}
tqdm.pandas(**tqdm_opts)


# sampling tools


def xy_to_geoseries(
    x: Union[float, list, np.ndarray], y: Union[float, list, np.ndarray], crs: CRSType = "epsg:4236"
) -> gpd.GeoSeries:
    """Converts x/y data into a geopandas geoseries.

    Args:
        x: 1-D array-like of x location values
        y: 1-D array-like of y location values
        crs: coordinate reference system. accepts pyproj.CRS / rio.crs.CRS objects
            or anything allowed by pyproj.CRS.from_user_input()

    Returns:
        gs: Point geometry geoseries
    """
    # handle single x/y location values
    x = to_iterable(x)
    y = to_iterable(y)

    points = [Point(x, y) for x, y in zip(x, y)]
    gs = gpd.GeoSeries(points, crs=crs)

    return gs


def sample_from_raster(raster_path: str, count: int, ignore_mask: bool = False) -> gpd.GeoSeries:
    """Creates a random geographic sampling of points based on a raster's extent.

    Selects from unmasked locations if the rasters nodata value is set.

    Args:
        raster_path: raster file path to sample locations from
        count: number of samples to generate
        ignore_mask: sample from the full extent of the raster instead of unmasked areas only

    Returns:
        points: Point geometry geoseries
    """
    # handle masked vs unmasked data differently
    with rio.open(raster_path) as src:

        if src.nodata is None or ignore_mask:
            xmin, ymin, xmax, ymax = src.bounds
            xy = np.random.uniform((xmin, ymin), (xmax, ymax), (count, 2))

        else:
            masked = src.read_masks(1)
            rows, cols = np.where(masked == 255)
            samples = np.random.randint(0, len(rows), count)
            xy = np.zeros((count, 2))
            for i, sample in enumerate(samples):
                xy[i] = src.xy(rows[sample], cols[sample])

        points = xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)
        return points


def sample_from_bias_file(raster_path: str, count: int, ignore_mask: bool = False) -> gpd.GeoSeries:
    """Creates a semi-random geographic sampling of points weighted towards biased areas.

    Args:
        raster_path: raster bias file path to sample from. pixel values can
            be in arbitrary range, but must be odered low -> high probability
        count: total number of samples to generate
        ignore_mask: sample from the full extent of the raster instead of unmasked areas only

    Returns:
        points: Point geometry geoseries
    """
    with rio.open(raster_path) as src:

        if src.nodata is None or ignore_mask:
            data = src.read(1)
            rows, cols = np.where(data)
            values = data.flatten()
            probabilities = (values - values.min()) / (values - values.min()).sum()
            samples = np.random.choice(len(rows), size=count, p=probabilities)

        else:
            data = src.read(1, masked=True)
            rows, cols = np.where(~data.mask)
            values = data[rows, cols]
            probabilities = (values - values.min()) / (values - values.min()).sum()
            samples = np.random.choice(len(rows), size=count, p=probabilities)

        xy = np.zeros((count, 2))
        for i, sample in enumerate(samples):
            xy[i] = src.xy(rows[sample], cols[sample])

        points = xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)
        return points


def sample_from_vector(vector_path: str, count: int, overestimate: float = 2) -> gpd.GeoSeries:
    """Creates a random geographic sampling of points inside of a polygon/multipolygon type vector file.

    Args:
        vector_path: path to a vector file (shp, geojson, etc)
        count: number of samples to generate
        overestimate: scaler to generate extra samples to
            toss points outside of the polygon/inside it's bounds

    Returns:
        points: Point geometry geoseries
    """
    gdf = gpd.read_file(vector_path)
    return sample_from_geoseries(gdf.geometry, count, overestimate=overestimate)


def sample_from_geoseries(geoseries: gpd.GeoSeries, count: int, overestimate: float = 2) -> gpd.GeoSeries:
    """Creates random geographic point samples inside a polygon/multipolygon

    Args:
        geoseries: geometry dataset (e.g. `gdf['geometry']`) with polygons/multipolygons
        count: number of samples to generate
        overestimate: scaler to generate extra samples
            to toss points outside of the polygon/inside it's bounds

    Returns:
        points: Point geometry geoseries
    """
    polygon = geoseries.unary_union
    xmin, ymin, xmax, ymax = polygon.bounds
    ratio = polygon.area / polygon.envelope.area

    samples = np.random.uniform((xmin, ymin), (xmax, ymax), (int(count / ratio * overestimate), 2))
    multipoint = MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    samples = np.array(multipoint)

    xy = samples[np.random.choice(len(samples), count)]
    points = xy_to_geoseries(xy[:, 0], xy[:, 1], crs=geoseries.crs)

    return points


# crs management tools


def parse_crs_string(string: str) -> str:
    """Parses a string to determine the CRS/spatial projection format.

    Args:
        string: a string with CRS/projection data.

    Returns:
        crs_type: Str in ["wkt", "proj4", "epsg", "string"].
    """
    if "epsg:" in string.lower():
        return "epsg"
    elif "+proj" in string:
        return "proj4"
    elif "SPHEROID" in string:
        return "wkt"
    else:
        return "string"


def string_to_crs(string: str) -> rio.crs.CRS:
    """Converts a crs/projection string to a pyproj-readable CRS object

    Args:
        string: a crs/projection string.
        crs_type: the type of crs/projection string, in ["wkt", "proj4", "epsg", "string"].

    Returns:
        crs: the coordinate reference system
    """
    crs_type = parse_crs_string(string)

    if crs_type == "epsg":
        auth, code = string.split(":")
        crs = rio.crs.CRS.from_epsg(int(code))
    elif crs_type == "proj4":
        crs = rio.crs.CRS.from_proj4(string)
    elif crs_type == "wkt":
        crs = rio.crs.CRS.from_wkt(string)
    else:
        crs = rio.crs.CRS.from_string(string)

    return crs


def crs_match(crs1: CRSType, crs2: CRSType) -> bool:
    """Evaluates whether two coordinate reference systems are the same.

    Args:
        crs1: the first CRS, from a rasterio dataset, a GeoDataFrame, or a string with projection parameters.
        crs2: the second CRS, from the same sources above.

    Returns:
        matches: Boolean for whether the CRS match.
    """
    # normalize string inputs via rasterio
    if type(crs1) is str:
        crs1 = string_to_crs(crs1)
    if type(crs2) is str:
        crs2 = string_to_crs(crs2)

    matches = crs1 == crs2

    return matches


# raster reading tools


def _read_pixel_value(point: gpd.GeoSeries, source: rio.io.DatasetReader) -> np.ndarray:
    """Reads raster value from an open rasterio dataset.

    Designed to be run using a `geodataframe.apply()` function.

    Args:
        point: a row from gdf.apply() or gdf.iterrows()
        source: an open rasterio data source

    Returns:
        values: 1-d n-length array with the pixel values of each raster band.
    """
    row, col = source.index(point.geometry.x, point.geometry.y)
    window = rio.windows.Window(col, row, 1, 1)
    values = source.read(window=window)
    return np.squeeze(values)


def raster_values_from_vector(
    vector_path: str, raster_paths: list, labels: list = None, drop_na: bool = True
) -> gpd.GeoDataFrame:
    """Reads and stores pixel values from rasters using a point-format vector file.

    Args:
        vector_path: path to a vector file (shp, geojson, etc)
        raster_paths: raster paths to extract pixel values from
        labels: band name labels. should match the total number of bands across all raster_paths
        drop_na: drop all records with no-data values

    Returns:
        gdf: GeoDataFrame annotated with the pixel values from each raster
    """
    gdf = gpd.read_file(vector_path)
    raster_df = raster_values_from_geoseries(gdf.geometry, raster_paths, labels, drop_na)
    gdf = pd.concat([gdf, raster_df.drop(["geometry"], axis=1, errors="ignore")], axis=1)
    return gdf


def raster_values_from_geoseries(
    geoseries: gpd.GeoSeries, raster_paths: list, labels: list = None, drop_na: bool = True, save_memory: bool = False
) -> gpd.GeoDataFrame:
    """Reads and stores pixel values from rasters using point locations.

    Args:
        geoseries: GeoSeries (e.g., gdf['geometry']) with point locations.
        raster_paths: raster paths to extract pixel values from.
        labels: band name labels. should match the total number of bands across all raster_paths.
        drop_na: drop all records with no-data values.
        save_memory: loop through each record instead of using .apply().

    Returns:
        gdf: GeoDataFrame annotated with the pixel values from each raster
    """
    # make sure the raster_paths are iterable
    raster_paths = to_iterable(raster_paths)

    # set raster band column labels
    n_bands = count_raster_bands(raster_paths)
    if labels is None:
        labels = make_band_labels(n_bands)

    n_labels = len(labels)
    assert n_labels == n_bands, "number of band labels ({n_labels}) != n_bands ({n_bands})"

    # annotate each point with the pixel values for each raster
    raster_values = []
    for raster_path in tqdm(raster_paths, desc="Raster"):
        with rio.open(raster_path, "r") as src:

            # reproject points to match raster and convert to a dataframe
            if not crs_match(geoseries.crs, src.crs):
                points = geoseries.to_crs(src.crs, inplace=True)

            points = geoseries.to_frame("geometry")

            # read slowly
            if save_memory:
                row_vals = []
                for idx, point in tqdm(points.iterrows(), total=len(points), **tqdm_opts):
                    row_vals.append(_read_pixel_value(point, src))
                values = pd.DataFrame(np.vstack(row_vals))

            # or read quickly
            else:
                values = points.progress_apply(_read_pixel_value, axis=1, result_type="expand", source=src)

            # filter out nodata pixels
            if drop_na and src.nodata is not None:
                values.replace(src.nodata, np.NaN, inplace=True)

            # explicitly cast the output data type
            raster_values.append(values.astype(src.profile["dtype"]))

    # merge the dataframes from each raster extraction
    df = pd.concat(raster_values, axis=1)
    df.set_index(geoseries.index, inplace=True)
    df.columns = labels

    # convert to a geodataframe
    gdf = gpd.GeoDataFrame(df, geometry=geoseries, crs=geoseries.crs)
    if drop_na:
        gdf.dropna(axis=0, inplace=True)

    return gdf


# raster writing tools


def apply_model_to_raster_array(
    model: BaseEstimator,
    array: np.ndarray,
    nodata: float,
    nodata_idx: int,
    transform: bool = None,
) -> np.ndarray:
    """Applies a model to an array of covariates.

    Covariate array should be of shape (nbands, nrows, ncols).

    Args:
        model: object with a `model.predict()` function
        array: array of shape (nbands, nrows, ncols) with pixel values
        predictions_window: array to fill with model prediction values
        nodata: numeric nodata value to apply to the output array
        nodata_idx: array of bools with shape (nbands, nrows, ncols) containing nodata locations
        transform: method for transforming maxent model output from ["raw", "exponential", "logistic", "cloglog"]

    Returns:
        predictions_window: Array of shape (1, nrows, ncols) with the predictions to write
    """
    # only run the computations for valid pixels
    valid = ~nodata_idx.any(axis=0)
    covariates = array[:, valid].reshape(-1, 1)
    ypred = model.predict(covariates, is_features=False, transform=transform)

    # reshape to the original window size
    ypred_window = np.zeros_like(valid, dtype=np.float32) + nodata
    ypred_window[valid] = ypred.to_numpy().transpose()

    return ypred_window


def apply_model_to_rasters(
    model: BaseEstimator,
    raster_paths: list,
    output_path: str,
    windowed: bool = True,
    transform: str = "logistic",
    template_idx: int = 0,
    resampling: rio.enums.Enum = rio.enums.Resampling.average,
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
) -> None:
    """Applies a trained model to a list of raster datasets.

    The list and band order of the rasters must match the order of the covariates
    used to train the model.It reads each dataset block-by-block, applies
    the model, and writes gridded predictions. If the raster datasets are not
    consistent (different extents, resolutions, etc.), it wll re-project the data
    on the fly, with the grid size, extent and projection based on a 'template'
    raster.

    Args:
        model: object with a model.predict() function
        raster_paths: raster paths of covariates to apply the model to
        output_path: path to the output file to create
        windowed: perform a block-by-block data read. slower, but reduces memory use.
        transform: model transformation type. select from ["raw", "logistic", "cloglog"].
        template_idx: index of the raster file to use as a template. template_idx=0 sets the first raster as template
        resampling: resampling algorithm to apply to on-the-fly reprojection (from rasterio.enums.Resampling)
        nodata: output nodata value to set
        driver: output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: compression type to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)

    Returns:
        None: saves model predictions to disk.
    """
    # make sure the raster_paths are iterable
    raster_paths = to_iterable(raster_paths)

    # get and set template parameters
    windows, dst_profile = create_output_raster_profile(
        raster_paths,
        template_idx,
        windowed=windowed,
        nodata=nodata,
        compress=compress,
        driver=driver,
        bigtiff=bigtiff,
    )

    # get the bands and indexes for each covariate raster
    nbands, band_idx = get_raster_band_indexes(raster_paths)

    # open all rasters to read from later
    srcs = [rio.open(raster_path) for raster_path in raster_paths]

    # check whether the raster paths are aligned to determine how the data are read
    aligned = check_raster_alignment(raster_paths)
    if not aligned:
        vrt_options = {
            "resampling": resampling,
            "transform": dst_profile["transform"],
            "crs": dst_profile["crs"],
            "height": dst_profile["height"],
            "width": dst_profile["width"],
        }
        srcs = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

    # read and reproject blocks from each data source and write predictions to disk
    with rio.open(output_path, "w", **dst_profile) as dst:
        for window in tqdm(windows, desc="Tiles"):
            covariates = np.zeros((nbands, window.height, window.width), dtype=np.float32)
            nodata_idx = np.ones_like(covariates, dtype=bool)

            try:
                for i, src in enumerate(srcs):
                    data = src.read(window=window, masked=True)
                    covariates[band_idx[i] : band_idx[i + 1]] = data
                    nodata_idx[band_idx[i] : band_idx[i + 1]] = data.mask

                    # skip blocks full of no-data
                    if data.mask.all():
                        raise NoDataException()

                predictions = apply_model_to_raster_array(
                    model,
                    covariates,
                    nodata,
                    nodata_idx,
                    transform=transform,
                )
                dst.write(predictions, 1, window=window)

            except NoDataException:
                continue
