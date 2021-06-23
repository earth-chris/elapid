"""Geospatial data operations like reading/writing/indexing raster and vector data."""

from itertools import tee
from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import MultiPoint, Point

from elapid.utils import (
    NoDataException,
    _ncpus,
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


def xy_to_geoseries(x, y, crs="epsg:4236"):
    """Converts x/y data into a geopandas geoseries.

    Args:
        x: 1-D array-like of x location values
        y: 1-D array-like of y location values
        crs: the coordinate reference string. accepts anything allowed by pyproj.CRS.from_user_input()

    Returns:
        gs: Point geometry geoseries
    """
    if not hasattr(x, "__iter__"):
        x = [x]
    if not hasattr(y, "__iter__"):
        y = [y]

    points = [Point(x, y) for x, y in zip(x, y)]
    gs = gpd.GeoSeries(points, crs=crs)

    return gs


def pseudoabsence_from_raster(raster_path, count, ignore_mask=False):
    """Creates a random geographic sampling of points based on a raster's extent.

    Selects from unmasked locations if the rasters nodata value is set.

    Args:
        raster_path: Str raster file path to sample locations from
        count: Int for number of samples to generate
        ignore_mask: Bool to sample from the full extent of the raster
            instead of unmasked areas only

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


def pseudoabsence_from_bias_file(raster_path, count, ignore_mask=False):
    """Creates a semi-random geographic sampling of points weighted towards biased areas.

    Args:
        raster_path: Str raster bias file path to sample from. Pixel values
            can be in arbitrary range, but must be odered low -> high probability
        count: the total number of samples to generate
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


def pseudoabsence_from_vector(vector_path, count, overestimate=2):
    """Creates a random geographic sampling of points inside of a polygon/multipolygon type vector file.

    Args:
        vector_path: Str path to a vector file (shp, geojson, etc)
        count: Int for number of samples to generate
        overestimate: Int for scaler to generate extra samples to
            toss points outside of the polygon/inside it's bounds

    Returns:
        points: Point geometry geoseries
    """
    gdf = gpd.read_file(vector_path)
    return pseudoabsence_from_geoseries(gdf.geometry, count, overestimate=overestimate)


def pseudoabsence_from_geoseries(geoseries, count, overestimate=2):
    """Creates random geographic point samples inside a polygon/multipolygon

    Args:
        geoseries: Geoseries (e.g. `gdf['geometry']`) with polygons/multipolygons
        count: Int for number of samples to generate
        overestimate: Int for scaler to generate extra samples
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


def parse_crs_string(string):
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


def string_to_crs(string):
    """Converts a crs/projection string to a pyproj-readable CRS object

    Args:
        string: a crs/projection string.
        crs_type: the type of crs/projection string, in ["wkt", "proj4", "epsg", "string"].

    Returns:
        crs: a rasterio.crs.CRS object
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


def crs_match(crs1, crs2):
    """Determines whether two coordinate reference systems are the same.

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


def _read_pixel_value(point, source):
    """Reads raster value from an open rasterio dataset.

    Designed to be run using a `geodataframe.apply()` function.

    Args:
        point: a row from gdf.apply() or gdf.iterrows()
        source: an open rasterio data source

    Returns:
        value: 1-d n-length array with the pixel values of each raster band.
    """
    row, col = source.index(point.geometry.x, point.geometry.y)
    window = rio.windows.Window(col, row, 1, 1)
    value = source.read(window=window)
    return np.squeeze(value)


def raster_values_from_vector(vector_path, raster_paths, labels=None, drop_na=True):
    """Reads and stores pixel values from rasters using a point-format vector file.

    Args:
        vector_path: Str path to a vector file (shp, geojson, etc)
        raster_paths: List of raster paths to extract pixel values from
        labels: List of band name labels. should match the total number of bands
            across all raster_paths
        drop_na: Boolean to drop all records with no-data values

    Returns:
        gdf: Geodataframe annotated with the pixel values from each raster
    """
    gdf = gpd.read_file(vector_path)
    raster_df = raster_values_from_geoseries(gdf.geometry, raster_paths, labels, drop_na)
    gdf = pd.concat([gdf, raster_df.drop(["geometry"], axis=1, errors="ignore")], axis=1)
    return gdf


def raster_values_from_geoseries(geoseries, raster_paths, labels=None, drop_na=True, iterate=False):
    """Reads and stores pixel values from rasters using point locations.

    Args:
        geoseries: Geoseries (e.g., gdf['geometry']) with point locations.
        raster_paths: List of raster paths to extract pixel values from.
        labels: List of band name labels. should match the total number of bands across all raster_paths.
        drop_na: Bool to drop all records with no-data values.
        iterate: Bool to loop through each record instead of using .apply(). Slower, but saves memory.

    Returns:
        gdf: Geodataframe annotated with the pixel values from each raster
    """

    # make sure the raster_paths are iterable
    if isinstance(raster_paths, (str)):
        raster_paths = [raster_paths]

    # set raster band column labels
    n_bands = count_raster_bands(raster_paths)
    if labels is None:
        labels = make_band_labels(n_bands)
    else:
        n_labels = len(labels)
        assert n_labels == n_bands, "Number of labels ({n_labels}) doesn't match band count ({n_bands})"

    # annotate each point with the pixel values for each raster
    raster_values = []
    for raster_path in tqdm(raster_paths, desc="Raster"):
        with rio.open(raster_path) as src:

            # reproject points to match raster if necessary
            if not crs_match(geoseries.crs, src.crs):
                points = geoseries.to_crs(src.crs).to_frame("geometry")
            else:
                points = geoseries.to_frame("geometry")

            # read the data one at a time or with apply()
            if iterate:
                row_vals = [
                    _read_pixel_value(point, src)
                    for idx, point in tqdm(points.iterrows(), total=len(points), **tqdm_opts)
                ]
                values = pd.DataFrame(np.vstack(row_vals))
            else:
                values = points.progress_apply(_read_pixel_value, axis=1, result_type="expand", source=src)

            # filter out nodata pixels
            if drop_na and src.nodata is not None:
                values.replace(src.nodata, np.NaN, inplace=True)

        raster_values.append(values)

    # format the values as a geodataframe
    df = pd.concat(raster_values, axis=1)
    df.columns = labels
    df.set_index(geoseries.index, inplace=True)
    gdf = gpd.GeoDataFrame(df, geometry=geoseries, crs=geoseries.crs)

    if drop_na:
        gdf.dropna(axis=0, inplace=True)

    return gdf


def apply_model_to_raster_array(model, array, predictions_window, nodata, nodata_idx, transform=None):
    """Applies a model to an array of covariates.

    Covariate array should be of shape (nbands, nrows, ncols).

    Args:
        model: Object with a `model.predict()` function
        array: Array of shape (nbands, nrows, ncols) with pixel values
        predictions_window: Array to fill with model prediction values
        dims: Tuple of the array dimensions as (nbands, nrows, ncols)
        nodata: Numeric nodata value to apply to the output array
        nodata_idx: Array of bool values with shape (nbands, nrows, ncols) containing nodata locations
        transform: Str method for transforming maxent model output from ["raw", "exponential", "logistic", "cloglog"]

    Returns:
        predictions_window: Array of shape (1, nrows, ncols) with the predictions to write
    """
    # run the computations for only good-data pixels
    good = ~nodata_idx.any(axis=0)
    ngood = good.sum()
    if ngood > 0:
        covariate_array = array[:, good].transpose()
        predictions_array = model.predict(covariate_array, is_features=False, transform=transform)
        predictions_window[:, good] = predictions_array.to_numpy().transpose()

    return predictions_window


def apply_model_to_rasters(
    model,
    raster_paths,
    output_path,
    windowed=True,
    transform="logistic",
    template_idx=0,
    resampling=rio.enums.Resampling.average,
    nodata=-9999,
    driver="GTiff",
    compress="deflate",
    bigtiff=True,
):
    """Applies a trained model to a list of raster datasets.

    The list and band order of the rasters mustm atch the order of the covariates
    used to train the model.It reads each dataset in a block-wise basis, applies
    the model, and writes gridded predictions. If the raster datasets are not
    consistent (different extents, resolutions, etc.), it wll re-project the data
    on the fly, with the grid size, extent and projection based on a 'template'
    raster.

    Args:
        model: Object with a model.predict() function
        raster_paths: List of raster paths of covariates to apply the model to
        output_path: Str path to the output file to create
        windowed: Boolean to perform a block-by-block data read. slower, but reduces memory use.
        transform: Str model transformation type. Select from ["raw", "exponential", "logistic", "cloglog"].
        template_idx: Int index of the raster file to use as a template. template_idx=0 sets the first raster as template
        resampling: Enum resampling algorithm to apply to on-the-fly reprojection (from rasterio.enums.Resampling)
        nodata: Numeric output nodata value to set
        driver: Str output raster file format (from rasterio.drivers.raster_driver_extensions())
        compress: Str of the compression type to apply to the output file
        bigtiff: Boolean of whether to specify the output file as a bigtiff (for rasters > 2GB)

    Returns:
        None: saves model predictions to disk.
    """

    # make sure the raster_paths are iterable
    if isinstance(raster_paths, (str)):
        raster_paths = [raster_paths]

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

    # check whether the raster paths are aligned to determine how the data are read
    aligned = check_raster_alignment(raster_paths)

    # read and reproject blocks from each data source and write predictions to disk
    with rio.open(output_path, "w", **dst_profile) as dst:

        # open all raster paths to read from later
        srcs = [rio.open(raster_path) for raster_path in raster_paths]

        if not aligned:
            vrt_options = {
                "resampling": resampling,
                "transform": dst_profile["transform"],
                "crs": dst_profile["crs"],
                "height": dst_profile["height"],
                "width": dst_profile["width"],
            }
            srcs = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

        # reach each data source block by block and apply the model
        windows, duplicate = tee(windows)
        nwindows = len(list(duplicate))

        for _, window in tqdm(windows, total=nwindows, desc="Tiles"):
            covariates = np.zeros((nbands, window.height, window.width), dtype=np.float32)
            predictions = np.zeros((1, window.height, window.width), dtype=np.float32) + nodata
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
                    predictions,
                    nodata,
                    nodata_idx,
                    transform=transform,
                )
                dst.write(predictions, window=window)

            except NoDataException:
                dst.write(predictions, window=window)
                continue

        for src in srcs:
            src.close()
