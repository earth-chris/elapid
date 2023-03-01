"""Geospatial data operations like reading/writing/indexing raster and vector data."""

import os
import warnings
from typing import Union

import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
import rasterio as rio
from rasterio.features import geometry_mask
from scipy.spatial import KDTree
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon
from sklearn.base import BaseEstimator

from elapid.stats import get_raster_stats_methods, normalize_sample_probabilities
from elapid.types import CRSType, Vector, to_iterable
from elapid.utils import (
    NoDataException,
    check_raster_alignment,
    create_output_raster_profile,
    format_band_labels,
    get_raster_band_indexes,
    get_tqdm,
    n_digits,
    tqdm_opts,
)

tqdm = get_tqdm()

# sampling tools


def xy_to_geoseries(
    x: Union[float, list, np.ndarray], y: Union[float, list, np.ndarray], crs: CRSType = "epsg:4326"
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


def stack_geodataframes(
    presence: Vector, background: Vector, add_class_label: bool = False, target_crs: str = "presence"
) -> gpd.GeoDataFrame:
    """Concatenate geometries from two GeoSeries/GeoDataFrames.

    Args:
        presence: presence geometry (y=1) locations
        background: background geometry (y=0) locations
        add_class_label: add a column labeling the y value for each point
        target_crs: if reprojection is necessary, use this variable's crs.
            valid options are "presence" and "background"

    Returns:
        merged GeoDataFrame with all geometries projected to the same crs.
    """
    validate_gpd(presence)
    validate_gpd(background)

    # cast to geodataframes
    if isinstance(presence, gpd.GeoSeries):
        presence = presence.to_frame("geometry")
    if isinstance(background, gpd.GeoSeries):
        background = background.to_frame("geometry")

    # handle projection mismatch
    crs = presence.crs
    if crs_match(presence.crs, background.crs):
        # explicitly set the two to exactly matching crs as geopandas
        # throws errors if there's any mismatch at all
        background.crs = presence.crs
    else:
        if target_crs.lower() == "presence":
            background.to_crs(crs, inplace=True)
        elif target_crs.lower() == "background":
            crs = background.crs
            presence.to_crs(crs, inplace=True)
        else:
            raise NameError(f"Unrecognized target_crs option: {target_crs}")

    if add_class_label:
        presence["class"] = 1
        background["class"] = 0

    matching = [col for col in presence.columns if col in background.columns]
    assert len(matching) > 0, "no matching columns found between data frames"

    merged = pd.concat((presence[matching], background[matching]), axis=0, ignore_index=True)
    gdf = gpd.GeoDataFrame(merged, crs=crs)

    return gdf


def sample_raster(raster_path: str, count: int, nodata: float = None, ignore_mask: bool = False) -> gpd.GeoSeries:
    """Create a random geographic sample of points based on a raster's extent.

    Selects from unmasked locations if the rasters nodata value is set.

    Args:
        raster_path: raster file path to sample locations from
        count: number of samples to generate
        nodata: add pixels with this value to the nodata mask
        ignore_mask: sample from the full extent of the raster instead of unmasked areas only

    Returns:
        points: Point geometry geoseries
    """
    # handle masked vs unmasked data differently
    with rio.open(raster_path) as src:
        if src.nodata is None or ignore_mask:
            if nodata is None:
                xmin, ymin, xmax, ymax = src.bounds
                xy = np.random.uniform((xmin, ymin), (xmax, ymax), (count, 2))
            else:
                data = src.read(1)
                mask = data != nodata
                rows, cols = np.where(mask)
                samples = np.random.randint(0, len(rows), count)
                xy = np.zeros((count, 2))
                for i, sample in enumerate(samples):
                    xy[i] = src.xy(rows[sample], cols[sample])

        else:
            if nodata is None:
                masked = src.read_masks(1)
                rows, cols = np.where(masked == 255)
            else:
                data = src.read(1, masked=True)
                data.mask += data.data == nodata
                rows, cols = np.where(~data.mask)

            samples = np.random.randint(0, len(rows), count)
            xy = np.zeros((count, 2))
            for i, sample in enumerate(samples):
                xy[i] = src.xy(rows[sample], cols[sample])

        points = xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)
        return points


def sample_bias_file(raster_path: str, count: int, ignore_mask: bool = False) -> gpd.GeoSeries:
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
            probabilities = normalize_sample_probabilities(values)
            samples = np.random.choice(len(rows), size=count, p=probabilities)

        else:
            data = src.read(1, masked=True)
            rows, cols = np.where(~data.mask)
            values = data[rows, cols]
            probabilities = normalize_sample_probabilities(values)
            samples = np.random.choice(len(rows), size=count, p=probabilities)

        xy = np.zeros((count, 2))
        for i, sample in enumerate(samples):
            xy[i] = src.xy(rows[sample], cols[sample])

        points = xy_to_geoseries(xy[:, 0], xy[:, 1], crs=src.crs)
        return points


def sample_vector(vector_path: str, count: int, overestimate: float = 2) -> gpd.GeoSeries:
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
    return sample_geoseries(gdf.geometry, count, overestimate=overestimate)


def sample_geoseries(geoseries: gpd.GeoSeries, count: int, overestimate: float = 2) -> gpd.GeoSeries:
    """Creates random geographic point samples inside a polygon/multipolygon

    Args:
        geoseries: geometry dataset (e.g. `gdf['geometry']`) with polygons/multipolygons
        count: number of samples to generate
        overestimate: scaler to generate extra samples
            to toss points outside of the polygon/inside it's bounds

    Returns:
        points: Point geometry geoseries
    """
    if type(geoseries) is gpd.GeoDataFrame:
        geoseries = geoseries.geometry

    polygon = geoseries.unary_union
    xmin, ymin, xmax, ymax = polygon.bounds
    ratio = polygon.area / polygon.envelope.area

    samples = np.random.uniform((xmin, ymin), (xmax, ymax), (int(count / ratio * overestimate), 2))
    multipoint = MultiPoint(samples)
    multipoint = multipoint.intersection(polygon)
    sample_array = np.zeros((len(multipoint.geoms), 2))
    for idx, point in enumerate(multipoint.geoms):
        sample_array[idx] = (point.x, point.y)

    xy = sample_array[np.random.choice(len(sample_array), count)]
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


def annotate(
    points: Union[str, gpd.GeoSeries, gpd.GeoDataFrame],
    raster_paths: Union[str, list],
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Read raster values for each point in a vector and append as new columns.

    Args:
        points: path to a point-format vector, OR
            GeoDataFrame with point locations, OR
            GeoSeries (e.g., gdf['geometry']) with point locations
        raster_paths: raster paths to extract pixel values from.
        labels: band name labels. number of labels should match the
            total number of bands across all raster_paths.
        drop_na: drop all records with no-data values.
        quiet: silence progress bar output.

    Returns:
        GeoDataFrame annotated with the pixel values from each raster
    """
    # format the inputs
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    # read raster values based on the points dtype
    if isinstance(points, gpd.GeoSeries):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )

    elif isinstance(points, gpd.GeoDataFrame) or isinstance(points, pd.DataFrame):
        points = points.reset_index(drop=True)
        gdf = annotate_geoseries(
            points.geometry,
            raster_paths,
            labels=labels,
            drop_na=drop_na,
            quiet=quiet,
        )

        # append annotations to the input dataframe
        gdf = pd.concat([points, gdf.drop(["geometry"], axis=1, errors="ignore")], axis=1)

    elif os.path.isfile(points):
        gdf = annotate_vector(points, raster_paths, labels=labels, drop_na=drop_na, quiet=quiet)

    else:
        raise TypeError("points arg must be a valid path, GeoDataFrame, or GeoSeries")

    if drop_na:
        try:
            valid = gdf["valid"] == 1
            gdf = gdf[valid].drop(columns="valid").dropna().reset_index(drop=True)
        except KeyError:
            pass

    return gdf


def annotate_vector(
    vector_path: str,
    raster_paths: list,
    labels: list = None,
    drop_na: bool = True,
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Reads and stores pixel values from rasters using a point-format vector file.

    Args:
        vector_path: path to a vector file (shp, geojson, etc)
        raster_paths: raster paths to extract pixel values from
        labels: band name labels. should match the total number of bands across all raster_paths
        drop_na: drop all records with no-data values
        quiet: silence progress bar output.

    Returns:
        gdf: GeoDataFrame annotated with the pixel values from each raster
    """
    # format the inputs
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    gdf = gpd.read_file(vector_path)
    raster_df = annotate_geoseries(gdf.geometry, raster_paths, labels, drop_na)
    gdf = pd.concat([gdf, raster_df.drop(["geometry"], axis=1, errors="ignore")], axis=1)

    return gdf


def annotate_geoseries(
    points: gpd.GeoSeries,
    raster_paths: list,
    labels: list = None,
    drop_na: bool = True,
    dtype: str = None,
    quiet: bool = False,
) -> (gpd.GeoDataFrame, np.ndarray):
    """Reads and stores pixel values from rasters using point locations.

    Args:
        points: GeoSeries with point locations.
        raster_paths: rasters to extract pixel values from.
        labels: band labels. must match the total number of bands for all raster_paths.
        drop_na: drop records with no-data values.
        dtype: output column data type. uses the first raster's dtype by default.
        quiet: silence progress bar output.

    Returns:
        gdf: GeoDataFrame annotated with the pixel values from each raster
    """
    # format the inputs
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    # get the dataset dimensions
    n_rasters = len(raster_paths)

    # create arrays and flags for updating
    raster_values = []
    valid_idxs = []
    nodata_flag = False

    # annotate each point with the pixel values for each raster
    for raster_idx, raster_path in tqdm(
        enumerate(raster_paths), desc="Raster", total=n_rasters, disable=quiet, **tqdm_opts
    ):
        with rio.open(raster_path, "r") as src:
            # reproject points to match raster and convert to a dataframe
            if not crs_match(points.crs, src.crs):
                points = points.to_crs(src.crs)

            # use the first rasters dtype for the output array if not set
            if raster_idx == 0 and dtype is None:
                dtype = src.dtypes[0]

            # get the raster row/col indices for each point and the respective read windows
            xys = [(point.x, point.y) for point in points]

            # read each pixel value
            n_points = len(points)
            samples_iter = list(
                tqdm(
                    src.sample(xys, masked=False),
                    desc="Sample",
                    total=n_points,
                    leave=False,
                    disable=quiet,
                    **tqdm_opts,
                )
            )
            samples = np.array(samples_iter, dtype=dtype)
            raster_values.append(samples)

            # identify nodata points to remove later
            if drop_na and src.nodata is not None:
                nodata_flag = True
                valid_idxs.append(samples[:, 0] != src.nodata)

    # merge the arrays from each raster
    values = np.concatenate(raster_values, axis=1, dtype=dtype)

    if nodata_flag:
        valid = np.all(valid_idxs, axis=0).reshape(-1, 1)
        values = np.concatenate([values, valid], axis=1, dtype=dtype)
        labels.append("valid")
        # values = values[valid, :]
        # points = points.iloc[valid]
        # points.index = range(valid.sum())

    # convert to a geodataframe
    gdf = gpd.GeoDataFrame(values, geometry=points.geometry, columns=labels)

    return gdf


# raster writing tools


def apply_model_to_array(
    model: BaseEstimator,
    array: np.ndarray,
    nodata: float,
    nodata_idx: int,
    count: int = 1,
    dtype: str = "float32",
    predict_proba: bool = False,
    **kwargs,
) -> np.ndarray:
    """Applies a model to an array of covariates.

    Covariate array should be of shape (nbands, nrows, ncols).

    Args:
        model: object with a `model.predict()` function
        array: array of shape (nbands, nrows, ncols) with pixel values
        nodata: numeric nodata value to apply to the output array
        nodata_idx: array of bools with shape (nbands, nrows, ncols) containing nodata locations
        count: number of bands in the prediction output
        dtype: prediction array dtype
        predict_proba: use model.predict_proba() instead of model.predict()
        **kwargs: additonal keywords to pass to model.predict()

    Returns:
        ypred_window: Array of shape (nrows, ncols) with model predictions
    """
    # only apply to valid pixels
    valid = ~nodata_idx.any(axis=0)
    covariates = array[:, valid].transpose()
    ypred = model.predict(covariates, **kwargs) if not predict_proba else model.predict_proba(covariates, **kwargs)

    # reshape to the original window size
    rows, cols = valid.shape
    ypred_window = np.zeros((count, rows, cols), dtype=dtype) + nodata
    ypred_window[:, valid] = ypred.transpose()

    return ypred_window


def apply_model_to_rasters(
    model: BaseEstimator,
    raster_paths: list,
    output_path: str,
    resampling: rio.enums.Enum = rio.enums.Resampling.average,
    count: int = 1,
    dtype: str = "float32",
    nodata: float = -9999,
    driver: str = "GTiff",
    compress: str = "deflate",
    bigtiff: bool = True,
    template_idx: int = 0,
    windowed: bool = True,
    predict_proba: bool = False,
    ignore_sklearn: bool = True,
    quiet: bool = False,
    **kwargs,
) -> None:
    """Applies a trained model to a list of raster datasets.

    The list and band order of the rasters must match the order of the covariates
    used to train the model. It reads each dataset block-by-block, applies
    the model, and writes gridded predictions. If the raster datasets are not
    consistent (different extents, resolutions, etc.), it wll re-project the data
    on the fly, with the grid size, extent and projection based on a 'template'
    raster.

    Args:
        model: object with a model.predict() function
        raster_paths: raster paths of covariates to apply the model to
        output_path: path to the output file to create
        resampling: resampling algorithm to apply to on-the-fly reprojection
            from rasterio.enums.Resampling
        count: number of bands in the prediction output
        dtype: the output raster data type
        nodata: output nodata value
        driver: output raster format
            from rasterio.drivers.raster_driver_extensions()
        compress: compression to apply to the output file
        bigtiff: specify the output file as a bigtiff (for rasters > 2GB)
        template_idx: index of the raster file to use as a template.
            template_idx=0 sets the first raster as template
        windowed: apply the model using windowed read/write
            slower, but more memory efficient
        predict_proba: use model.predict_proba() instead of model.predict()
        ignore_sklearn: silence sklearn warning messages
        quiet: silence progress bar output
        **kwargs: additonal keywords to pass to model.predict()

    Returns:
        None: saves model predictions to disk.
    """
    # make sure the raster_paths are iterable
    raster_paths = to_iterable(raster_paths)

    # get and set template parameters
    windows, dst_profile = create_output_raster_profile(
        raster_paths,
        template_idx,
        count=count,
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

    # set a dummy nodata variable if none is set
    # (acutal nodata reads handled by rasterios src.read(masked=True) method)
    nodata = nodata or 0

    # turn off sklearn warnings
    if ignore_sklearn:
        warnings.filterwarnings("ignore", category=UserWarning)

    # open all rasters to read from later
    srcs = [rio.open(raster_path) for raster_path in raster_paths]

    # use warped VRT reads to align all rasters pixel-pixel if not aligned
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
        for window in tqdm(windows, desc="Window", disable=quiet, **tqdm_opts):
            # create stacked arrays to handle multi-raster, multi-band inputs
            # that may have different nodata locations
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

                predictions = apply_model_to_array(
                    model,
                    covariates,
                    nodata,
                    nodata_idx,
                    count=count,
                    dtype=dtype,
                    predict_proba=predict_proba,
                    **kwargs,
                )
                dst.write(predictions, window=window)

            except NoDataException:
                continue


def validate_gpd(geo: Vector) -> None:
    """Validates whether an input is a GeoDataFrame or a GeoSeries.

    Args:
        geo: an input variable that should be in GeoPandas format

    Raises:
        TypeError: geo is not a GeoPandas dataframe or series
    """
    if not (isinstance(geo, gpd.GeoDataFrame) or isinstance(geo, gpd.GeoSeries)):
        raise TypeError("Input must be a GeoDataFrame or GeoSeries")


def validate_polygons(geometry: Vector) -> pd.Index:
    """Iterate over a geoseries to find rows with invalid geometry types.

    Args:
        geometry: a GeoSeries or GeoDataFrame with polygon geometries

    Returns:
        an index of rows with valid polygon types
    """
    if isinstance(geometry, gpd.GeoDataFrame):
        geometry = geometry.geometry

    index = []
    for idx, geom in enumerate(geometry):
        if not (isinstance(geom, Polygon) or isinstance(geom, MultiPolygon)):
            index.append(idx)

    if len(index) > 0:
        warnings.warn(
            f"Input geometry had {len(index)} invalid geometries. "
            "These will be dropped, but with the original index preserved."
        )
        geometry.drop(index=index, inplace=True)

    return geometry.index


def read_raster_from_polygon(src: rio.DatasetReader, poly: Union[Polygon, MultiPolygon]) -> np.ma.MaskedArray:
    """Read valid pixel values from all locations inside a polygon
        Uses the polygon as a mask in addition to the existing raster mask

    Args:
        src: an open rasterio dataset to read from
        poly: a shapely Polygon or MultiPolygon

    Returns:
        masked array of shape (nbands, nrows, ncols)
    """
    # get the read parameters
    window = rio.windows.from_bounds(*poly.bounds, src.transform)
    transform = rio.windows.transform(window, src.transform)

    # get the data
    data = src.read(window=window, masked=True, boundless=True)
    bands, rows, cols = data.shape
    poly_mask = geometry_mask([poly], transform=transform, out_shape=(rows, cols))

    # update the mask
    data[:, poly_mask] = np.ma.masked

    return data


def zonal_stats(
    polygons: Vector,
    raster_paths: list,
    labels: list = None,
    all_touched: bool = True,
    mean: bool = True,
    stdv: bool = True,
    min: bool = False,
    max: bool = False,
    count: bool = False,
    sum: bool = False,
    skew: bool = False,
    kurtosis: bool = False,
    mode: bool = False,
    all: bool = False,
    percentiles: list = [],
    quiet: bool = False,
) -> gpd.GeoDataFrame:
    """Compute raster summary stats for each polygon in a GeoSeries or GeoDataFrame.

    Args:
        polygons: GeoSeries or GeoDataFrame with polygon geometries.
        raster_paths: list of paths to rasters to summarize
        labels: band labels. must match the total number of bands for all raster_paths.
        all_touched: include all pixels that touch a polygon.
            set to False to only include pixels whose centers intersect the polygon
        mean, min, max, count, sum, stdv, skew, kurtosis, mode:
            set to True to compute these stats
        all: compute all of the above stats
        percentiles: list of 0-100 percentile ranges to compute
        quiet: silence progress bar output

    Returns:
        GeoDataFrame with zonal stats for each raster band in new columns.
            If `polygons` is a GeoDataFrame, the zonal stats columns are appended
            to the original input.
    """
    # format the input geometries
    validate_gpd(polygons)
    valid_idx = validate_polygons(polygons)
    polygons = polygons.iloc[valid_idx]
    is_df = isinstance(polygons, gpd.GeoDataFrame)
    polys = polygons.geometry if is_df else polygons

    # format the input labels
    raster_paths = to_iterable(raster_paths)
    labels = format_band_labels(raster_paths, labels)

    # get the bands and indexes for each covariate raster
    nbands, band_idx = get_raster_band_indexes(raster_paths)

    # get the stats methods to compute for each feature
    stats_methods = get_raster_stats_methods(
        mean=mean,
        min=min,
        max=max,
        count=count,
        sum=sum,
        stdv=stdv,
        skew=skew,
        kurtosis=kurtosis,
        mode=mode,
        percentiles=percentiles,
        all=all,
    )

    # create dataframes for each raster and concatenate at the end
    raster_dfs = []

    # run zonal stats raster-by-raster (instead of iterating first over geometries)
    for r, raster in tqdm(enumerate(raster_paths), total=len(raster_paths), desc="Raster", disable=quiet, **tqdm_opts):
        # format the band labels
        band_labels = labels[band_idx[r] : band_idx[r + 1]]
        n_raster_bands = band_idx[r + 1] - band_idx[r]
        stats_labels = []
        for method in stats_methods:
            stats_labels.append([f"{band}_{method.name}" for band in band_labels])

        # open the raster for reading
        with rio.open(raster, "r") as src:
            # reproject the polygon data as necessary
            if not crs_match(polys.crs, src.crs):
                polys = polys.to_crs(src.crs)

            # create output arrays to store each stat's output
            stats_arrays = []
            for method in stats_methods:
                dtype = method.dtype or src.dtypes[0]
                stats_arrays.append(np.zeros((len(polys), n_raster_bands), dtype=dtype))

            # iterate over each geometry to read data and compute stats
            for p, poly in tqdm(
                enumerate(polys), total=len(polys), desc="Polygon", leave=False, disable=quiet, **tqdm_opts
            ):
                data = read_raster_from_polygon(src, poly)
                for method, array in zip(stats_methods, stats_arrays):
                    array[p, :] = method.reduce(data)

        # convert each stat's array into dataframes and merge them together
        stats_dfs = [pd.DataFrame(array, columns=labels) for array, labels in zip(stats_arrays, stats_labels)]
        raster_dfs.append(pd.concat(stats_dfs, axis=1))

    # merge the outputs from each raster
    if is_df:
        merged = gpd.GeoDataFrame(pd.concat([polygons] + raster_dfs, axis=1), crs=polygons.crs)
    else:
        merged = gpd.GeoDataFrame(pd.concat(raster_dfs, axis=1), geometry=polygons, crs=polygons.crs)

    return merged


# sample weighting methods


def nearest_point_distance(
    points1: Vector, points2: Vector = None, n_neighbors: int = 1, cpu_count: int = -1
) -> np.ndarray:
    """Compute the average euclidean distance to the nearest point in a series.

    Args:
        points1: return the closest distance *from* these points
        points2: return the closest distance *to* these points
            if None, compute the distance to the nearest points
            in the points1 series
        n_neighbors: compute the average distance to the nearest n_neighbors.
            set to -1 to compute the distance to all neighbors.
        cpu_count: number of cpus to use for estimation.
            -1 uses all cores

    Returns:
        array of shape (len(points),) with the distance to
            each point's nearest neighbor
    """
    if points1.crs.is_geographic:
        warnings.warn("Computing distances using geographic coordinates is bad")

    pta1 = np.array(list(zip(points1.geometry.x, points1.geometry.y)))
    k_offset = 1

    if points2 is None:
        pta2 = pta1
        k_offset += 1

    else:
        pta2 = np.array(list(zip(points2.geometry.x, points2.geometry.y)))
        if not crs_match(points1.crs, points2.crs):
            warnings.warn("CRS mismatch between points")

    if n_neighbors < 1:
        n_neighbors = len(pta2) - k_offset

    tree = KDTree(pta1)
    k = np.arange(n_neighbors) + k_offset
    distance, idx = tree.query(pta2, k=k, workers=cpu_count)

    return distance.mean(axis=1)


def distance_weights(points: Vector, n_neighbors: int = -1, center: str = "median", cpu_count: int = -1) -> np.ndarray:
    """Compute sample weights based on the distance between points.

    Assigns higher scores to isolated points, lower scores to clustered points.

    Args:
        points: point-format GeoSeries or GeoDataFrame
        n_neighbors: compute weights based on average distance to the nearest n_neighbors
            set to -1 to compute the distance to all neighbors.
        center: rescale the weights to center the mean or median of the array on 1
            accepts either 'mean' or 'median' as input.
            pass None to ignore.
        cpu_count: number of cpus to use for estimation.
            -1 uses all cores

    Returns:
        array of shape (len(points),) with scaled sample weights. Scaling
            is performed by dividing by the maximum value, preserving the
            relative scale of differences between the min and max distance.
    """
    distances = nearest_point_distance(points, n_neighbors=n_neighbors, cpu_count=cpu_count)
    weights = distances / distances.max()

    if center is not None:
        if center.lower() == "mean":
            weights /= weights.mean()

        elif center.lower() == "median":
            weights /= np.median(weights)

    return weights
