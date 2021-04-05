"""Geospatial data operations like reading/writing/indexing raster and vector data"""

from multiprocessing import Pool

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
from shapely.geometry import MultiPoint, Point

from elapid.utils import _ncpus


def xy_to_geoseries(x, y, crs="epsg:4326"):
    """
    Converts x/y data into a geopandas geoseries dataframe.

    :param x: 1-D array-like of x location values
    :param y: 1-D array-like of y location values
    :param crs: the coordinate reference string. accepts anything allowed by pyproj.CRS.from_user_input(). assumes lat/lon.
    :returns: gs, a geopandas Point geometry geoseries
    """
    points = [Point(x, y) for x, y in zip(x, y)]
    gs = gpd.GeoSeries(points, crs=crs)

    return gs


def pseudoabsence_from_raster(raster_path, count, ignore_mask=False):
    """
    Creates a random geographic sampling of points based on a raster's extent.
      Selects from unmasked locations if the rasters nodata value is set.

    :param raster_path: str raster file path to sample locations from
    :param count: the total number of samples to generate
    :param ignore_mask: sample from the full extent of the raster instead of unmasked areas only
    :returns: points, a geopandas Point geoseries
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
    """
    Creates a semi-random geographic sampling of points weighted towards biased areas.

    :param raster_path: str raster bias file path to sample from. Pixel values can occupy an arbitrary range but must be odered from low -> high probability
    :param count: the total number of samples to generate
    :param ignore_mask: sample from the full extent of the raster instead of unmasked areas only
    :returns: points, a geopandas Point geoseries
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
    """
    Creates a random geographic sampling of points inside of a polygon/multipolygon type vector file.

    :param vector_path: str path to a vector file (shp, geojson, etc)
    :param count: the total number of random samples to generate
    :param overestimate: a scaler to generate extra samples to toss points outside of the polygon/inside it's bounds
    :returns: points, a geopandas Point geoseries
    """
    gdf = gpd.read_file(vector_path)
    return pseudoabsence_from_geoseries(gdf.geometry, count, overestimate=overestimate)


def pseudoabsence_from_geoseries(geoseries, count, overestimate=2):
    """
    Creates a random geographic sampling of points inside of a geoseries polygon/multipolygon

    :param geoseries: a geopandas geoseries (e.g., gdf['geometry']) with polygons/multipolygons
    :param count: the total number of random samples to generate
    :param overestimate: a scaler to generate extra samples to toss points outside of the polygon/inside it's bounds
    :returns: points, a geopandas Point geoseries
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


def raster_values_from_vector(vector_path, raster_paths, labels=None):
    """
    Reads and stores pixel values from a set of raster paths based on a point-format vector file.

    :param vector_path: str path to a vector file (shp, geojson, etc)
    :param raster_paths: a list of raster paths to extract pixel values from
    :param labels: a list of band name labels. should match the total number of bands across all raster_paths
    :returns: gdf, a geodataframe with the pixel values from each raster appended to the original vector columns
    """
    gdf = gpd.read_file(vector_path)
    raster_df = raster_values_from_geoseries(gdf.geometry, raster_paths, labels)
    gdf = pd.concat([gdf, raster_df.drop(["geometry"], axis=1, errors="ignore")], axis=1)
    return gdf


def raster_values_from_geoseries(geoseries, raster_paths, labels=None):
    """
    Reads and stores pixel values from a set of raster paths based on a geoseries of point locations.

    :param geoseries: a geopandas geoseries (e.g., gdf['geometry']) with point locations
    :param raster_paths: a list of raster paths to extract pixel values from
    :param labels: a list of band name labels. should match the total number of bands across all raster_paths
    :returns: gdf, a geopandas geodataframe with the geoseries point locations and pixel values from each raster
    """

    # make sure the raster_paths are iterable
    if isinstance(raster_paths, (str)):
        raster_paths = [raster_paths]

    # get the band count to use and reconcile with the number of labels passed
    n_bands = 0
    for path in raster_paths:
        with rio.open(path) as src:
            n_bands += src.count

    if labels is None:
        labels = ["band_{:03d}".format(i + 1) for i in range(n_bands)]
    else:
        assert len(labels) == n_bands, "Number of raster bands ({}) does not match number of labels ({})".format(
            n_bands, len(labels)
        )

    # apply this function over every geoseries row
    def read_pixel_value(point, source):
        row, col = source.index(point.x, point.y)
        window = rio.windows.Window(col, row, 1, 1)
        value = source.read(window=window)
        return np.squeeze(value)

    # apply this function over every raster_path
    def parallel_raster_reads(raster_path):
        with rio.open(raster_path) as src:
            points = geoseries.to_crs(src.crs)
            values = points.apply(read_pixel_value, source=src)

        return values

    # TODO: multiprocess the raster reads
    # pool = Pool(n_workers)
    # df = pd.concat(pool.map(parallel_raster_reads, raster_paths), axis=1)
    # pool.close()
    # pool.join()
    df = pd.concat([parallel_raster_reads(raster_path) for raster_path in raster_paths], axis=1)
    df.columns = labels
    gdf = gpd.GeoDataFrame(df, geometry=geoseries, crs=geoseries.crs)

    return gdf


def apply_model_to_rasters(
    model,
    raster_paths,
    output_path,
    transform="logistic",
    template_idx=0,
    resampling=rio.enums.Resampling.average,
    output_driver="GTiff",
    compress="deflate",
    bigtiff=True,
):
    """
    Applies a trained model to a list of raster datasets. The list and band order of the rasters must
      match the order of the covariates used to train the model. This function can be applied to rasters
      of different projections, grid sizes, etc. It resamples each dataset on the fly in a tile-wise
      basis, applies the model, and writes gridded predictions. It selects the grid size, extent and
      projection based on a 'template' raster

    :param model: the trained model with a model.predict() function
    :param raster_paths: a list of raster paths of covariates to apply the model to
    :param output_path: path to the output file to create
    :param transform: the maxent model transformation type. Select from ["raw", "exponential", "logistic", "cloglog"].
    :param template_ids: the index of the raster file to use as a template. template_idx=0 sets the first raster as template
    :param resampling: the resampling algorithm to apply to on-the-fly reprojection (from rasterio.enums.Resampling)
    :param output_driver: the output raster file format (from rasterio.drivers.raster_driver_extensions())
    :param use_default_creation_options: bool to use elapid-recommended raster format options (e.g. compression)
    :param creation_options: list of driver-specific raster creation options
    :returns: none, saves model predictions to disk.
    """

    # make sure the raster_paths are iterable
    if isinstance(raster_paths, (str)):
        raster_paths = [raster_paths]

    # get and set template parameters
    with rio.open(raster_paths[template_idx]) as src:
        windows = src.block_windows()
        dst_profile = src.profile
        dst_profile.update(
            count=1,
            dtype="float32",
            compress=compress,
            driver=output_driver,
        )
        if bigtiff:
            dst_profile.update(BIGTIFF="YES")

    vrt_options = {
        "resampling": resampling,
        "transform": dst_profile["transform"],
        "crs": dst_profile["crs"],
        "height": dst_profile["height"],
        "width": dst_profile["width"],
    }

    # get the bands and indexes for each covariate raster
    nbands = 0
    band_idx = [0]
    for i, raster_path in enumerate(raster_paths):
        with rio.open(raster_path) as src:
            nbands += src.count
            band_idx.append(band_idx[i] + src.count)

    # read and reproject blocks from each data source and write predictions to disk
    with rio.open(output_path, "w", **dst_profile) as dst:

        # open all raster paths to read from later
        srcs = [rio.open(raster_path) for raster_path in raster_paths]
        vrts = [rio.vrt.WarpedVRT(src, **vrt_options) for src in srcs]

        # iterate over each data block, read from each source, and apply the model
        for _, window in windows:
            ncols = window.width
            nrows = window.height
            covariate_window = np.zeros((nbands, nrows, ncols), dtype=np.float) - 1.0
            vrt_options.update(width=ncols, height=nrows)

            for i, vrt in enumerate(vrts):
                covariate_window[band_idx[i] : band_idx[i + 1]] = vrt.read(window=window)
            # for i, src in enumerate(srcs):
            #    with rio.vrt.WarpedVRT(src, **vrt_options) as vrt:
            #        print(vrt.read(window=window).shape)
            #        print(vrt.profile)
            #        covariate_window[band_idx[i]:band_idx[i + 1]] = vrt.read(window=window)

            covariate_array = covariate_window.transpose((1, 2, 0)).reshape((nrows * ncols, nbands))
            predictions_array = model.predict(covariate_array, is_features=False, transform=transform)
            predictions_window = predictions_array.to_numpy(dtype=np.float32).reshape((1, nrows, ncols))
            dst.write(predictions_window, window=window)

        for src in srcs:
            src.close()
