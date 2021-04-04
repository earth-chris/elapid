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


def sample_polygon_vector(vector_path, count, overestimate=2):
    """
    Creates a random geographic sampling of points inside of a vector file.

    :param vector_path: str path to a vector file (shp, geojson, etc)
    :param count: the total number of random samples to generate
    :param overestimate: a scaler to generate extra samples to toss points outside of the polygon/inside it's bounds
    :returns: points, a geopandas Point geoseries
    """
    gdf = gpd.read_file(vector_path)
    return sample_polygon_geoseries(gdf.geometry, count, overestimate=overestimate)


def sample_polygon_geoseries(geoseries, count, overestimate=2):
    """
    Creates a random geographic sampling of points inside of a geoseries polygon/multipolygon

    :param geoseries: a geopandas geoseries (e.g., gdf['geometry']) with polygons/multipolygons
    :param count: the total number of random samples to generate
    :param overestimate: a scaler to generate extra samples to toss points outside of the polygon/inside it's bounds
    :returns: points, a geopandas Point geoseries
    """
    polygon = geoseries.unary_union
    min_x, min_y, max_x, max_y = polygon.bounds
    ratio = polygon.area / polygon.envelope.area

    samples = np.random.uniform((min_x, min_y), (max_x, max_y), (int(count / ratio * overestimate), 2))
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

    # make sure the paths are iterable
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
        window = rio.windows.Window(row, col, 1, 1)
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
