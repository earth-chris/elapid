"""Geospatial data operations like reading/writing/indexing raster and vector data"""
import geopandas as gpd
import rasterio as rio
from shapely.geometry import Point


def xy_to_geoseries(x, y, crs="epsg:4326"):
    """
    Converts x/y data into a geopandas geoseries dataframe.

    :param x: 1-D array-like of x location values
    :param y: 1-D array-like of y location values
    :param crs: the coordinate reference string. accepts anything allowed by pyproj.CRS.from_user_input(). assumes lat/lon.
    :returns: gs, a geopandas geometry geoseries
    """
    points = [Point(x, y) for x, y in zip(x, y)]
    gs = gpd.GeoSeries(points, crs=crs)

    return gs
