import os
from copy import copy

import geopandas as gpd
import numpy as np
import rasterio as rio

from elapid import geo

# set the test raster data paths
directory_path, script_path = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(directory_path, "data")
points = os.path.join(data_path, "test-point-samples.gpkg")
poly = os.path.join(data_path, "test-polygon.gpkg")
raster_1b = os.path.join(data_path, "test-raster-1band.tif")
raster_2b = os.path.join(data_path, "test-raster-2bands.tif")
raster_1b_offset = os.path.join(data_path, "test-raster-1band-offset.tif")
with rio.open(raster_1b, "r") as src:
    raster_1b_profile = copy(src.profile)

# set some crs variables
wkt = 'GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],UNIT["degree",0.01745329251994328,AUTHORITY["EPSG","9122"]],AUTHORITY["EPSG","4326"]]'
proj4 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
epsg_code = 4326
epsg_str = f"EPSG:{epsg_code}"


def test_xy_to_geoseries():
    lon, lat = (-120, 38)
    geoseries = geo.xy_to_geoseries(lon, lat, crs=epsg_str)
    assert geoseries.x[0] == lon
    assert geoseries.y[0] == lat


def test_sample_raster():
    count = 20
    input_raster = raster_1b
    for ignore_mask in [True, False]:
        points = geo.sample_raster(input_raster, count, ignore_mask=ignore_mask)
        with rio.open(input_raster, "r") as src:
            raster_crs = src.crs
            rxmin, rymin, rxmax, rymax = src.bounds

        pxmin, pymin, pxmax, pymax = points.total_bounds

        assert geo.crs_match(points.crs, raster_crs)
        assert pxmin >= rxmin
        assert pymin >= rymin
        assert pxmax <= rxmax
        assert pymax <= rymax


# TODO
def test_sample_bias_file():
    pass


def test_sample_from_vector():
    count = 20
    points = geo.sample_vector(poly, count)
    poly_df = gpd.read_file(poly)

    assert len(points) == count
    for point in points:
        assert poly_df.contains(point).iloc[0]


# TODO
def test_sample_from_geoseries():
    count = 20
    poly_df = gpd.read_file(poly)
    points = geo.sample_geoseries(poly_df.geometry, count)

    assert len(points) == count
    for point in points:
        assert poly_df.contains(point).iloc[0]


def test_parse_crs_string():
    assert geo.parse_crs_string(wkt) == "wkt"
    assert geo.parse_crs_string(proj4) == "proj4"
    assert geo.parse_crs_string(epsg_str) == "epsg"


def test_string_to_crs():
    crs = geo.string_to_crs(wkt)
    assert crs.to_epsg() == epsg_code

    crs = geo.string_to_crs(proj4)
    assert crs.to_epsg() == epsg_code

    crs = geo.string_to_crs(epsg_str)
    assert crs.to_epsg() == epsg_code


def test_crs_match():
    lon, lat = (-120, 38)
    geoseries = geo.xy_to_geoseries(lon, lat, crs=epsg_str)
    gpd_crs = geoseries.crs
    rio_crs = rio.crs.CRS.from_epsg(epsg_code)

    assert geo.crs_match(rio_crs, gpd_crs) is True
    assert geo.crs_match(gpd_crs, wkt) is True
    assert geo.crs_match(rio_crs, epsg_str) is True
    assert geo.crs_match(wkt, proj4) is True
    assert geo.crs_match(rio_crs, "epsg:32610") is False


def test_annotate_vector():
    df = geo.annotate_vector(points, [raster_2b], labels=["band_1", "band_2"])
    pts = gpd.read_file(points)
    assert len(df) == len(pts)
    assert np.isfinite(df["band_1"]).all()
    assert np.isfinite(df["band_2"]).all()


def test_annotate_geoseries():
    # create a single point in the origin of the test data
    with rio.open(raster_1b, "r") as src:
        x, y = src.xy(0, 0)

    geoseries = geo.xy_to_geoseries(x, y, crs=src.crs)

    # test on one band input
    df = geo.annotate_geoseries(geoseries, [raster_1b], labels=["band_1"])
    b1 = df["band_1"].iloc[0]
    assert b1 == 0

    # test on two band input
    df = geo.annotate_geoseries(geoseries, [raster_2b], labels=["band_1", "band_2"])
    b1 = df["band_1"].iloc[0]
    b2 = df["band_2"].iloc[0]
    assert b1 == 0
    assert b2 == 65280

    # test on multi-raster input
    df = geo.annotate_geoseries(geoseries, [raster_1b, raster_2b], labels=["band_1", "band_2", "band_3"])
    b1 = df["band_1"].iloc[0]
    b2 = df["band_2"].iloc[0]
    b3 = df["band_3"].iloc[0]
    assert b1 == 0
    assert b2 == 0
    assert b3 == 65280

    # ensure on-the-fly reprojection works
    geoseries_ll = geoseries.to_crs("EPSG:4326")
    df = geo.annotate_geoseries(geoseries_ll, [raster_2b], labels=["band_1", "band_2"])
    b1 = df["band_1"].iloc[0]
    b2 = df["band_2"].iloc[0]
    assert b1 == 0
    assert b2 == 65280


def test_annotate():
    raster_paths = [raster_1b, raster_2b]
    labels = ["band_1", "band_2", "band_3"]
    points_df = gpd.read_file(points)

    from_path = geo.annotate(points, raster_paths, labels=labels, drop_na=True)
    from_df = geo.annotate(points_df, raster_paths, labels=labels, drop_na=False)
    from_gs = geo.annotate(points_df.geometry, raster_paths, labels=labels, drop_na=True)

    assert (from_path.geometry == from_df.geometry).all()
    assert (from_path.geometry == from_gs.geometry).all()
    assert (from_path.band_1 == from_df.band_1).all()
    assert (from_path.band_3 == from_gs.band_3).all()


# TODO: more thorough evaluation: just test that these run for now
def test_zonal_stats():
    # test all stats
    poly_df = gpd.read_file(poly)
    zs = geo.zonal_stats(poly_df, raster_2b, all=True)
    assert len(zs) == len(poly_df)
    assert geo.crs_match(zs.crs, poly_df.crs)

    # test geoseries read
    zs = geo.zonal_stats(poly_df.geometry, raster_2b, labels=["r1", "r2"])
    assert len(zs) == len(poly_df)
    assert geo.crs_match(zs.crs, poly_df.crs)

    # test reprojection works
    rp = geo.zonal_stats(poly_df.to_crs("EPSG:4326"), raster_2b)
    assert len(rp) == len(poly_df)
    assert geo.crs_match(zs.crs, poly_df.crs)

    # test percentiles
    pc = geo.zonal_stats(poly_df, raster_2b, percentiles=[10, 90])
    assert len(pc) == len(poly_df)
    assert geo.crs_match(zs.crs, poly_df.crs)

    # test multi-raster read
    mr = geo.zonal_stats(poly_df, [raster_1b_offset, raster_2b])
    assert len(mr) == len(poly_df)
    assert geo.crs_match(zs.crs, poly_df.crs)


def test_nearest_point_distance():
    pts = gpd.read_file(points)
    dist = geo.nearest_point_distance(pts)
    xmin, ymin, xmax, ymax = pts.total_bounds
    assert dist.min() > 0
    assert dist.max() <= xmax - xmin
    assert dist.max() <= ymax - ymin


def test_distance_weights():
    pts = gpd.read_file(points)
    weights = geo.distance_weights(pts, center=None)
    assert weights.max() == 1.0
    assert weights.min() > 0

    cmed = geo.distance_weights(pts, center="median")
    assert cmed.max() > 1.0

    cmean = geo.distance_weights(pts, center="mean")
    print(cmean.mean())
    np.testing.assert_approx_equal(cmean.mean(), 1.0, 5)
