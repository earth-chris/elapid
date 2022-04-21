"""Generates some dummy raster data to test elapid functions."""

import logging
import os
import sys
from copy import copy

import geopandas as gpd
import numpy as np
import rasterio as rio
from shapely.geometry import box

import elapid as ela

# set logging
logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s %(levelname)s %(name)s | %(message)s"),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# set the output raster directory
directory_path, script_path = os.path.split(os.path.abspath(__file__))
data_path = os.path.join(directory_path, "data")
if not os.path.exists(data_path):
    os.mkdir(data_path)

# set georeferencing
width, height = (256, 256)
xres, yres = (20, 20)
crs = rio.crs.CRS.from_epsg(32610)  # utm 10 N
xmin, ymax = (894160, 4181885)  # upper left corner, for the middle of california
xmax = xmin + (xres * width)
ymin = ymax - (yres * height)
transform = rio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

# create an array to store as data
dtype = np.uint16
data = np.arange(width * height, dtype=dtype).reshape((height, width))

# create the output raster profile
profile = copy(rio.profiles.default_gtiff_profile)
profile.update(
    width=width,
    height=height,
    crs=crs,
    transform=transform,
    dtype="uint16",
    count=1,
    nodata=None,
)

# write the 1-band output
output_1b = os.path.join(data_path, "test-raster-1band.tif")
logger.info(f"Writing: {output_1b}")
with rio.open(output_1b, "w+", **profile) as dst:
    dst.write(data, 1)

# write a 2-band output as well to test multi-band reads
output_2b = os.path.join(data_path, "test-raster-2bands.tif")
profile.update(count=2)
logger.info(f"Writing: {output_2b}")
with rio.open(output_2b, "w+", **profile) as dst:
    dst.write(data, 1)
    dst.write(np.flip(data, axis=0), 2)

# move the georeferencing by 1 pixel in each direction to test raster alignment
output_1b_offset = os.path.join(data_path, "test-raster-1band-offset.tif")
jittered_transform = rio.transform.from_bounds(xmin - xres, ymin - yres, xmax - xres, ymax - yres, width, height)
profile.update(count=1, transform=jittered_transform)
logger.info(f"Writing: {output_1b_offset}")
with rio.open(output_1b_offset, "w+", **profile) as dst:
    dst.write(data, 1)

# create a point vector to extract pixel values from
output_samples = os.path.join(data_path, "test-point-samples.gpkg")
count = 50
samples = ela.geo.sample_from_raster(output_1b, count)
samples_df = samples.to_frame("geometry")
logger.info(f"Writing: {output_samples}")
samples_df.to_file(output_samples, driver="GPKG")

# create a square polygon
output_poly = os.path.join(data_path, "test-polygon.gpkg")
bbox = box(*samples_df.total_bounds)
poly_df = gpd.GeoDataFrame(geometry=[bbox], crs=samples_df.crs)
logger.info(f"Writing: {output_poly}")
poly_df.to_file(output_poly, driver="GPKG")
