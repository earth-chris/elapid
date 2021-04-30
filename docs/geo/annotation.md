Annotation refers to reading and storing raster values at the locations of a series of point occurrence records.

Once you have your species presence and pseudo-absence records, you can annotate these records with the covariate data from each location.

```python
pseudoabsence_covariates = elapid.raster_values_from_geoseries(
    pseudoabsence_points,
    raster_path,
    drop_na = True,
)
```

This could also be done with `raster_values_from_vector(vector_path, raster_path)` if you haven't already loaded the geoseries data into memory.

This function, since it's geographically indexed, doesn't require the point data and the raster data to be in the same projection. `elapid` handles reprojection and sampling on the fly.

It also allows you to pass multiple raster files, which can be in different projections, extents, or grid sizes. This means you don't have to explicitly re-sample your raster data prior to analysis, which is always a chore.

```python
raster_paths = [
    "/home/slug/california-leaf-area-index.tif", # 1-band vegetation data
    "/home/slug/global-cloud-cover.tif", # 3-band min, mean, max annual cloud cover
    "/home/slug/usa-mean-temperature.tif", # 1-band mean temperature
]

# since you have five raster bands total, specify each band label
labels = [
    "LAI",
    "CLD-min",
    "CLD-mean",
    "CLD-max",
    "TMP-mean",
]

pseudoabsence_covariates = elapid.raster_values_from_geoseries(
    pseudoabsence_points,
    raster_paths,
    labels = labels
    drop_na = True,
)
```

If you don't specify the labels, `elapid` will assign `['band_001', 'band_002', ...]`.
