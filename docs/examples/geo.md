# Geospatial Data Support

The following code snippets are guidelines for how to work with spatially-explicit datasets in `elapid`.

## A note on point-format data types

Almost all of the `elapid` point sampling and indexing tools use `geopandas.GeoDataFrame` or `geopandas.GeoSeries` objects. The latter are the format of the `geometry` column for a `GeoDataFrame`.

```python
import geopandas as gpd

gdf = gpd.read_file('/path/to/some/vector.shp')
print(type(gdf.geometry))

> <class 'geopandas.geoseries.GeoSeries'>
```

Several `elapid` routines return `GeoSeries` objects (like `elapid.sample_raster()`) or `GeoDataFrame` objects (like `elapid.annotate()`). It also includes tools for converting x/y data from list, `numpy.ndarray`, or `pandas.DataFrame` to `GeoSeries`.

---

## Working with X/Y data

**From CSVs**

Sometimes you don't have a vector of point-format location data. The `java` implementation of maxent uses csv files, for example. You can convert those using the `elapid.xy_to_geoseries()` function:

```python
import pandas as pd

csv_path = "/home/cba/ariolimax-californicus.csv"
df = pd.read_csv(csv_path)
presence = elapid.xy_to_geoseries(df.x, df.y, crs="EPSG:32610")
```

This assumes that the input CSV file has an `x` and a `y` column.

Make sure you specify the projection of your x/y data. The default assumption is lat/lon, which in many cases is not correct.

**From arrays or lists**

You can also convert arbitrary arrays of x/y data:

```python
lons = [-122.49, 151.0]
lats = [37.79, -33.87]
locations = elapid.xy_to_geoseries(lons, lats)
print(locations)

> 0    POINT (-122.49000 37.79000)
> 1    POINT (151.00000 -33.87000)
> dtype: geometry
```

---

## Drawing point location samples

In addition to species occurrence records (`y = 1`), Maxent requires a set of pseudo-absence/background points (`y = 0`). These are a random geographic sampling of where you might expect to find a species across the target landscape.

**From a raster's extent**

You can use `elapid` to create a random geographic sampling of points from unmasked locations within a raster's extent:

```python
count = 10000 # the number of points to generate
pseudoabsence_points = elapid.sample_raster(raster_path, count)
```

If you have a large raster that doesn't fit in memory, you can also set `ignore_mask=True` to use the rectangular bounds of the raster to draw samples.

```python
pseudoabsence_points = elapid.sample_raster(raster_path, count, ignore_mask=True)
```

**From a polygon vector**

Species occurrence records are often biased in their collection (collected near roads, in protected areas, etc.), so we typically need to be more precise in where we select pseudo-absence points. You could use a vector with a species range map to select records:

```python
range_path = "/home/slug/ariolimax-californicus-range.shp"
pseudoabsence_points = elapid.sample_vector(range_path, count)
```

This currently computes the spatial union of all polygons within a vector, compressing the geometry into a single MultiPolygon object to sample from.

If you've already computed a polygon using geopandas, you can pass it instead to `elapid.sample_geoseries()`, which is what `sample_vector()` does under the hood.

**From a bias raster**

You could also pass a raster bias file, where the raster grid cells contain information on the probability of sampling an area:

```python
# assume covariance between vertebrate and invertebrate banana slugs
bias_path = "/home/slug/proximity-to-ucsc.tif"
pseudoabsence_points = sample_bias_file(bias_path)
```

The grid cells can be an arbitrary range of values. What's important is that the values encode a linear range of numbers that are higher where you're more likely to draw a sample. The probability of drawing a sample is dependent on two factors: the range of values provided and the frequency of values across the dataset.

So, for a raster with values of `1` and `2`, you're sampling probability for raster locations of `2` is twice that as `1` locations. If these occur in equal frequency (i.e. half the data are `1` valuess, half are `2` values), then you'll likely sample twice as many areas with `2` values. But if the frequency of `1` values is much greater than `2` values, you'll shift the distribution. But you're still more likely, on a per-draw basis, to draw samples from `2` locations.

The above example prioritizes sampling frequency in the areas around UC Santa Cruz, home to all types of slug, based on the distance to the campus.

---

## Point Annotation

Annotation refers to reading and storing raster values at the locations of a series of point occurrence records in a single `GeoDataFrame` table.

Once you have your species presence and pseudo-absence records, you can annotate these records with the covariate data from each location.

```python
pseudoabsence_covariates = elapid.annotate(
    pseudoabsence_points,
    list_of_raster_paths,
    drop_na = True,
)
```

This function, since it's geographically indexed, doesn't require the point data and the raster data to be in the same projection. `elapid` handles reprojection and sampling on the fly.

It also allows you to pass multiple raster files, which can be in different projections, extents, or grid sizes. This means you don't have to explicitly re-sample your raster data prior to analysis, which is always a chore.

```python
raster_paths = [
    "/home/slug/california-leaf-area-index.tif", # 1-band vegetation data
    "/home/slug/global-cloud-cover.tif", # 3-band min, mean, max annual cloud cover
    "/home/slug/usa-mean-temperature.tif", # 1-band mean temperature
]

# this fake dataset has five raster bands total, so specify each band label
labels = [
    "LAI",
    "CLD-min",
    "CLD-mean",
    "CLD-max",
    "TMP-mean",
]

pseudoabsence_covariates = elapid.annotate(
    pseudoabsence_points,
    raster_paths,
    labels = labels
    drop_na = True,
)
```

If you don't specify the labels, `elapid` will assign `['b1', 'b2', ..., 'bn']` labels to each column.

Setting `drop_na=True` requires that the raster datasets have `nodata` values assigned in their metadata. These point locations will be dropped from the output dataframe, which will have fewer rows than the input points.

---

## Zonal statistics

In addition to the tools for working with Point data, `elapid` contains a routine for calculating zonal statistics from Polygon or MutliPolygon geometry types.

This routine reads an array of raster data covering the extent of a polygon, masks the areas outside the polygon, and computes summary statistics such as the mean, standard deviation, and mode of the array.

```python
ecoregions = gpd.read_file('/path/to/california-ecoregions.shp')
zs = elapid.zonal_stats(
    ecoregions,
    raster_paths,
    labels = labels,
    mean = True,
    stdv = True,
    percentiles = [10, 90],
)
```

Which stats are reported is managed by a set of keywords (`count=True`, `sum=True`, `skew=True`). The `all=True` keyword is a shortcut to compute all of the stats. You'll still have to explicitly pass a list of `percentiles`, however, like this:

```python
zs = elapid.zonal_stats(
    ecoregions,
    raster_paths,
    labels = labels,
    all = True,
    percentiles = [25, 50, 75],
)
```

What sets the `elapid` zonal stats method apart from other zonal stats packages is it:

- handles reprojection on the fly, meaning the input vector/raster data don't need to be reprojected a priori
- handles mutli-band input, computing summary stats on all bands (instead of having to specify which band)
- handles multi-raster input, reading inputs in serial but creating a single output `GeoDataFrame`.

---

## Applying predictions to data

Once you've fit a model (we're skipping a step here, but see the [Maxent overview](../sdm/maxent.md) for an example), you can apply it to a set of raster covariates to produce gridded habitat suitability maps.

```python
elapid.apply_model_to_rasters(
    model,
    raster_paths,
    output_path,
    template_idx = 0,
    transform = "cloglog",
    nodata = -9999,
)
```

The list and band order of the rasters must match the order of the covariates used to train the model. It reads each dataset in a block-wise basis, applies the model, and writes gridded predictions.

If the raster datasets are not consistent (different extents, resolutions, etc.), it wll re-project the data on the fly, with the grid size, extent and projection based on a 'template' raster. Use the `template_idx` keyword to specify the index of which raster file to use as the template (`template_idx=0` sets the first raster as the template).

In the example above, it's important to set the template to the `california-leaf-area-index.tif` file. This is because this is the smallest extent with data, and it'll only read and apply the model to the `usa` and `global` datasets in the area covered by `california`. If you were to set the extent to `usa-mean-temperature.tif`, it would still technically function, but there would be a large region of `nodata` values where there's insufficient covariate coverage.

**Applying other model predictions**

The `apply_model_to_rasters()` function can be used to apply model predictions from any estimator with a `model.predict()` method. This includes the majority of `sklearn` model estimators.

If you wanted to train and apply a Random Forest model, you'd use a pattern like this:

```python
import elapid
from sklearn.ensemble import RandomForestClassifier

x, y = elapid.load_sample_data()

model = RandomForestClassifier()
model.fit(x, y)

input_rasters = ['/path/to/raster1.tif', '/path/to/raster2.tif']
output_raster = 'rf-model-prediction-categorical.tif'
elapid.apply_model_to_rasters(
  model,
  input_rasters,
  output_raster,
)
```

These models contain an additional method for estimating the continuous prediction probabilities. To write these out, set `predict_proba=True`. To use this option, however, you also have to specify the number of output raster bands. Since the sample data is a 2-class model, the output prediction probabilities are 2-band outputs, so we set `count=2`.

```python
output_raster = 'rf-model-prediction-probabilities.tif'
elapid.apply_model_to_rasters(
  model,
  input_rasters,
  output_raster,
  predict_proba = True,
  count = 2,
)
```
