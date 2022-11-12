```python
import elapid as ela
```

---

Almost all of the `elapid` point sampling and indexing tools use `geopandas.GeoDataFrame` or `geopandas.GeoSeries` objects. The latter are the format of the `geometry` column for a GeoDataFrame.

```python
import geopandas as gpd

gdf = gpd.read_file('/path/to/some/vector.shp')
print(type(gdf.geometry))

> <class 'geopandas.geoseries.GeoSeries'>
```

Several `elapid` routines return GeoSeries objects (like `ela.sample_raster()`). Others return GeoDataFrames (like `ela.annotate()` and `ela.zonal_stats()`). Be prepared to work with GeoPandas objects.

---

## Working with X/Y data

**From CSVs**

Sometimes you don't have a vector of point-format location data. You may instead have a series of x/y locations in a CSV file, for example. You can convert those using the `elapid.xy_to_geoseries()` function:

```python
import pandas as pd
import elapid as ela

csv_path = "/home/slug/ariolimax-californicus.csv"
df = pd.read_csv(csv_path)
presence = ela.xy_to_geoseries(df.x, df.y, crs="EPSG:32610")
```

This assumes that the input CSV file has an `x` and a `y` column.

Be sure to specify the projection of your x/y data! The default assumption is lat/lon, which in many cases is not correct.

**From arrays or lists**

You can also convert arbitrary arrays of x/y data to GeoSeries:

```python
lons = [-122.49, 151.0]
lats = [37.79, -33.87]
locations = ela.xy_to_geoseries(lons, lats)
print(locations)

> 0    POINT (-122.49000 37.79000)
> 1    POINT (151.00000 -33.87000)
> dtype: geometry
```

---

## Drawing point location samples

In addition to species occurrence records (where `y = 1`), species distributions models often require a set of random pseudo-absence/background points (`y = 0`). These are a random geographic sampling of where you might expect to find a species across the target landscape.

**From a raster's extent**

You can use `elapid` to create a uniform random geographic sample from unmasked locations within a raster's extent:

```python
count = 10000 # the number of points to generate
pseudoabsence_points = ela.sample_raster(raster_path, count)
```

If you have a large raster that doesn't fit in memory, you can also set `ignore_mask=True` to use the rectangular bounds of the raster to draw samples.

```python
pseudoabsence_points = ela.sample_raster(raster_path, count, ignore_mask=True)
```

**From a vector polygon**

Species occurrence records are often collected with some bias (near roads, in protected areas, etc.), so we typically need to be more precise in where we select background points. One way to do this would be to draw samples from that species range map, which you can do with the `sample_vector()` function:

```python
range_path = "/home/slug/ariolimax-californicus-range.shp"
pseudoabsence_points = ela.sample_vector(range_path, count)
```

This currently computes the spatial union of all polygons within a vector, compressing the geometry into a single MultiPolygon object to sample.

If you've already computed a polygon using geopandas, you can pass it instead to `ela.sample_geoseries()`, which is what `sample_vector()` calls under the hood.

**From a bias raster**

You could also pass a raster bias file, where the raster grid cells contain information on the probability of sampling an area:

```python
# assume covariance between vertebrate and invertebrate banana slugs
bias_path = "/home/slug/proximity-to-uc-santa-cruz.tif"
pseudoabsence_points = ela.sample_bias_file(bias_path)
```

The grid cells can be an arbitrary range of values. What's important is that the values encode a linear range of numbers that are higher where you're more likely to draw a sample. The probability of drawing a sample is dependent on two factors: the range of values provided and the frequency of values across the dataset.

So, for a raster with values of `1` and `2`, you're sampling probability for raster locations of `2` is twice that as `1` locations. If these occur in equal frequency (i.e. half the data are `1` valuess, half are `2` values), then you'll likely sample twice as many areas with `2` values. But if the frequency of `1` values is much greater than `2` values, you'll shift the distribution. But you're still more likely, on a per-draw basis, to draw samples from `2` locations.

**A tip for background sampling**

[This paper](https://besjournals.onlinelibrary.wiley.com/doi/10.1111/j.2041-210X.2011.00172.x) by Morgane Barbet-Massin et al. is an excellent resource to consult when it comes to thinking about how background point samples should be selected.

---

## Stacking dataframes

`elapid` provides a convenience function for merging the geometries of two GeoSeries/GeoDataFrames. It's designed for stacking presence/background data.

It will reproject geometries on the fly if needed, and you can optionally add a 'class' column to the output GeoDataFrame.

Return a 1-column geodataframe with pseudoabsences concatenated to presence records:

```python
presence_points = gpd.read_file('/path/to/occurrence-records.gpkg')
point_stack = ela.stack_geometries(presence_points, pseudoabsence_points)
```

Return 2 columns, with class labels assigned (1 for presences, 0 for pseudoabsences):

```python
point_stack = ela.stack_geometries(
  presence_points,
  pseudoabsence_points,
  add_class_label=True,
)
```

If the geometries are in different crs, default is to reproject to the presence crs. Override this with target_crs="background":

```python
point_stack = ela.stack_geometries(
  presence_points,
  pseudoabsence_points,
  add_class_label=True,
  target_crs="background",
)
```

---

## Point annotation

Annotation refers to reading and storing raster values at the locations of a series of point occurrence records in a single `GeoDataFrame` table.

Once you have your species presence and pseudo-absence records, you can annotate these records with the covariate data from each location.

```python
covariates = ela.annotate(
    point_stack,
    list_of_rasters,
    drop_na = True,
)
```

This function, since it's geographically indexed, doesn't require the point data and the raster data to be in the same projection. `elapid` handles reprojection and resampling on the fly.

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

covariates = ela.annotate(
    point_stack,
    raster_paths,
    labels = labels
    drop_na = True,
)
```

If you don't specify the labels, `elapid` will assign `['b1', 'b2', ..., 'bn']` labels to each column.

Setting `drop_na=True` requires that the raster datasets have `nodata` values assigned in their metadata. These point locations will be dropped from the output dataframe, which will have fewer rows than the input points.

---

## Geographic sample weighting

Despite ingesting spatial data, many statistically-driven SDMs are not spatial models in a traditional sense: they don't handle geographic information during model selection or in model scoring.

One way to add spatial information to a model is to compute geographically-explicit sample weights.

`elapid` does this by calculating sample weights based on the distance to the nearest neighbor. Points nearby other points receive lower weight scores; far-away points receive higher weight scores.

```python
sample_weight = ela.distance_weights(point_stack)
```

The default is to compute weights based on the distance to the nearest point. You can instead compute the average distance to `n` nearest points instead to compute sample weights using point densities instead of the distance to the single nearest point. This may be useful if you have small clusters of a few points far away from large, densely populated regions.

```python
sample_weight = ela.distance_weights(point_stack, n_neighbors=10)
```

These weights can be passed to many many model fitting routines, typically via `model.fit(x, y, sample_weight=sample_weight)`. This is supported for `ela.MaxentModel()`, as well as many `sklearn` methods.

This function uses `ela.nearest_point_distance()`, a handy function for computing the distance between each point and it's nearest neighbor.

---

## Train/test splits

Uniformly random train/test splits are generally discouraged in spatial modeling because of the strong spatial structure inherent in many datasets. The non-independence of these data is referred to as spatial autocorrelation. Using distance- or density-based sample weights is one way to mitigate these effects. Another is to split the data into geographically distinct train/test regions to try and prioritize model generalization.

### Checkerboard splits

With a "checkerbox" train/test split, points are intersected along a regular grid, and every other grid is used to split the data into train/test sets.

```python
train, test = ela.checkerboard_split(point_stack, grid_size=1000)
```

The height and width of the grid used to split the data is controlled by the `grid_size` parameter. This should specify distance in the units of the point data's CRS. The above call would split data along a 1x1 km grid if the CRS units were in meters.

The black and white structure of the checkerboard means this method can only generate one train/test split.

### Geographic k-fold splits

Alternatively, you can create `k` geographically-clustered folds using the `GeographicKFold` cross validation strategy. This method is effective for understanding how well models fit in one region will generalize to areas outside the training domain.

```python
gfolds = ela.GeographicKFold(n_folds=4)
for train_idx, test_idx in gfolds.split(point_stack):
    train_points = point_stack.iloc[train_idx]
    test_points = point_stack.iloc[test_idx]
    # split x/y data, fit models, evaluate, etc.
```

This method uses KMeans clustering, fit with the x/y locations of the point data, to group points into spatially distinct clusters. This cross-validation strategy is a good way to test how well models generalize outside of their training extents into novel geographic regions.

### Buffered leave-one-out cross-validation

Leave-one-out cross-validation refers to training on *n-1* points and testing on a single test point, looping over the full dataset to test on each point separately.

The buffered leave-one-out strategy, as described by [Ploton et al. 2020](https://www.nature.com/articles/s41467-020-18321-y), modifies this approach. While each point is iterated over for testing, the pool of available training points is reduced at each iteration by dropping training points located within a certain distance of the test data. The purpose of this method is to evaluate how a model performs far away from where it was trained.

```python
bloo = ela.BufferedLeaveOneOut(distance=1000)
for train_idx, test_idx in bloo.split(point_stack):
    train_points = point_stack.iloc[train_idx]
    test_point = point_stack.iloc[test_idx]
    # split x/y data, fit models, evaluate, etc.
```

This function also modifies the "leave-one-out" approach to support testing on multiple points per-fold. You may want to run your cross-validation to evaluate test performance across multiple ecoregions, for example.

You can do this by passing the `group` keyword during train/test splitting, and the value should correspond to a column name in the GeoDataFrame you pass.

```python
bloo = ela.BufferedLeaveOneOut(distance=1000)
for train_idx, test_idx in bloo.split(point_stack, group="Ecoregion"):
    train_points = point_stack.iloc[train_idx]
    test_point = point_stack.iloc[test_idx]
    # assumes `point_stack` has an 'Ecoregion' column
```

This will find the unique values in the `Ecoregion` column and iterate over those values, using the points from each ecoregion as test folds and excluding points within 1000m of those locations from the training data.

---

## Zonal statistics

In addition to the tools for working with Point data, `elapid` contains a routine for calculating zonal statistics from Polygon or MutliPolygon geometry types.

This routine reads an array of raster data covering the extent of a polygon, masks the areas outside the polygon, and computes summary statistics such as the mean, standard deviation, and mode of the array.

```python
ecoregions = gpd.read_file('/path/to/california-ecoregions.shp')
zs = ela.zonal_stats(
    ecoregions,
    raster_paths,
    labels = labels,
    mean = True,
    stdv = True,
    percentiles = [10, 90],
)
```

The stats reported are managed by a set of keywords (`count=True`, `sum=True`, `skew=True`). The `all=True` keyword is a shortcut to compute all of the stats. You'll still have to explicitly pass a list of `percentiles`, however, like this:

```python
zs = ela.zonal_stats(
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
- handles multi-raster input, reading inputs in serial but creating a single output GeoDataFrame.

---

## Applying predictions to data

Once you've fit a model (we're skipping a step here, but see the [Maxent overview](../sdm/maxent.md) for an example), you can apply it to a set of raster covariates to produce gridded habitat suitability maps.

```python
ela.apply_model_to_rasters(
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

x, y = ela.load_sample_data()

model = RandomForestClassifier()
model.fit(x, y)

input_rasters = ['/path/to/raster1.tif', '/path/to/raster2.tif']
output_raster = 'rf-model-prediction-categorical.tif'
ela.apply_model_to_rasters(
  model,
  input_rasters,
  output_raster,
)
```

These models contain an additional method for estimating the continuous prediction probabilities. To write these out, set `predict_proba=True`. To use this option, however, you also have to specify the number of output raster bands. Since the sample data is a 2-class model, the output prediction probabilities are 2-band outputs, so we set `count=2`.

```python
output_raster = 'rf-model-prediction-probabilities.tif'
ela.apply_model_to_rasters(
  model,
  input_rasters,
  output_raster,
  predict_proba = True,
  count = 2,
)
```
