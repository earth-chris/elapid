# elapid

<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

`elapid` provides python support for species distribution modeling. This includes a custom [MaxEnt][home-maxent] implementation and general spatial processing tools. It will soon include tools for working with [GBIF][home-gbif]-format datasets.

The name is an homage to *A Biogeographic Analysis of Australian Elapid Snakes* (H.A. Nix, 1986), the paper widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

The maxent modeling tools and feature transformations are translations of the R `maxnet` [package][r-maxnet]. It uses the `glmnet` [python bindings][glmnet], and is implemented using `sklearn` conventions.

### Table of Contents

- [Background](#background)
- [Installation](#installation)
- [Working with elapid](#working-with-elapid)
- [Geospatial support](#geospatial-support)
  - [Working with x-y data](#working-with-x-y-data)
  - [Generating pseudo-absence records](#generating-pseudo-absence-records)
  - [Extracting raster values](#extracting-raster-values)
  - [Applying models to rasters](#applying-models-to-rasters)
- [Package design](#package-design)
- [Contact](#contact)


## Background

Maxent is a species distribution modeling (SDM) system, which uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

Its a presence/background model, meaning it uses data on where a species is present and, instead of data on where it is absent, a random sample of the region where you might expect to find that species. This is convenient because it reduces data collection burdens (absence data isn't required) but it also introduces some unique challenges for fitting and interpreting models.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using species occurrence records (`y = 1`), randomly sampled "background" location records (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing pairwise products between covariates, setting random covariate thresholds). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as:

```
Pr(y = 1 | f(z)) = (f1(z) * Pr(y = 1)) / f(z)
```

`elapid` provides python tools for fitting maxent models, computing features, and working with geospatial data. It's goal is to help us better understand where the species are, why they are there, and where they will go in the future.

## Installation

This library can be installed via `pip`.

```bash
pip install elapid
```

You can also clone the source repository and install it locally.

```bash
git clone https://github.com/earth-chris/elapid.git
cd elapid
pip install -e .
```

### With conda

You can use `conda` to ensure you have all the required dependencies (`glmnet` and `rasterio` have some library dependencies). From the repository's base directory:

```bash
conda env update
conda activate elapid
pip install -e .
```

This will create an environment named `elapid` and install an editable version of the package that you can use.

## Working with elapid

There are two primary Maxent functions: fitting features and fitting models. You can do it all at once with:

```python
import elapid

x, y = elapid.load_sample_data()
model = elapid.MaxentModel()
model.fit(x, y)
```

Where:

- `x` is an array or dataframe of environmental covariates of shape (`n_samples`, `n_covariates`)
- `y` is an array or series of species presence/background labels (rows labeled `1` or `0`)

The `elapid.MaxentModel()` object takes these data, fits features from covariate data, computes sample weights and feature regularization, fits a series of models, and returns an estimator that can be used for applying predictions to new data.

`MaxentModel()` behaves like an sklearn `estimator` class. Use `model.fit(x, y)` to train a model, and `model.predict(x)` to generate model predictions.

You can also generate and evaluate features before passing them to the model:

```python
features = elapid.MaxentFeatureTransformer()
z = features.fit_transform(x)
model.fit(z, y, is_features=True)
```

`MaxentFeatureTransformer()` behaves like an sklearn `preprocessing` class. Use `features.fit(x_train)` to fit features, `features.transform(x_test)` to apply to new covariate data, or `features.fit_transform(x)` to fit features and return the transformed data.

The base Maxent classes can be modified with parameters that are available in other Maxent implementations:

```python
model = elapid.MaxentModel(
    feature_types = ['linear', 'hinge', 'product'], # the feature transformations
    tau = 0.5, # prevalence scaler
    clamp = True, # set covariate min/max based on range of training data
    scorer = 'roc_auc', # metric to optimize (from sklearn.metrics.SCORERS)
    beta_multiplier = 1.0, # regularization scaler (high values drop more features)
    beta_lqp = 1.0, # linear, quadratic, product regularization scaler
    beta_hinge = 1.0, # hinge regularization scaler
    beta_threshold = 1.0, # threshold regularization scaler
    beta_categorical = 1.0, # categorical regularization scaler
    n_hinge_features = 50, # number of hinge features to compute
    n_threshold_features = 50, # number of threshold features to compute
    convergence_tolerance = 1e-07, # model fit convergence threshold
    use_lambdas = 'last', # set to 'best' (least overfit), 'last' (highest score)
    n_cpus = 4, # number of cpu cores to use
)
```


## Geospatial support

In addition to the maxent modeling support tools, `elapid` includes a series of geospatial data processing routines. These should make it easy to work with species occurrence records and raster covariates in multiple formats. The workflows rely on `geopandas` for vectors and `rasterio` for rasters.

Here's an example end-to-end workflow (using dummy paths to demonstrate functionality).

```python
import elapid

vector_path = "/home/slug/ariolimax-californicus.shp"
raster_path = "/home/slug/california-climate-veg.tif"
output_path = "/home/slug/ariolimax-californicus-habitat.tif"
model_path = "/home/slug/ariolimax-claifornicus-model.ela"

# sample the raster values at point locations
presence = elapid.raster_values_from_vector(vector_path, raster_path)
pseudoabsence_points = elapid.pseudoabsence_from_raster(raster_path)
pseudoabsence = elapid.raster_values_from_geoseries(pseudoabsence_points, raster_path)

# merge the datasets into one dataframe
pseudoabsence['presence'] = 0
presence['presence'] = 1
y = presence['presence'].append(pseudoabsence['presence']).reset_index(drop=True)
x = presence.drop(['presence'], axis=1).append(pseudoabsence.drop(['presence'], axis=1)).reset_index(drop=True)

# train the model
model = elapid.MaxentModel()
model.fit(x, y)

# apply it to the full extent and save the model for later
elapid.apply_model_to_rasters(model, raster_path, output_path, transform="logistic")
elapid.save_object(model, model_path)
```

To work with this saved model later, you can run:

```python
model = elapid.load_object(model_path)
```

Let's dig in to the components of this workflow.

### Working with x-y data

Almost all of the data sampling and indexing uses `geopandas.GeoSeries` objects. These are the format of the `geometry` column for a `GeoDataFrame`.

```python
import geopandas as gpd

gdf = gpd.read_file(vector_path)
print(type(gdf.geometry))

> <class 'geopandas.geoseries.GeoSeries'>
```

Sometimes you don't have a vector of point-format location data. The `java` implementation of maxent uses csv files, for example. You can work with those using the `xy_to_geoseries` function:

```python
import pandas as pd

csv_path = "/home/cba/ariolimax-californicus.csv"
df = pd.read_csv(csv_path)
presence = elapid.xy_to_geoseries(df.x, df.y, crs="EPSG:32610")
```

Make sure you specify the projection of your x/y data. The default assumption is lat/lon, which in many cases is not correct.

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

### Generating pseudo-absence records

In addition to species occurrence records, maxent requires a set of pseudo-absence (i.e. background) points. These are a random geographic samping of where you might expect to find a species.

You can use `elapid` to create a random geographic sampling of points from unmasked locations within a raster's extent:

```python
count = 10000 # the number of points to generate
pseudoabsence_points = elapid.pseudoabsence_from_raster(raster_path, count)
```

Species occurrence records are often biased in their collection (collected near roads, in protected areas, etc.), so we typically need to be more precise in where we select pseudo-absence points. You could use a vector with a species range map to select records:

```python
range_path = "/home/slug/ariolimax-californicus-range.shp"
pseudoabsence_points = elapid.pseudoabsence_from_vector(range_path, count)
```

If you've already computed a polygon using geopandas, you can pass it instead to `elapid.pseudoabsence_from_geoseries()`, which is what `pseudoabsence_from_vector()` does under the hood.

You could also pass a raster bias file, where the raster grid cells contain information on the probability of sampling an area:

```python
# assume covariance between vertebrate and invertebrate banana slugs
bias_path = "/home/slug/proximity-to-ucsc.tif"
pseudoabsence_points = pseudoabsence_from_bias_file(bias_path)
```

The grid cells can be an arbitrary range of values. What's important is that the values encode a linear range of numbers that are higher where you're more likely to draw a sample. The probability of drawing a sample is dependent on two factors: the range of values provided and the frequency of values across the dataset.

So, for a raster with values of `1` and `2`, you're sampling probability for raster locations of `2` is twice that as `1` locations. If these occur in equal frequency (i.e. half the data are `1` valuess, half are `2` values), then you'll likely sample twice as many areas with `2` values. But if the frequency of `1` values is much greater than `2` values, you'll shift the distribution. But you're still more likely, on a per-draw basis, to draw samples from `2` locations.

The above example prioritizes sampling frequency in the areas around UC Santa Cruz, home to all types of slug, based on the distance to the campus.

### Extracting raster values

Once you have your species presence and pseudo-absence records, you can extract the covariate data from each location.

```python
pseudoabsence_covariates = elapid.raster_values_from_geoseries(
    pseudoabsence_points,
    raster_path,
    drop_na = True,
)
```

This could also be done with `raster_values_from_vector(vector_path, raster_path)` if you haven't already loaded the geoseries data into memory.

This function, since it's geographically indexed, doesn't require the point data and the raster data to be in the same projection. `elapid` handles reprojection and sampling on the fly.

It also allows you to pass multiple raster files, which can be in different projections, extents, grid sizes, etc. This means you don't have to explicitly re-sample your raster data prior to analysis, which is always a chore.

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

### Applying models to rasters

Once you've fit a model, you can apply it to a set of raster covariates to produce gridded suitability maps.

```python
elapid.apply_model_to_rasters(
    model,
    raster_paths,
    output_path,
    template_idx = 0,
    transform = "cloglog",
    nodata=-9999,
)
```

The list and band order of the rasters must match the order of the covariates used to train the model. It reads each dataset in a block-wise basis, applies the model, and writes gridded predictions.

If the raster datasets are not consistent (different extents, resolutions, etc.), it wll re-project the data on the fly, with the grid size, extent and projection based on a 'template' raster. Use the `template_idx` keyword to specify the index of which raster file to use as the template (`template_idx=0` sets the first raster as the template).

In the example above, it's important to set the template to the `california-leaf-area-index.tif` file. This is because this is the smallest extent with data, and it'll only read and apply the model to the `usa` and `global` datasets in the area covered by `california`. If you were to set the extent to `usa-mean-temperature.tif`, it would still technically function, but there would be a large region of `nodata` values where there's insufficient covariate coverage.


## Package design

This python package was developed to overcome some of the limitations of the Maxent [java][home-maxent] package. Some gripes with the `java` distribution include:

- It requires using specific and inconvenient raster data formats (`mxe`, `grd`, or `bil` raster data), which create large files that take a long time to read/write.
- There's no support for working with common point occurence formats, like shapefiles or geojson.
- Its hard to include multi-temporal environmental covariate data, meaning time series data need to be aggregated (e.g., averaged) in some way prior to analysis.
- Applying a previously-trained model to new data (e.g., under climate projections) is difficult.
- Training and applying batches of models is laborious.

`elapid` was designed to support common geospatial data formats and contemporary approaches to training statistical models. It uses `sklearn` conventions to fit and apply models, `rasterio` to handle raster operations, `geopandas` for vector operations, and returns `pandas` dataframes.

This places some burden on the user to handle processes like creating train/test cross-validation splits and sampling background data. The hope is that this extra flexibility removes some of the frustrations with the black-box approach of the java implementation and enables users to better tune and evaluate their models.

There's also more flexibility supporting contemporary geospatial data formats, like cloud-optimized geotiffs. This means you can train and apply models using cloud-hosted covariate data (e.g., from `s3`) without having to download the data to a local hard-drive.

## Contact

* Christopher Anderson [[web][home-cba]] [[email][email-cba]] [[github][github-cba]].


[email-cba]: mailto:cbanders@stanford.edu
[github-cba]: https://github.com/earth-chris
[home-cba]: https://earth-chris.github.io
[home-conda]: https://docs.conda.io/
[home-gbif]: https://gbif.org
[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
[r-maxnet]: https://github.com/mrmaxent/maxnet
[glmnet]: https://github.com/civisanalytics/python-glmnet/
