# elapid

<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

`elapid` provides python support for species distribution modeling. This includes a custom [MaxEnt][home-maxent] implementation and general spatial processing tools. It will soon include tools for working with [GBIF][home-gbif]-format datasets.

The name was chosen as homage to the paper, *A Biogeographic Analysis of Australian Elapid Snakes* (H.A. Nix, 1986), which is widely credited with defining the essential bioclimatic variables to use in species distribution modeling. It's also a snake pun (a python wrapper for mapping snake biogeography).

The maxent modeling tools and feature transformations are translations of the R `maxnet` [package][r-maxnet]. It uses the `glmnet` [python bindings][glmnet], as is implemented using `sklearn` conventions.

### Table of Contents

- [Installation](#installation)
- [Working with elapid](#working-with-elapid)
- [Geospatial support](#geospatial-support)
  - [Working with x-y data](#working-with-x-y-data)
  - [Generating pseudo-absence records](#generating-pseudo-absence-records)
  - [Extracting raster values](#extracting-raster-values)
  - [Applying models to rasters](#applying-models-to-rasters)
- [Background](#background)
- [Contact](#contact)


## Installation

This library can be installed via `pip`.

```bash
pip install elapid
```

You can also clone the source repository and install it locally.

```bash
git clone https://github.com/earth-chris/elapid.git
cd elapid
pip install -e . -r requirements.txt
```

### conda

You can use `conda` to ensure you have all the required dependencies for the build. This should reduce the headache of housing your own `gdal` build. From the repository's base directory:

```bash
conda env update
conda activate elapid
pip install -e .
```

This will create an environment named `elapid` and install an editable version of the package that you can use.

## Working with elapid

There are two primary Maxent functions: fitting features and fitting models. You can do it all in one go with:

```python
import elapid

x, y = elapid.load_sample_data()
m = elapid.MaxentModel()
m.fit(x, y)
```

Where:

- `x` is an array or dataframe of environmental covariates (`nrows`, `ncols`)
- `y` is a series of species presence/background labels (1/0)

The `elapid.MaxentModel()` object takes these data, fits features based on the covariate data, computes weights and variable regularization, fits a series of models, and returns an object that can be used for applying predictions to new data (`m.predict(x_test)`, for example).

You can also generate and evaluate features then pass them to the model object:

```python
features = elapid.MaxentFeatureTransformer()
z = features.fit_transform(x)
m.fit(z, y, is_features=True)
```

The `MaxentModel` and `MaxentFeatureTransformer` classes can be modified with parameters that are available in other Maxent implementations (e.g., `feature_types=['linear', 'quadratic', 'hinge'`]).


## Geospatial support

In addition to the maxent modeling support tools, `elapid` includes a series of geospatial data processing routines. These should make it easy to work with species occurrence records and raster covariates in multiple formats. The workflows rely on `geopandas` for vector support and `rasterio` for raster support.

These tools allow you to run and apply a maxent model using just species occurrence records (point format vector) and environmental covariates (raster(s)). Here's an example end-to-end workflow:

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
model = elapid.MaxentModel(feature_types=["linear", "product", "hinge"])
model.fit(x, y)

# apply it to the full extent and save the model for later
elapid.apply_model_to_rasters(model, raster_path, output_path, transform="logistic")
elapid.save_object(model, model_path)
```

### Working with x-y data

Almost all of the data sampling and indexing uses `geopandas.GeoSeries` objects. These are the format of the `geometry` column for a `GeoDataFrame`.

```python
import geopandas as gpd

vector_path = "/home/cba/ariolimax-californicus.shp"
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
print(presence.head())

>
```

Make sure you specify the projection of your x/y data. The default assumption is lat/lon, which in many cases is not correct.

You can also convert arbitrary arrays of x/y data:

```python
lons = [-122.49, 151.0]
lats = [37.79, -33.87]
locations = elapid.xy_to_geoseries(lons, lats)
print(locations)

>
```

### Generating pseudo-absence records

`pseudoabsence_from_bias_file()`
`pseudoabsence_from_geoseries()`
`pseudoabsence_from_raster()`
`pseudoabsence_from_vector()`

### Extracting raster values

`raster_values_from_geoseries()`
`raster_values_from_vector()`

### Applying models to rasters

`elapid.apply_model_to_rasters()`

## Background

Maxent is a species distribution modeling (SDM) approach, which uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

Its a presence/background model, meaning it uses data on where a species is present and, instead of data on where it is absent, a random sample of the region where you might expect to find that species. This is convenient because it reduces data collection burdens (absence data isn't required) but it also introduces some unique challenges for interpreting model outputs.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using species occurrence records (`y = 1`), randomly sampled "background" location records (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing the products between each feature, random covariate thresholds, etc.). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as `Pr(y = 1 | f(z))`.

### Motivation

This python package was developed to overcome some of the limitations of the Maxent [java][home-maxent] package. Some gripes with the `java` distribution include:

- It requires using specific and inconvenient raster data formats (`mxe`, `grd`, or `bil` raster data), which create large files that take a long time to read/write.
- There's no support for working with common point occurence formats, like shapefiles or geojson.
- Its hard to include multi-temporal environmental covariate data, meaning time series data need to be aggregated (e.g., averaged) in some way prior to analysis.
- Applying a previously-trained model to new data (e.g., under climate projections) is difficult.
- Training and applying batches of models is laborious.

The `elapid` package was designed to support common geospatial data formats and contemporary approaches to training statistical models. It uses `sklearn` conventions to fit and apply models, `rasterio` to handle raster operations, `geopandas` for vector operations, and returns `pandas` dataframes.

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
