# elapid

<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

Species distribution modeling tools in python. This includes a custom [MaxEnt][home-maxent] implementation and general spatial processing tools. It will soon include tools for working with [GBIF][home-gbif]-format datasets.

The name was chosen as homage to the paper by H.A. Nix, *A Biogeographic Analysis of Australian Elapid Snakes* (1986), which is widely credited with defining the essential bioclimatic variables to use in species distribution modeling.It's also a snake pun (a python wrapper for mapping snake biogeography).

This package is a python translation of the R `maxnet` [package][r-maxnet], using `glmnet` [python bindings][glmnet], implemented using `sklearn` conventions.

## Background

Maxent is a species distribution modeling (SDM) approach, which uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

Maxent is a presence/background SDM. This means it uses data on where a species is present and, instead of data on where it is absent, a random sample of the region where you might expect to find that species. This is convenient because it reduces data collection burdens (absence data is not required) but it also introduces some unique challenges for interpreting model outputs.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using species occurrence records (`y = 1`), randomly sampled "background" location records (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing the products between each feature, random covariate thresholds, etc.). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as `Pr(y = 1 | f(z))`.

### Motivation

This python package was developed to overcome some of the limitations of the Maxent [java][home-maxent] package. Some gripes with the `java` distribution include:

- It requires using specific and inconvenient raster data formats (`mxe`, `grd`, or `bil` raster data), which create large files that take a long time to read/write.
- There's no support for working with common point occurence formats, like shapefiles or geojson.
- It is very difficult to include multi-temporal environmental covariate data, meaning that data need to be aggregated (e.g., averaged) in some way prior to analysis.
- Applying a trained model to new data (e.g., under climate projections) is also very difficult.
- Training and applying batches of models is labor intensive.

The `elapid` package was designed to support common geospatial data formats and contemporary approaches to training statistical models.

It uses `sklearn` conventions to fit and apply models, `rasterio` to handle raster operations, `geopandas` for vector operations, and returns `pandas` dataframes.

This places some additional burden on the user to handle processes like creating train/test splits for cross-validation, and on running commands to sample background data. The hope is that this extra flexibility removes some of the frustrations with the black-box approach of the java implementation and enables users to better tune and evaluate their models.

There's also more flexibility supporting contemporary geospatial data formats, like cloud-optimized geotiffs. This means you can train and apply models using cloud-hosted covariate data (e.g., from `s3`) without having to download the data to a local hard-drive.

## Installation

It's probably best to use `conda` for now. From the repository's base directory:

```bash
conda env update
conda activate elapid
pip install -e .
```

This will create an environment named `elapid` and install an editable version of the package that you can use.

## Getting started

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
