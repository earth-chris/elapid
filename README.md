<img src="http://earth-chris.github.io/images/design/amazon.jpg" alt="the amazon"/>

# elapid

Species distribution modeling tools in python. This includes a custom [MaxEnt][home-maxent] implementation and general spatial processing tools. It will soon include tools for working with [GBIF][home-gbif]-format datasets.

The name was chosen as homage to the paper by H.A. Nix, *A Biogeographic Analysis of Australian Elapid Snakes* (1986), which is widely credited with defining the essential bioclimatic variables to use in species distribution modeling.It's also a snake pun (a python wrapper for mapping snake biogeography).

This package is a python translation of the R `maxnet` [package][r-maxnet], using `glmnet` [python bindings][glmnet], implemented using `sklearn` conventions.

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

## Maxent modeling fundamentals

Maxent is a species distribution modeling tool that estimates habitat suitability (i.e. the fundamental niche) using a series of species occurrence records (`y = 1`), randomly sampled "background" locations (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y~x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing the products between each feature, random covariate thresholds, etc.). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as `Pr(y = 1 | f(z))`.

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
