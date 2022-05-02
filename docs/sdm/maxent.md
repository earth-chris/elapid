# Maxent SDM

Maxent is a species distribution modeling (SDM) system, which uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

Its a presence/background model, meaning it uses data on where a species is present and, instead of data on where it is absent, a random sample of the region where you might expect to find that species. This is convenient because it reduces data collection burdens (absence data isn't required) but it also introduces some unique challenges for fitting and interpreting models.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using species occurrence records (`y = 1`), randomly sampled "background" location records (`y = 0`), and environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons (`z`) to the covariate data (e.g. computing pairwise products between covariates, setting random covariate thresholds). Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as:

```
Pr(y = 1 | f(z)) = (f1(z) * Pr(y = 1)) / f(z)
```

`elapid` provides python tools for fitting Maxent models, computing features, and applying models to raster data. Below are some instructions for the first two; the latter is reviewed [here](../examples/geo.md#applying-predictions-to-data).

---

## Maxent in elapid

### Training models

There are two primary Maxent functions: fitting features and fitting models. You can do it all at once with:

```python
import elapid

x, y = elapid.load_sample_data()
model = elapid.MaxentModel()
model.fit(x, y)
```

Where:

- `x` is an ndarray or dataframe of environmental covariates of shape (`n_samples`, `n_covariates`)
- `y` is an ndarray or series of species presence/background labels (rows labeled `1` or `0`)

The `elapid.MaxentModel()` object takes these data, fits features from covariate data, computes sample weights and feature regularization, fits a series of models, and returns an estimator that can be used for applying predictions to new data.

`MaxentModel()` behaves like an sklearn `estimator` class. Use `model.fit(x, y)` to train a model, and `model.predict(x)` to generate model predictions.

### Feature transformations

You can also generate and evaluate features before passing them to the model:

```python
features = elapid.MaxentFeatureTransformer()
z = features.fit_transform(x)
model.fit(z, y, is_features=True)
```

`MaxentFeatureTransformer()` behaves like an sklearn `preprocessing` class. Use `features.fit(x_train)` to fit features, `features.transform(x_test)` to apply to new covariate data, or `features.fit_transform(x)` to fit features and return the transformed data.

Setting the `is_features=True` flag is important here because, by default, the `MaxentModel()` class will automatically fit and apply feature tranformations

### Configuration

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
    n_hinge_features = 10, # number of hinge features to compute
    n_threshold_features = 10, # number of threshold features to compute
    convergence_tolerance = 1e-07, # model fit convergence threshold
    use_lambdas = 'best', # set to 'best' (least overfit), 'last' (highest score)
    n_cpus = 4, # number of cpu cores to use
)
```

You can find the default configuration parameters in [elapid/config](https://github.com/earth-chris/elapid/blob/main/elapid/config.py).

---

## Additional reading

A practical guide to MaxEnt for modeling species’ distributions: what it does, and why inputs and settings matter, Merow et al. 2013 [[pdf](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1600-0587.2013.07872.x)]

A statistical explanation of MaxEnt for ecologists, Elith et al. 2011 [[pdf](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1472-4642.2010.00725.x)]

Opening the black box: an open-source release of Maxent, Phillips et al. 2017 [[pdf](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.03049)]

Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation, Philliips et al. 2008 [[pdf](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.0906-7590.2008.5203.x)]
