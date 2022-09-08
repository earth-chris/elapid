# Maxent SDM

## Background

Maxent is a species distribution modeling (SDM) system that uses species observations and environmental data to predict where a species might be found under past, present or future environmental conditions.

It's a presence/background model. Maxent uses data on where a species is present and, instead of data on where it is absent, it uses a random sample of the landscape where you might expect to find that species. This is convenient because it reduces data collection requirements, but it also introduces some unique challenges for fitting and interpreting models.

Formally, Maxent estimates habitat suitability (i.e. the fundamental niche) using:

1. species occurrence records (`y = 1`),
1. randomly sampled "background" location records (`y = 0`), and
1. environmental covariate data (`x`) for each location of `y`.

Maxent doesn't directly estimate relationships between presence/background data and environmental covariates (so, not just `y ~ x`). Instead, it fits a series of feature transformatons to the covariate data (`z = f(x)`), like computing quadratic transformations or computing the pairwise products of each covariate. Maxent then estimates the conditional probability of finding a species given a set of environmental conditions as:

```
Pr(y = 1 | f(z)) = (f_1(z) * Pr(y = 1)) / f(z)
```

This can be interpreted as: "the relative likelihood of observing a species at any point on the landscape is determined by differences in the distributions of environmental conditions where a species is found relative to the distributions at a random sample across the landscape."

A null model would posit that a species is likely to be distributed uniformly across a landscape; you're equally likely to find it wherever you go. Maxent uses this null model—the "background" conditions—as a prior probability distribution to estimate relative suitability based on the conditions where species have been observed.

Because of this formulation, the definition of both the landscape and the background are directly related to the definition of how habitat suitability is estimated. These should be defined with care.

`elapid` provides python tools for fitting Maxent models, computing features, and applying models to raster data. Below are some instructions for the first two; the latter is reviewed [here](../examples/geo.md#applying-predictions-to-data).

---

## Using Maxent in `elapid`

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
    beta_multiplier = 1.5, # regularization scaler (high values drop more features)
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

## Differences between `elapid` and `maxnet`

`elapid` was written to match a series of routines defined in [maxnet](https://github.com/mrmaxent/maxnet), the R version of Maxent. The goal was to provide a near-equivalent Python implementation.

It has been tested to verify numerical equivalence when computing the custom parameters used to fit penalized logistic regression models with `glmnet`. This includes [sample weights](https://github.com/mrmaxent/maxnet/blob/534614d0a8f326c31bbf9aa2b5263081bb510263/R/maxnet.R#L18), [lambda scores](https://github.com/mrmaxent/maxnet/blob/534614d0a8f326c31bbf9aa2b5263081bb510263/R/maxnet.R#L20), [regularization scores](https://github.com/mrmaxent/maxnet/blob/master/R/maxnet.default.regularization.R), and [feature transformations](https://github.com/mrmaxent/maxnet/blob/master/R/maxnet.formula.R).

Yet there are some differences between models fit using `elapid` and `maxnet`. These differences are determined by opinionated changes to default model parameters. Before we enumerate these differences, let's quickly review an important topic: feature regularization.

Regularization parameters—or beta values—are estimated on a feature-by-feature basis, and determine the likelihood that a feature is identified by the model as meaningful. Regularization is often used to reduce overfitting by dropping meaningless or noisy features. Increasing regularization values typically increases feature penalties and increases model generality. Several `elapid` defaults and feature transformation routines modify how regularization is handled compared to `maxnet`.

### Opinionated default model parameters

`beta_multiplier: 1.5` - users have control over how regularization is applied to specific feature types (e.g. `beta_hinge`, `beta_categorical`) and across all feature types (via the `beta_multiplier`).

- `elapid` increases the default `beta_multipier` to 1.5, compared to 1.0 in `maxnet`, in order to increase regularization penalties.
- This was selected after some quantitative analyses—finding a value that minimized cross-validation error in multiple SDM contexts—and after inspecting partial dependence plots for ecological coherence.
- Increasing regularization across the board almost always leads to smoother model predictions and lower out-of-bag error rates, indicating highly penalized models generalize well. This seemed like a sensible, conservative value.


`use_lambdas: 'best'` - the underlying regularized logistic regression model fits a series of models using an array of lambda values, which modify the scale of regularization applied (the number of fits is determined by the `n_lambdas` parameter).

- `maxnet` uses the 'last' lambda value, which typically applies the lowest regularization penalties.
- `elapid` uses the 'best' lambda value, which is the lambda value that minimizes the cross-validation error estimated during model fitting.

`feature_types: ["linear", "hinge", "product"]` - `elapid` only estimates these three feature types by default to reduce complexity and to prioritize ecologically realistic models.

- linear features capture the direct effects of each environmental covariate on species occurrence patterns.
- hinge features provide context for each covariate's local value relative to the global range of that covariate's values.
- product features allow fitting interaction terms between covariates.
- quadratic features rarely provide any additional information beyond linear features, and are dropped by default.
- threshold features—which quantify stepwise functions—are uncommon in biogeography, and are regularly cited as a source of overfitting. These features tend to be redundant with hinge features, which are like smoothed threshold features across the range of each covariate, making threshold features redundant.

`n_hinge_features: 10` - `elapid` computes fewer hinge features by default than `maxnet` (50), which seems sufficiently complex.

- The number of hinge features effectively determines the number of bins that each covariate gets expanded into, computing hinges or "knots" at each bin.
- Increasing `n_hinge_features` will partition up the data into finer and finer bins, but I've found that 10-15 features captures enough detail.


### Feature regularization vs. normalization

One of the primary differences between `elapid` and `maxnet` is how feature normalization is handled, which has downstream effects on estimated regularization values.

Normalization is the process of rescaling covariate values from their native range to a standardized range (e.g. converting mean annual temperatures and precipitation values from degrees C and mm/year to scaled 0.0-1.0 values). Normalization reduces the likelihood that model fits are overly sensitive to the range of observed covariates.

By default, `elapid` normalizes all features to a 0-1 range prior to model fitting.

`maxnet` does not. Instead, it scales regularization penalties based on the range and variance of each covariate, increasing regularization scores for covariates with high numerical ranges (like population density, which could scale from 0-6,000 people per km2 in Tokyo) and decreasing regularization scores for covariates with low numerical ranges (like humidity, which ranges from 0.002 to 0.007 kg/kg in California).

I don't know that there's a "right" way to handle this.

By applying feature normalization, regularization values are consistent across each feature type in `elapid`. That is, all linear features receive uniform regularization penalties, and all hinge features receive uniform regularization penalties, though there are differences between linear and hinge regularization values (hinge features are more penalized).

`maxnet`, by contrast, uses regularization parameters in order to impose normalization constraints. But, due to the internal mechanics by which this is implemented, I've found that there can be some nonlinear effects, and that this approach is less consistent—and the regularization values less interpretable—than when using a uniform normalization and regularization strategy.

There's still a high level of agreement between methods: comparing `elapid` and `maxnet` predictions to the same sample dataset computed an r2 score of 0.91 and a mean absolute error score of 0.047 (using the 0-1 scaled cloglog output). But the differences in model predictions arise in large part due to differences in how feature regularization is calculated.

`elapid` is opinionated by design, uniformly handling normalization and regularization to make it easier to understand model performance and to minimize any effects introduced by differences in covariate ranges.

Ecological and environmental data span many scales, species niche preferences are complex, and small perturbations in environmental conditions can have large effects on estimated habitat suitability. Statistically-driven SDMs should be less sensitive to patterns like the absolute range of conditions and more sensitive to relative niche space that species occupy within that range, which we don't know *a priori*.

---

## Additional reading

Below are links to some useful journal articles describing how Maxent works.

A practical guide to MaxEnt for modeling species’ distributions: what it does, and why inputs and settings matter, Merow et al. 2013 [[pdf](https://onlinelibrary.wiley.com/doi/pdf/10.1111/j.1600-0587.2013.07872.x)]

A statistical explanation of MaxEnt for ecologists, Elith et al. 2011 [[pdf](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/j.1472-4642.2010.00725.x)]

Opening the black box: an open-source release of Maxent, Phillips et al. 2017 [[pdf](https://onlinelibrary.wiley.com/doi/pdfdirect/10.1111/ecog.03049)]

Modeling of species distributions with Maxent: new extensions and a comprehensive evaluation, Philliips et al. 2008 [[pdf](https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.0906-7590.2008.5203.x)]
