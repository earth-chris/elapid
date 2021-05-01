# Working with elapid

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


[home-maxent]: https://biodiversityinformatics.amnh.org/open_source/maxent/
