"""Functions to transform covariate data into complex model features."""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer

from elapid.config import MaxentConfig
from elapid.types import ArrayLike, validate_boolean, validate_feature_types, validate_numeric_scalar
from elapid.utils import make_band_labels, repeat_array


class LinearTransformer(MinMaxScaler):
    """Applies linear feature transformations to rescale features from 0-1."""

    clamp: bool = False
    feature_range: Tuple[float, float] = (0.0, 1.0)

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        super().__init__(clip=clamp, feature_range=feature_range)


class QuadraticTransformer(BaseEstimator):
    """Applies quadtratic feature transformations and rescales features from 0-1."""

    clamp: bool = None
    feature_range: Tuple[float, float] = None
    estimator: BaseEstimator

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)

    def fit(self, x: ArrayLike) -> None:
        self.estimator.fit(np.array(x) ** 2)

    def transform(self, x: ArrayLike) -> np.array:
        return self.estimator.transform(np.array(x) ** 2)

    def fit_transform(self, x: ArrayLike) -> np.array:
        self.fit(x)
        return self.estimator.transform(np.array(x) ** 2)

    def inverse_transform(self, x: ArrayLike) -> np.array:
        return self.estimator.inverse_transform(np.array(x)) ** 0.5


class ProductTransformer(BaseEstimator):
    """Computes the column-wise product of an array of input features, rescaling from 0-1."""

    clamp: bool = None
    feature_range: Tuple[float, float] = None
    estimator: BaseEstimator

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)

    def fit(self, x: ArrayLike):
        self.estimator.fit(column_product(np.array(x)))

    def transform(self, x: ArrayLike) -> np.array:
        return self.estimator.transform(column_product(np.array(x)))

    def fit_transform(self, x: ArrayLike) -> np.array:
        self.fit(x)
        return self.transform(x)


class ThresholdTransformer(BaseEstimator):
    """Applies binary thresholds to each covariate based on n evenly-spaced
    thresholds across it's min/max range."""

    n_thresholds_: int = None
    mins_: np.array = None
    maxs_: np.array = None
    threshold_indices_: np.array = None

    def __init__(self, n_thresholds: int = MaxentConfig.n_threshold_features):
        self.n_thresholds_ = n_thresholds

    def fit(self, x: ArrayLike):
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.threshold_indices_ = np.linspace(self.mins_, self.maxs_, self.n_thresholds_)

    def transform(self, x: ArrayLike) -> np.array:
        x = np.array(x)
        xarr = repeat_array(x, len(self.threshold_indices_), axis=-1)
        tarr = repeat_array(self.threshold_indices_.transpose(), len(x), axis=0)
        thresh = (xarr > tarr).reshape(x.shape[0], -1)
        return thresh.astype(np.uint8)

    def fit_transform(self, x: ArrayLike) -> np.array:
        self.fit(x)
        return self.transform(x)


class HingeTransformer(BaseEstimator):
    """Fits hinge transformations to an array of covariates."""

    n_hinges_: int = None
    mins_: np.array = None
    maxs_: np.array = None
    hinge_indices_: np.array = None

    def __init__(self, n_hinges: int = MaxentConfig.n_hinge_features):
        self.n_hinges_ = n_hinges

    def fit(self, x: ArrayLike):
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.hinge_indices_ = np.linspace(self.mins_, self.maxs_, self.n_hinges_)

    def transform(self, x: ArrayLike) -> np.array:
        x = np.array(x)
        xarr = repeat_array(x, self.n_hinges_ - 1, axis=-1)
        lharr = repeat_array(self.hinge_indices_[:-1].transpose(), len(x), axis=0)
        rharr = repeat_array(self.hinge_indices_[1:].transpose(), len(x), axis=0)
        lh = left_hinge(xarr, lharr, self.maxs_)
        rh = right_hinge(xarr, self.mins_, rharr)
        return np.concatenate((lh, rh), axis=2).reshape(x.shape[0], -1)

    def fit_transform(self, x: ArrayLike) -> np.array:
        self.fit(x)
        return self.transform(x)


class CategoricalTransformer(BaseEstimator):
    """Applies one-hot encoding to categorical covariate datasets."""

    estimators_: list = None

    def __init__(self):
        pass

    def fit(self, x: ArrayLike):
        self.estimators_ = []
        x = np.array(x)
        if x.ndim == 1:
            estimator = OneHotEncoder(dtype=np.uint8, sparse=False)
            self.estimators_.append(estimator.fit(x.reshape(-1, 1)))
        else:
            nrows, ncols = x.shape
            for col in range(ncols):
                xsub = x[:, col].reshape(-1, 1)
                estimator = OneHotEncoder(dtype=np.uint8, sparse=False)
                self.estimators_.append(estimator.fit(xsub))

    def transform(self, x: ArrayLike) -> np.array:
        x = np.array(x)
        if x.ndim == 1:
            estimator = self.estimators_[0]
            return estimator.transform(x.reshape(-1, 1))
        else:
            class_data = []
            nrows, ncols = x.shape
            for col in range(ncols):
                xsub = x[:, col].reshape(-1, 1)
                estimator = self.estimators_[col]
                class_data.append(estimator.transform(xsub))
            return np.concatenate(class_data, axis=1)

    def fit_transform(self, x: ArrayLike) -> np.array:
        self.fit(x)
        return self.transform(x)


class CumulativeTransformer(QuantileTransformer):
    """Applies a percentile-based transform to estimate cumulative suitability."""

    def __init__(self):
        super().__init__(n_quantiles=100, output_distribution="uniform")


class MaxentFeatureTransformer(BaseEstimator):
    """Transforms covariate data into maxent-format feature data."""

    feature_types_: list = None
    clamp_: bool = None
    n_hinge_features_: int = None
    n_threshold_features_: int = None
    categorical_: list = None
    continuous_: list = None
    labels_: list = None
    estimators_: dict = {
        "linear": None,
        "quadratic": None,
        "product": None,
        "threshold": None,
        "hinge": None,
        "categorical": None,
    }
    feature_ranges_: dict = {
        "linear": None,
        "quadratic": None,
        "product": None,
        "threshold": None,
        "hinge": None,
        "categorical": None,
    }

    def __init__(
        self,
        feature_types: Union[str, list] = MaxentConfig.feature_types,
        clamp: bool = MaxentConfig.clamp,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
    ):
        """Computes features based on the maxent feature types specified (like linear, quadratic, hinge).

        Implemented using sklearn conventions (with `.fit()` and `.transform()` functions.

        Args:
            feature_types: list of maxent features to generate.
            clamp: set feature values to global mins/maxs during prediction
            n_hinge_features: number of hinge knots to generate
            n_threshold_features: nuber of threshold features to generate
        """
        self.feature_types_ = validate_feature_types(feature_types)
        self.clamp_ = validate_boolean(clamp)
        self.n_hinge_features_ = validate_numeric_scalar(n_hinge_features)
        self.n_threshold_features_ = validate_numeric_scalar(n_threshold_features)

    def _format_covariate_data(self, x: ArrayLike) -> Tuple[np.array, np.array]:
        if type(x) is np.ndarray:
            if self.categorical_ is None:
                con = x
                cat = None
            else:
                con = x[:, self.continuous_]
                cat = x[:, self.categorical_]

        elif type(x) is pd.DataFrame:
            con = x[self.continuous_].to_numpy()
            if len(self.categorical_) > 0:
                cat = x[self.categorical_].to_numpy()
            else:
                cat = None

        else:
            raise TypeError(f"Unsupported x dtype: {type(x)}. Must be pd.DataFrame or np.array")

        return con, cat

    def _format_labels_and_dtypes(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        if type(x) is np.ndarray:
            nrows, ncols = x.shape
            if categorical is None:
                continuous = list(range(ncols))
            else:
                continuous = list(set(range(ncols)).difference(set(categorical)))
            self.labels_ = labels or make_band_labels(ncols)
            self.categorical_ = categorical
            self.continuous_ = continuous

        elif type(x) is pd.DataFrame:
            x.drop(["geometry"], axis=1, errors="ignore", inplace=True)
            self.labels_ = labels or list(x.columns)
            self.continuous_ = list(x.select_dtypes(exclude="category").columns)
            self.categorical_ = list(x.select_dtypes(include="category").columns)

    def fit(self, x: ArrayLike, categorical: list = None, labels: list = None):
        self._format_labels_and_dtypes(x, categorical=categorical, labels=labels)
        con, cat = self._format_covariate_data(x)

        if "linear" in self.feature_types_:
            estimator = LinearTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["linear"] = estimator

        if "quadratic" in self.feature_types_:
            estimator = QuadraticTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["quadratic"] = estimator

        if "product" in self.feature_types_:
            estimator = ProductTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["product"] = estimator

        if "threshold" in self.feature_types_:
            estimator = ThresholdTransformer(n_thresholds=self.n_threshold_features_)
            estimator.fit(con)
            self.estimators_["threshold"] = estimator

        if "hinge" in self.feature_types_:
            estimator = HingeTransformer(n_hinges=self.n_hinge_features_)
            estimator.fit(con)
            self.estimators_["hinge"] = estimator

        if cat is not None:
            estimator = CategoricalTransformer()
            estimator.fit(cat)
            self.estimators_["categorical"] = estimator

    def transform(self, x: ArrayLike) -> np.ndarray:
        con, cat = self._format_covariate_data(x)
        features = []

        if "linear" in self.feature_types_:
            features.append(self.estimators_["linear"].transform(con))

        if "quadratic" in self.feature_types_:
            features.append(self.estimators_["quadratic"].transform(con))

        if "product" in self.feature_types_:
            features.append(self.estimators_["product"].transform(con))

        if "threshold" in self.feature_types_:
            features.append(self.estimators_["threshold"].transform(con))

        if "hinge" in self.feature_types_:
            features.append(self.estimators_["hinge"].transform(con))

        if cat is not None:
            features.append(self.estimators_["categorical"].transform(cat))

        return np.concatenate(features, axis=1)


# helper functions


def column_product(array: np.array) -> np.array:
    """Computes the column-wise product of a 2D array."""
    nrows, ncols = array.shape

    if ncols == 1:
        return array
    else:
        products = []
        for xstart in range(0, ncols - 1):
            products.append(array[:, xstart].reshape(nrows, 1) * array[:, xstart + 1 :])
        return np.concatenate(products, axis=1)


def left_hinge(x: ArrayLike, mn: float, mx: float) -> ArrayLike:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    return np.minimum(1, np.maximum(0, (x - mn) / (repeat_array(mx, mn.shape[-1], axis=1) - mn)))


def right_hinge(x: ArrayLike, mn: float, mx: float) -> ArrayLike:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    mn_broadcast = repeat_array(mn, mx.shape[-1], axis=1)
    return np.minimum(1, np.maximum(0, (x - mn_broadcast) / (mx - mn_broadcast)))


def compute_weights(y: ArrayLike, pbr: int = 100) -> ArrayLike:
    """Compute Maxent-format per-sample model weights.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        pbr: presence-to-background weight ratio. pbr=100 sets background samples to 1/100 weight of presence samples.

    Returns:
        weights: array with glmnet-formatted sample weights
    """
    weights = np.array(y + (1 - y) * pbr)
    return weights


def compute_lambdas(
    y: ArrayLike, weights: ArrayLike, reg: ArrayLike, n_lambdas: int = MaxentConfig.n_lambdas
) -> ArrayLike:
    """Computes lambda parameter values for elastic lasso fits.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        weights: per-sample model weights
        reg: per-feature regularization coefficients
        n_lambdas: number of lambda values to estimate

    Returns:
        lambdas: Array of lambda scores of length n_lambda
    """
    n_presence = np.sum(y)
    mean_regularization = np.mean(reg)
    total_weight = np.sum(weights)
    seed_range = np.linspace(4, 0, n_lambdas)
    lambdas = 10 ** (seed_range) * mean_regularization * (n_presence / total_weight)

    return lambdas
