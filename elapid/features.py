"""Functions to transform covariate data into complex model features."""

from typing import Any, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, QuantileTransformer

from elapid.config import MaxentConfig, RegularizationConfig
from elapid.types import ArrayLike, validate_boolean, validate_feature_types, validate_numeric_scalar
from elapid.utils import make_band_labels, repeat_array


class LinearTransformer(MinMaxScaler):
    """Applies linear feature transformations to rescale features from 0-1."""

    clamp: bool = None
    feature_range: None

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
    estimator: BaseEstimator = None

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)

    def fit(self, x: ArrayLike) -> None:
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
        self.estimator.fit(np.array(x) ** 2)

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        return self.estimator.transform(np.array(x) ** 2)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.estimator.transform(np.array(x) ** 2)

    def inverse_transform(self, x: ArrayLike) -> np.ndarray:
        """Revert from transformed features to original covariate values.

        Args:
            x: array-like of shape (n_xamples, n_features)
                Transformed feature data to convert to covariate data.

        Returns:
            ndarray with unscaled covariate values.
        """
        return self.estimator.inverse_transform(np.array(x)) ** 0.5


class ProductTransformer(BaseEstimator):
    """Computes the column-wise product of an array of input features, rescaling from 0-1."""

    clamp: bool = None
    feature_range: Tuple[float, float] = None
    estimator: BaseEstimator = None

    def __init__(
        self,
        clamp: bool = MaxentConfig.clamp,
        feature_range: Tuple[float, float] = (0.0, 1.0),
    ):
        self.clamp = clamp
        self.feature_range = feature_range
        self.estimator = MinMaxScaler(clip=self.clamp, feature_range=self.feature_range)

    def fit(self, x: ArrayLike):
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
        self.estimator.fit(column_product(np.array(x)))

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        return self.estimator.transform(column_product(np.array(x)))

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.transform(x)


class ThresholdTransformer(BaseEstimator):
    """Applies binary thresholds to each covariate based on n evenly-spaced
    thresholds across it's min/max range."""

    n_thresholds_: int = None
    mins_: np.ndarray = None
    maxs_: np.ndarray = None
    threshold_indices_: np.ndarray = None

    def __init__(self, n_thresholds: int = MaxentConfig.n_threshold_features):
        self.n_thresholds_ = n_thresholds

    def fit(self, x: ArrayLike):
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.threshold_indices_ = np.linspace(self.mins_, self.maxs_, self.n_thresholds_)

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        x = np.array(x)
        xarr = repeat_array(x, len(self.threshold_indices_), axis=-1)
        tarr = repeat_array(self.threshold_indices_.transpose(), len(x), axis=0)
        thresh = (xarr > tarr).reshape(x.shape[0], -1)
        return thresh.astype(np.uint8)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.transform(x)


class HingeTransformer(BaseEstimator):
    """Fits hinge transformations to an array of covariates."""

    n_hinges_: int = None
    mins_: np.ndarray = None
    maxs_: np.ndarray = None
    hinge_indices_: np.ndarray = None

    def __init__(self, n_hinges: int = MaxentConfig.n_hinge_features):
        self.n_hinges_ = n_hinges

    def fit(self, x: ArrayLike):
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
        x = np.array(x)
        self.mins_ = x.min(axis=0)
        self.maxs_ = x.max(axis=0)
        self.hinge_indices_ = np.linspace(self.mins_, self.maxs_, self.n_hinges_)

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
        x = np.array(x)
        xarr = repeat_array(x, self.n_hinges_ - 1, axis=-1)
        lharr = repeat_array(self.hinge_indices_[:-1].transpose(), len(x), axis=0)
        rharr = repeat_array(self.hinge_indices_[1:].transpose(), len(x), axis=0)
        lh = left_hinge(xarr, lharr, self.maxs_)
        rh = right_hinge(xarr, self.mins_, rharr)
        return np.concatenate((lh, rh), axis=2).reshape(x.shape[0], -1)

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x)
        return self.transform(x)


class CategoricalTransformer(BaseEstimator):
    """Applies one-hot encoding to categorical covariate datasets."""

    estimators_: list = None

    def __init__(self):
        pass

    def fit(self, x: ArrayLike):
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
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

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
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

    def fit_transform(self, x: ArrayLike) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
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
    categorical_pd_: list = None
    continuous_pd_: list = None
    labels_: list = None
    estimators_: dict = {
        "linear": None,
        "quadratic": None,
        "product": None,
        "threshold": None,
        "hinge": None,
        "categorical": None,
    }
    feature_names_: list = None

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
        """Reads input x data and formats it to consistent array dtypes.

        Args:
            x: array-like of shape (n_samples, n_features)

        Returns:
            (continuous, categorical) tuple of ndarrays with continuous and
                categorical covariate data.
        """
        if isinstance(x, np.ndarray):
            if self.categorical_ is None:
                con = x
                cat = None
            else:
                con = x[:, self.continuous_]
                cat = x[:, self.categorical_]

        elif isinstance(x, pd.DataFrame):
            con = x[self.continuous_pd_].to_numpy()
            if len(self.categorical_pd_) > 0:
                cat = x[self.categorical_pd_].to_numpy()
            else:
                cat = None

        else:
            raise TypeError(f"Unsupported x dtype: {type(x)}. Must be pd.DataFrame or np.array")

        return con, cat

    def _format_labels_and_dtypes(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Read input x data and lists of categorical data indices and band
            labels to format and store this info for later indexing.

        Args:
            s: array-like of shape (n_samples, n_features)
            categorical: indices indicating which x columns are categorical
            labels: covariate column labels. ignored if x is a pandas DataFrame
        """
        if isinstance(x, np.ndarray):
            nrows, ncols = x.shape
            if categorical is None:
                continuous = list(range(ncols))
            else:
                continuous = list(set(range(ncols)).difference(set(categorical)))
            self.labels_ = labels or make_band_labels(ncols)
            self.categorical_ = categorical
            self.continuous_ = continuous

        elif isinstance(x, pd.DataFrame):
            x.drop(["geometry"], axis=1, errors="ignore", inplace=True)
            self.labels_ = labels or list(x.columns)

            # store both pandas and numpy indexing of these values
            self.continuous_pd_ = list(x.select_dtypes(exclude="category").columns)
            self.categorical_pd_ = list(x.select_dtypes(include="category").columns)

            all_columns = list(x.columns)
            self.continuous_ = [all_columns.index(item) for item in self.continuous_pd_ if item in all_columns]
            if len(self.categorical_pd_) != 0:
                self.categorical_ = [all_columns.index(item) for item in self.categorical_pd_ if item in all_columns]
            else:
                self.categorical_ = None

    def fit(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Compute the minimum and maximum for scaling.

        Args:
            x: array-like of shape (n_samples, n_features)
                The data used to compute the per-feature minimum and maximum
                used for later scaling along the features axis.
            categorical: indices indicating which x columns are categorical
            labels: covariate column labels. ignored if x is a pandas DataFrame

        Returns:
            None. Updates the transformer with feature fitting parameters.
        """
        self._format_labels_and_dtypes(x, categorical=categorical, labels=labels)
        con, cat = self._format_covariate_data(x)
        nrows, ncols = con.shape

        feature_names = []
        if "linear" in self.feature_types_:
            estimator = LinearTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["linear"] = estimator
            feature_names += ["linear"] * estimator.n_features_in_

        if "quadratic" in self.feature_types_:
            estimator = QuadraticTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["quadratic"] = estimator
            feature_names += ["quadratic"] * estimator.estimator.n_features_in_

        if "product" in self.feature_types_:
            estimator = ProductTransformer(clamp=self.clamp_)
            estimator.fit(con)
            self.estimators_["product"] = estimator
            feature_names += ["product"] * estimator.estimator.n_features_in_

        if "threshold" in self.feature_types_:
            estimator = ThresholdTransformer(n_thresholds=self.n_threshold_features_)
            estimator.fit(con)
            self.estimators_["threshold"] = estimator
            feature_names += ["threshold"] * (estimator.n_thresholds_ * ncols)

        if "hinge" in self.feature_types_:
            estimator = HingeTransformer(n_hinges=self.n_hinge_features_)
            estimator.fit(con)
            self.estimators_["hinge"] = estimator
            feature_names += ["hinge"] * ((estimator.n_hinges_ - 1) * 2 * ncols)

        if cat is not None:
            estimator = CategoricalTransformer()
            estimator.fit(cat)
            self.estimators_["categorical"] = estimator
            for est in estimator.estimators_:
                feature_names += ["categorical"] * len(est.categories_[0])

        self.feature_names_ = feature_names

    def transform(self, x: ArrayLike) -> np.ndarray:
        """Scale covariates according to the feature range.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data that will be transformed.

        Returns:
            ndarray with transformed data.
        """
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

    def fit_transform(self, x: ArrayLike, categorical: list = None, labels: list = None) -> np.ndarray:
        """Fits scaler to x and returns transformed features.

        Args:
            x: array-like of shape (n_samples, n_features)
                Input data to fit the scaler and to transform.

        Returns:
            ndarray with transformed data.
        """
        self.fit(x, categorical=categorical, labels=labels)
        return self.transform(x)


# helper functions


def column_product(array: np.ndarray) -> np.ndarray:
    """Computes the column-wise product of a 2D array.

    Args:
        array: array-like of shape (n_samples, n_features)

    Returns:
        ndarray with of shape (n_samples, factorial(n_features-1))
    """
    nrows, ncols = array.shape

    if ncols == 1:
        return array
    else:
        products = []
        for xstart in range(0, ncols - 1):
            products.append(array[:, xstart].reshape(nrows, 1) * array[:, xstart + 1 :])
        return np.concatenate(products, axis=1)


def left_hinge(x: ArrayLike, mn: float, mx: float) -> np.ndarray:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    return np.minimum(1, np.maximum(0, (x - mn) / (repeat_array(mx, mn.shape[-1], axis=1) - mn)))


def right_hinge(x: ArrayLike, mn: float, mx: float) -> np.ndarray:
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


def compute_weights(y: ArrayLike, pbr: int = 100) -> np.ndarray:
    """Compute Maxent-format per-sample model weights.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        pbr: presence-to-background weight ratio. pbr=100 sets background samples to 1/100 weight of presence samples.

    Returns:
        weights: array with glmnet-formatted sample weights
    """
    weights = np.array(y + (1 - y) * pbr)
    return weights


def compute_regularization(
    y: ArrayLike,
    z: np.ndarray,
    feature_labels: List[str],
    beta_multiplier: float = MaxentConfig.beta_multiplier,
    beta_lqp: float = MaxentConfig.beta_lqp,
    beta_threshold: float = MaxentConfig.beta_threshold,
    beta_hinge: float = MaxentConfig.beta_hinge,
    beta_categorical: float = MaxentConfig.beta_hinge,
) -> np.ndarray:
    """Computes variable regularization values for all feature data.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        z: model features (transformations applied to covariates)
        feature_labels: list of length n_features, with labels identifying each column's feature type
            with options ["linear", "quadratic", "product", "threshold", "hinge", "categorical"]
        beta_multiplier: scaler for all regularization parameters. higher values exclude more features
        beta_lqp: scaler for linear, quadratic and product feature regularization
        beta_threshold: scaler for threshold feature regularization
        beta_hinge: scaler for hinge feature regularization
        beta_categorical: scaler for categorical feature regularization

    Returns:
        max_reg: Array with per-feature regularization parameters
    """
    # compute regularization based on presence-only locations
    z1 = z[y == 1]
    nrows, ncols = z1.shape
    labels = np.array(feature_labels)
    nlabels = len(feature_labels)

    assert nlabels == ncols, f"number of feature_labels ({nlabels}) must match number of features ({ncols})"

    # create arrays to store the regularization params
    base_regularization = np.zeros(ncols)
    hinge_regularization = np.zeros(ncols)
    threshold_regularization = np.zeros(ncols)

    # use a different reg table based on the features set
    if "product" in labels:
        table_lqp = RegularizationConfig.product
    elif "quadratic" in labels:
        table_lqp = RegularizationConfig.quadratic
    else:
        table_lqp = RegularizationConfig.linear

    if "linear" in labels:
        linear_idxs = labels == "linear"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[linear_idxs] = reg

    if "quadratic" in labels:
        quadratic_idxs = labels == "quadratic"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[quadratic_idxs] = reg

    if "product" in labels:
        product_idxs = labels == "product"
        fr_max, fr_min = table_lqp
        multiplier = beta_lqp
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[product_idxs] = reg

    if "threshold" in labels:
        threshold_idxs = labels == "threshold"
        fr_max, fr_min = RegularizationConfig.threshold
        multiplier = beta_threshold
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[threshold_idxs] = reg

        # increase regularization for uniform threshlold values
        all_zeros = np.all(z1 == 0, axis=0)
        all_ones = np.all(z1 == 1, axis=0)
        threshold_regularization[all_zeros] = 1
        threshold_regularization[all_ones] = 1

    if "hinge" in labels:
        hinge_idxs = labels == "hinge"
        fr_max, fr_min = RegularizationConfig.hinge
        multiplier = beta_hinge
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[hinge_idxs] = reg

        # increase regularization for extreme hinge values
        hinge_std = np.std(z1[:, hinge_idxs], ddof=1, axis=0)
        hinge_sqrt = np.zeros(len(hinge_std)) + (1 / np.sqrt(nrows))
        std = np.max((hinge_std, hinge_sqrt), axis=0)
        hinge_regularization[hinge_idxs] = (0.5 * std) / np.sqrt(nrows)

    if "categorical" in labels:
        categorical_idxs = labels == "categorical"
        fr_max, fr_min = RegularizationConfig.categorical
        multiplier = beta_categorical
        ap = np.interp(nrows, fr_max, fr_min)
        reg = multiplier * ap / np.sqrt(nrows)
        base_regularization[categorical_idxs] = reg

    # compute the maximum regularization based on a few different approaches
    default_regularization = 0.001 * (np.max(z, axis=0) - np.min(z, axis=0))
    variance_regularization = np.std(z1, ddof=1, axis=0) * base_regularization
    max_regularization = np.max(
        (default_regularization, variance_regularization, hinge_regularization, threshold_regularization), axis=0
    )

    # apply the final scaling factor
    max_regularization *= beta_multiplier

    return max_regularization


def compute_lambdas(
    y: ArrayLike, weights: ArrayLike, reg: ArrayLike, n_lambdas: int = MaxentConfig.n_lambdas
) -> np.ndarray:
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
