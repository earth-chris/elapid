"""Functions to transform covariate data into complex model features."""

from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from elapid.utils import MAXENT_DEFAULTS, ArrayLike, repeat_array


class MaxentFeatureTransformer(object):
    """Transforms covariate data into maxent-format feature data.

    Attributes:
        feature_types_: List of the feature types (quadratic, etc.) to compute
        clamp_: Boolean to clamp feature to the min/max of the fit data
        n_hinge_features_: Int of the linear hinge features to fit
        n_threshold_features_: Int of the number of thresholds to fit
        initialized_: Boolean flag identifying the transformer's fit status
        hinge_ranges_: List of the hinge ranges fit
        threshold_ranges_: List of the thresholds fit
        categorical_encoders_: List of `sklearn` one-hot class encoders
        categorical_: Array-like of column indices for which columns are categorical
        feature_mins_: Array-like of the feature minimum values for clamping
        feature_maxs_: Array-like of the feature maximum values for clamping
        labels_: List of covariate labels
    """

    def __init__(
        self,
        feature_types: Union[str, list] = MAXENT_DEFAULTS["feature_types"],
        clamp: bool = MAXENT_DEFAULTS["clamp"],
        n_hinge_features: int = MAXENT_DEFAULTS["n_hinge_features"],
        n_threshold_features: int = MAXENT_DEFAULTS["n_threshold_features"],
    ):
        """Computes features based on the maxent feature types specified (like linear, quadratic, hinge).

        Implemented using sklearn conventions (usng `.fit()` and `.transform()` functions.

        Args:
            feature_types: list of maxent features to generate.
            clamp: set feature values to global mins/maxs during prediction
            n_hinge_features: number of hinge knots to generate
            n_threshold_features: nuber of threshold features to generate
        """
        # user-specified parameters
        self.feature_types_ = validate_feature_types(feature_types)
        self.clamp_ = validate_boolean(clamp)
        self.n_hinge_features_ = validate_numeric_scalar(n_hinge_features)
        self.n_threshold_features_ = validate_numeric_scalar(n_threshold_features)

        # data-driven parameters
        self.initialized_ = False
        self.hinge_ranges_ = dict()
        self.threshold_ranges_ = dict()
        self.categorical_encoders_ = dict()
        self.categorical_ = None
        self.feature_mins_ = None
        self.feature_maxs_ = None
        self.labels_ = None

    def fit(self, x: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Fits features to covariates.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            categorical: indices for which columns are categorical
            labels: covariate labels. ignored if x is a pandas DataFrame

        Returns:
            None: Updates the transformer object with feature fitting parameters.
        """
        con, cat = self._format_covariate_data(x, categorical=categorical, labels=labels)
        self._compute_features(con, cat, transform=False)
        self.initialized_ = True

    def transform(self, x: ArrayLike, categorical: list = None, labels: list = None) -> pd.DataFrame:
        """Applies feature transformations to covariates.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            categorical: indices for which columns are categorical
            labels: list of covariate labels. ignored if x is a pandas DataFrame

        Returns:
            features: DataFrame with feature transformations applied to x.
        """
        assert self.initialized_, "Transformer must be fit first"

        if categorical is None:
            categorical = self.categorical_

        if labels is None:
            labels = self.labels_

        con, cat = self._format_covariate_data(x, categorical=categorical, labels=labels)
        features = self._compute_features(con, cat, transform=True)

        if self.clamp_:
            features = self._clamp_features(features)

        return features

    def fit_transform(self, x: ArrayLike, categorical: list = None, labels: list = None) -> pd.DataFrame:
        """Fits features and applies transformations to covariates x.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            categorical: indices for which columns are categorical
            labels: list of covariate labels. ignored if x is a pandas DataFrame

        Returns:
            features: DataFrame with feature transformation applied to x
        """
        con, cat = self._format_covariate_data(x, categorical=categorical, labels=labels)
        features = self._compute_features(con, cat, transform=False)
        self.initialized_ = True

        return features

    def _format_covariate_data(
        self, x: ArrayLike, categorical: list = None, labels: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Standardizes array-like input data to a consistent data structure.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            categorical: indices for which columns are categorical
            labels: list of covariate labels. ignored if x is a pandas DataFrame

        Returns:
            (con, cat): pandas DataFrames with continuous and categorical covariates
        """
        if isinstance(x, pd.DataFrame):
            x.drop(["geometry"], axis=1, errors="ignore", inplace=True)
            con = x.select_dtypes(exclude="category")
            cat = x.select_dtypes(include="category")
            self.labels_ = list(x.columns) if labels is None else labels
        else:
            self.categorical_ = categorical
            con, cat = self._covariates_to_df(x, categorical=categorical, labels=labels)

        return con, cat

    def _covariates_to_df(
        self, x: ArrayLike, categorical: list = None, labels: list = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Converts 2D numerical arrays into labeled pandas DataFrames for continuous and categorical variables

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            categorical: indices for which columns are categorical
            labels: list of covariate labels. ignored if x is a pandas DataFrame

        Returns:
            (con, cat): pandas DataFrames with continuous and categorical covariates
        """
        # cast x to 2d if only one feature is passed
        if np.ndim(x) == 1:
            x = x.reshape(-1, 1)

        # auto-generate class labels
        if labels is None:
            labels = [f"cov_{i+1}" for i in range(x.shape[1])]

        # subst the continuous / categorical data based on how the "categorical" argument is passed
        if categorical is None:
            con = x
            cat = None
            con_labels = labels
            cat_labels = []

        # treat 1-d categorical parameters as an index
        else:
            if np.ndim(categorical) == 1:
                continuous = list(range(x.shape[1]))
                [continuous.pop(cat_idx) for cat_idx in categorical]
                con = x[:, continuous]
                cat = x[:, categorical]
                con_labels = [labels[con_idx] for con_idx in continuous]
                cat_labels = [labels[cat_idx] for cat_idx in categorical]

            # treat n-d arrays as covariates
            else:
                con = x
                cat = categorical
                con_labels = labels
                cat_labels = [f"cov_{len(labels)+i+1}" for i in range(cat.shape[1])]

        # concatenate multiple series to get around stupid multi-category pandas issues
        if cat is None:
            cat_df = pd.DataFrame()
        else:
            cat_list = [
                pd.Series(cat[:, i], name=cat_label, dtype="category") for i, cat_label in enumerate(cat_labels)
            ]
            cat_df = pd.concat(cat_list, axis=1)

        con_df = pd.DataFrame(con, columns=con_labels)

        # save labels for applying to new datasets
        if not self.initialized_:
            self.labels_ = labels
            self.categorical_ = self.categorical_

        return (con_df, cat_df)

    def _compute_features(self, con: pd.DataFrame, cat: pd.DataFrame, transform: bool = False) -> pd.DataFrame:
        """Transforms input data to the features used for model training.

        Args:
            con: DataFrame encoded with continuous (i.e. numeric) covariates
            cat: DataFrame encoded with categorical (i.e. class) covariates
            transform: apply already-fit transformations.

        Returns:
            features: DataFrame with the feature transformations applied to each column
        """
        categorical_covariates = list(cat.columns)
        continuous_covariates = list(con.columns)
        feature_list = list()

        # categorical feature transforms
        for covariate in categorical_covariates:

            series = cat[covariate]
            classes = list(series.unique())
            classes.sort()
            feature_names = [f"{covariate}_class_{clas}" for clas in classes]

            if transform:
                encoder = self.categorical_encoders_[covariate]
                one_hot_encoded = encoder.transform(series.to_numpy().reshape(-1, 1))
            else:
                encoder = OneHotEncoder(sparse=False, dtype=np.uint8)
                one_hot_encoded = encoder.fit_transform(series.to_numpy().reshape(-1, 1))
                self.categorical_encoders_[covariate] = encoder

            feature_df = pd.DataFrame(one_hot_encoded, columns=feature_names)
            feature_list.append(feature_df)

        # continuous feature transforms
        for covariate in continuous_covariates:
            series = con[covariate]

            if "linear" in self.feature_types_:

                feature_list.append(series.rename(f"{covariate}_linear"))

            if "quadratic" in self.feature_types_:

                feature_list.append((series ** 2).rename(f"{covariate}_squared"))

            if "hinge" in self.feature_types_:

                if not transform:
                    self.hinge_ranges_[covariate] = [series.min(), series.max()]

                hinges = hinge(series.to_numpy(), n_hinges=self.n_hinge_features_, range=self.hinge_ranges_[covariate])
                feature_names = [f"{covariate}_hinge_{i+1:03d}" for i in range((self.n_hinge_features_ - 1) * 2)]
                feature_df = pd.DataFrame(hinges, columns=feature_names)
                feature_list.append(feature_df)

            if "threshold" in self.feature_types_:

                if not transform:
                    self.threshold_ranges_[covariate] = [series.min(), series.max()]

                thresholds = threshold(
                    series.to_numpy(), n_thresholds=self.n_threshold_features_, range=self.threshold_ranges_[covariate]
                )
                feature_names = [f"{covariate}_threshold_{i+1:03d}" for i in range(self.n_threshold_features_ - 2)]
                feature_df = pd.DataFrame(thresholds, columns=feature_names)
                feature_list.append(feature_df)

            if "product" in self.feature_types_:

                idx_cov = continuous_covariates.index(covariate)
                for i in range(idx_cov, len(continuous_covariates) - 1):
                    feature_name = f"{covariate}_x_{continuous_covariates[i+1]}"
                    product = series * con[continuous_covariates[i + 1]]
                    feature_df = pd.DataFrame(product, columns=[feature_name])
                    feature_list.append(feature_df)

        features = pd.concat(feature_list, axis=1)

        # store mins and maxes to clamp features later
        if not transform:
            self.feature_mins_ = features.min()
            self.feature_maxs_ = features.max()

        return features

    def _clamp_features(self, features: ArrayLike) -> ArrayLike:
        """Sets features to the min/max of the global range of the original fit.

        Args:
            features: array-like of shape (n_samples, n_features)

        Returns:
            features: array-like with values clamped to global min/max
        """
        assert self.initialized_, "Transformer must be fit first"

        return features.apply(clamp_row, axis=1, raw=True, mins=self.feature_mins_, maxs=self.feature_maxs_)


def hingeval(x: ArrayLike, mn: float, mx: float) -> ArrayLike:
    """Computes hinge transformation values.

    Args:
        x: Array-like of covariate values
        mn: Minimum covariate value to fit hinges to
        mx: Maximum covariate value to fit hinges to

    Returns:
        Array of hinge features
    """
    return np.minimum(1, np.maximum(0, (x - mn) / (mx - mn)))


def hinge(
    x: ArrayLike, n_hinges: int = MAXENT_DEFAULTS["n_hinge_features"], range: Tuple[float, float] = None
) -> ArrayLike:
    """Fits hinge transformations to an array of covariates.

    Args:
        x: Array-like of covariate values
        n_hinges: number of transformations to apply
        range: min/max covariate values to fit across

    Returns:
        Array of hinge features of shape (n_samples, (n_hinges-1) * 2)
    """
    mn = range[0] if range is not None else np.min(x)
    mx = range[1] if range is not None else np.max(x)
    k = np.linspace(mn, mx, n_hinges)

    xarr = repeat_array(x, len(k) - 1, axis=1)
    lharr = repeat_array(k[:-1], len(x), axis=0)
    rharr = repeat_array(k[1:], len(x), axis=0)

    lh = hingeval(xarr, lharr, mx)
    rh = hingeval(xarr, mn, rharr)

    return np.concatenate((lh, rh), axis=1)


def threshold(
    x, n_thresholds: int = MAXENT_DEFAULTS["n_threshold_features"], range: Tuple[float, float] = None
) -> ArrayLike:
    """Fits arbitrary threshold transformations to an array of covariates.

    Args:
        x: array-like of covariate values
        n_thresholds: number of transformations to apply
        range: min/max covariate values to fit across

    Returns:
        array of thresholds features of shape (n_samples, n_thresholds - 2)
    """
    mn = range[0] if range is not None else np.min(x)
    mx = range[1] if range is not None else np.max(x)
    k = np.linspace(mn, mx, n_thresholds + 2)[2:-2]

    xarr = repeat_array(x, len(k), axis=1)
    tarr = repeat_array(k, len(x), axis=0)

    return (xarr > tarr).astype(np.uint8)


def clamp_row(row: pd.Series, mins: ArrayLike, maxs: ArrayLike) -> ArrayLike:
    """Clamps feature data to a min/max range. Designed for df.apply()

    Args:
        row: DataFrame row / 1-d array of feature values
        mins: array of global feature minimum values
        maxs: array of global feature maximum values

    Returns:
        array of clamped feature values
    """
    return np.min([maxs, np.max([row, mins], axis=0)], axis=0)


def compute_lambdas(y: ArrayLike, weights: ArrayLike, reg: ArrayLike, n_lambda: int = 200) -> ArrayLike:
    """Computes lambda parameter values for elastic lasso fits.

    Args:
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        weights: per-sample model weights
        reg: per-feature regularization coefficients
        n_lambda: number of lambda values to estimate

    Returns:
        lambdas: Array of lambda scores of length n_lambda
    """
    n_presence = np.sum(y)
    mean_regularization = np.mean(reg)
    total_weight = np.sum(weights)
    seed_range = np.linspace(4, 0, n_lambda)
    lambdas = 10 ** (seed_range) * mean_regularization * (n_presence / total_weight)

    return lambdas


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


def compute_regularization(
    f: pd.DataFrame,
    y: ArrayLike,
    beta_multiplier: float = MAXENT_DEFAULTS["beta_multiplier"],
    beta_lqp: float = MAXENT_DEFAULTS["beta_lqp"],
    beta_threshold: float = MAXENT_DEFAULTS["beta_threshold"],
    beta_hinge: float = MAXENT_DEFAULTS["beta_hinge"],
    beta_categorical: float = MAXENT_DEFAULTS["beta_hinge"],
) -> ArrayLike:
    """Computes variable regularization values for all feature data.

    Args:
        f: DataFrame with feature transformations applied
        y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        beta_multiplier: scaler for all regularization parameters. higher values exclude more features
        beta_lqp: scaler for linear, quadratic and product feature regularization
        beta_threshold: scaler for threshold feature regularization
        beta_hinge: scaler for hinge feature regularization
        beta_categorical: scaler for categorical feature regularization

    Returns:
        max_reg: Array with per-feature regularization parameters
    """

    # tailor the regularization to presence-only locations
    mm = f[y == 1]
    n_points = len(mm)
    features = list(f.columns)
    n_features = len(features)
    regularization = np.zeros(n_features)

    # set the default regularization values
    q_features = len([i for i in features if "_squared" in i])
    p_features = len([i for i in features if "_x_" in i])
    if q_features > 0:
        regtable = [[0, 10, 17, 30, 100], [1.3, 0.8, 0.5, 0.25, 0.05]]
    elif p_features > 0:
        regtable = [[0, 10, 17, 30, 100], [2.6, 1.6, 0.9, 0.55, 0.05]]
    else:
        regtable = [[0, 10, 30, 100], [1, 1, 0.2, 0.05]]

    for i, feature in enumerate(features):

        if "_linear" in feature or "_squared" in feature or "_x_" in feature:
            freg = regtable
            multiplier = beta_lqp
        elif "_hinge" in feature:
            freg = [[0, 1], [0.5, 0.5]]
            multiplier = beta_hinge
        elif "_threshold" in feature:
            freg = [[0, 100], [2, 1]]
            multiplier = beta_threshold
        elif "_class" in feature:
            freg = [[0, 10, 17], [0.65, 0.5, 0.25]]
            multiplier = beta_categorical

        ap = np.interp(n_points, freg[0], freg[1])
        regularization[i] = multiplier * ap / np.sqrt(n_points)

    # increase regularization for extreme hinge values
    hinge_features = [i for i in features if "_hinge_" in i]
    hinge_reg = np.zeros(n_features)
    for hinge_feature in hinge_features:
        hinge_idx = features.index(hinge_feature)
        std = np.max([np.std(mm[hinge_feature], ddof=1), (1 / np.sqrt(n_points))])
        hinge_reg[hinge_idx] = (0.5 * std) / np.sqrt(n_points)

    # increase threshold regularization for uniform values
    threshold_features = [i for i in features if "_threshold_" in i]
    threshold_reg = np.zeros(n_features)
    for threshold_feature in threshold_features:
        threshold_idx = features.index(threshold_feature)
        all_zeros = np.all(mm[threshold_feature] == 0)
        all_ones = np.all(mm[threshold_feature] == 1)
        threshold_reg[threshold_idx] = 1 if all_zeros or all_ones else 0

    # report the max regularization value
    default_reg = 0.001 * (np.max(f, axis=0) - np.min(f, axis=0))
    variance_reg = np.std(mm, axis=0, ddof=1) * regularization
    max_reg = np.max([default_reg, variance_reg, hinge_reg, threshold_reg], axis=0)

    # apply the final scaling factor
    max_reg *= beta_multiplier

    return max_reg


def validate_feature_types(features: Union[str, list]) -> list:
    """Ensures the feature classes passed are maxent-legible

    Args:
        features: List or string that must be in ["linear", "quadratic", "product",
            "hinge", "threshold", "auto"] or string "lqphta"

    Returns:
        valid_features: List of formatted valid feature values
    """
    valid_list = ["linear", "quadratic", "product", "hinge", "threshold"]
    valid_string = "lqpht"
    valid_features = list()

    assert features is not None, "Features cannot be empty"

    # ensure the string features are valid, and translate to a standard feature list
    if type(features) is str:
        for feature in features:

            if feature == "a":
                return valid_list

            assert feature in valid_string, "Invalid feature passed: {}".format(feature)

            if feature == "l":
                valid_features.append("linear")
            elif feature == "q":
                valid_features.append("quadratic")
            elif feature == "p":
                valid_features.append("product")
            elif feature == "h":
                valid_features.append("hinge")
            elif feature == "t":
                valid_features.append("threshold")

    # or ensure the list features are valid
    elif type(features) is list:
        for feature in features:
            if feature == "auto":
                return valid_list

            assert feature in valid_list, "Invalid feature passed: {}".format(feature)

            valid_features.append(feature)

    return valid_features


def validate_boolean(var: Any) -> bool:
    """Evaluates whether an argument is boolean True/False

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not boolean
    """
    assert isinstance(var, bool), "Argument must be True/False"
    return var


def validate_numeric_scalar(var: Any) -> bool:
    """Evaluates whether an argument is a single numeric value.

    Args:
        var: the input argument to validate

    Returns:
        var: the value if it passes validation

    Raises:
        AssertionError: `var` was not numeric.
    """
    assert isinstance(var, (int, float)), "Argument must be single numeric value"
    return var
