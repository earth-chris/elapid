"""Classes for training species distribution models."""
from typing import List, Tuple, Union
from warnings import warn

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from elapid import features as _features
from elapid.config import MaxentConfig, NicheEnvelopeConfig
from elapid.types import ArrayLike, Number, validate_feature_types
from elapid.utils import NCPUS, make_band_labels

# handle windows systems without functioning gfortran compilers
FORCE_SKLEARN = False
try:
    from glmnet.logistic import LogitNet

except ModuleNotFoundError:
    warn(
        "Failed to import glmnet: using sklearn for Maxent. Interpret results with caution.",
        category=RuntimeWarning,
    )
    FORCE_SKLEARN = True


class MaxentModel(BaseEstimator):
    """Model estimator for Maxent-style species distribution models."""

    # passed to __init__
    feature_types: list = MaxentConfig.feature_types
    tau: float = MaxentConfig.tau
    clamp: bool = MaxentConfig.clamp
    scorer: str = MaxentConfig.scorer
    beta_multiplier: float = MaxentConfig.beta_multiplier
    beta_hinge: float = MaxentConfig.beta_hinge
    beta_lqp: float = MaxentConfig.beta_lqp
    beta_threshold: float = MaxentConfig.beta_threshold
    beta_categorical: float = MaxentConfig.beta_categorical
    n_hinge_features: int = MaxentConfig.n_hinge_features
    n_threshold_features: int = MaxentConfig.n_threshold_features
    convergence_tolerance: float = MaxentConfig.tolerance
    use_lambdas: str = MaxentConfig.use_lambdas
    n_lambdas: str = MaxentConfig.n_lambdas
    class_weights: Union[str, float] = None
    n_cpus: int = NCPUS
    use_sklearn: bool = False

    # computed during model fitting
    initialized_: bool = False
    estimator: BaseEstimator = None
    preprocessor: BaseEstimator = None
    transformer: BaseEstimator = None
    regularization_: np.ndarray = None
    sample_weights_: np.ndarray = None
    lambdas_: np.ndarray = None
    beta_scores_: np.array = None
    entropy_: float = 0.0
    alpha_: float = 0.0

    def __init__(
        self,
        feature_types: Union[list, str] = MaxentConfig.feature_types,
        tau: float = MaxentConfig.tau,
        clamp: bool = MaxentConfig.clamp,
        scorer: str = MaxentConfig.scorer,
        beta_multiplier: float = MaxentConfig.beta_multiplier,
        beta_lqp: float = MaxentConfig.beta_lqp,
        beta_hinge: float = MaxentConfig.beta_hinge,
        beta_threshold: float = MaxentConfig.beta_lqp,
        beta_categorical: float = MaxentConfig.beta_categorical,
        n_hinge_features: int = MaxentConfig.n_hinge_features,
        n_threshold_features: int = MaxentConfig.n_threshold_features,
        convergence_tolerance: float = MaxentConfig.tolerance,
        use_lambdas: str = MaxentConfig.use_lambdas,
        n_lambdas: int = MaxentConfig.n_lambdas,
        class_weights: Union[str, float] = MaxentConfig.class_weights,
        n_cpus: int = NCPUS,
        use_sklearn: bool = FORCE_SKLEARN,
    ):
        """Create a maxent model object.

        Args:
            feature_types: maxent feature types to fit. must be in string "lqphta" or
                list ["linear", "quadratic", "product", "hinge", "threshold", "auto"]
            tau: maxent prevalence value for scaling logistic output
            clamp: set features to min/max range from training during prediction
            scorer: sklearn scoring function for model training
            beta_multiplier: scaler for all regularization parameters.
                higher values drop more coeffiecients
            beta_lqp: linear, quadratic and product feature regularization scaler
            beta_hinge: hinge feature regularization scaler
            beta_threshold: threshold feature regularization scaler
            beta_categorical: categorical feature regularization scaler
            n_hinge_features: the number of hinge features to fit in feature transformation
            n_threshold_features: the number of thresholds to fit in feature transformation
            convergence_tolerance: model convergence tolerance level
            use_lambdas: guide for which model lambdas to select (either "best" or "last")
            n_lambdas: number of lamba values to fit models with
            class_weights: strategy for weighting presence samples.
                pass "balance" to compute the ratio based on sample frequency
                or pass a float for the presence:background weight ratio
                the R `maxnet` package uses a value of 100 as default
            n_cpus: threads to use during model training
            use_sklearn: force using `sklearn` for fitting logistic regression.
                turned off by default to use `glmnet` for fitting.
                this feature was turned on to support Windows users
                who install the package without a fortran compiler.
        """
        self.feature_types = validate_feature_types(feature_types)
        self.tau = tau
        self.clamp = clamp
        self.scorer = scorer
        self.beta_multiplier = beta_multiplier
        self.beta_hinge = beta_hinge
        self.beta_lqp = beta_lqp
        self.beta_threshold = beta_threshold
        self.beta_categorical = beta_categorical
        self.n_hinge_features = n_hinge_features
        self.n_threshold_features = n_threshold_features
        self.convergence_tolerance = convergence_tolerance
        self.n_cpus = n_cpus
        self.use_lambdas = use_lambdas
        self.n_lambdas = n_lambdas
        self.class_weights = class_weights
        self.use_sklearn = use_sklearn

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categorical: List[int] = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
    ) -> None:
        """Trains a maxent model using a set of covariates and presence/background points.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: indices for which columns are categorical
            labels: covariate labels. ignored if x is a pandas DataFrame
            preprocessor: an `sklearn` transformer with a .transform() and/or
                a .fit_transform() method. Some examples include a PCA() object or a
                RobustScaler().
        """
        # clear state variables
        self.alpha_ = 0.0
        self.entropy_ = 0.0

        # format the input data
        y = format_occurrence_data(y)

        # apply preprocessing
        if preprocessor is not None:
            self.preprocessor = preprocessor
            try:
                x = self.preprocessor.transform(x)
            except NotFittedError:
                x = self.preprocessor.fit_transform(x)

        # fit the feature transformer
        self.transformer = _features.MaxentFeatureTransformer(
            feature_types=self.feature_types,
            clamp=self.clamp,
            n_hinge_features=self.n_hinge_features,
            n_threshold_features=self.n_threshold_features,
        )
        features = self.transformer.fit_transform(x, categorical=categorical, labels=labels)
        feature_labels = self.transformer.feature_names_

        # compute sample weights
        pbr = len(y) / y.sum() if self.class_weights == "balanced" else self.class_weights
        self.sample_weights_ = _features.compute_weights(y, pbr=pbr)

        # model fitting with sklearn
        if self.use_sklearn:
            C = estimate_C_from_betas(self.beta_multiplier)
            self.initialize_sklearn_model(C)
            self.estimator.fit(features, y, sample_weight=self.sample_weights_)
            self.beta_scores_ = self.estimator.coef_[0]

        # model fitting with glmnet
        else:
            # set feature regularization parameters
            self.regularization_ = _features.compute_regularization(
                y,
                features,
                feature_labels=feature_labels,
                beta_multiplier=self.beta_multiplier,
                beta_lqp=self.beta_lqp,
                beta_threshold=self.beta_threshold,
                beta_hinge=self.beta_hinge,
                beta_categorical=self.beta_categorical,
            )

            # get model lambda scores to initialize the glm
            self.lambdas_ = _features.compute_lambdas(
                y, self.sample_weights_, self.regularization_, n_lambdas=self.n_lambdas
            )

            # model fitting
            self.initialize_glmnet_model(lambdas=self.lambdas_)
            self.estimator.fit(
                features,
                y,
                sample_weight=self.sample_weights_,
                relative_penalties=self.regularization_,
            )

            # get the beta values based on which lambda selection method to use
            if self.use_lambdas == "last":
                self.beta_scores_ = self.estimator.coef_path_[0, :, -1]
            elif self.use_lambdas == "best":
                self.beta_scores_ = self.estimator.coef_path_[0, :, self.estimator.lambda_max_inx_]

        # store initialization state
        self.initialized_ = True

        # apply maxent-specific transformations
        raw = self.predict(x[y == 0], transform="raw")

        # alpha is a normalizing constant that ensures that f1(z) integrates (sums) to 1
        self.alpha_ = maxent_alpha(raw)

        # the distance from f(z) is the relative entropy of f1(z) WRT f(z)
        self.entropy_ = maxent_entropy(raw)

    def predict(self, x: ArrayLike, transform: str = "cloglog") -> ArrayLike:
        """Applies a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            transform: maxent model transformation type. select from
                ["raw", "logistic", "cloglog"].

        Returns:
            predictions: array-like of shape (n_samples,) with model predictions
        """
        if not self.initialized_:
            raise NotFittedError("Model must be fit first")

        # feature transformations
        x = x if self.preprocessor is None else self.preprocessor.transform(x)
        features = x if self.transformer is None else self.transformer.transform(x)

        # apply the model
        engma = np.matmul(features, self.beta_scores_) + self.alpha_

        # scale based on the transform type
        if transform == "raw":
            return maxent_raw_transform(engma)

        elif transform == "logistic":
            return maxent_logistic_transform(engma, self.entropy_, self.tau)

        elif transform == "cloglog":
            return maxent_cloglog_transform(engma, self.entropy_)

    def fit_predict(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categorical: list = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
        transform: str = "cloglog",
    ) -> ArrayLike:
        """Trains and applies a model to x/y data.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: column indices indicating which columns are categorical
            labels: Covariate labels. Ignored if x is a pandas DataFrame
            preprocessor: an `sklearn` transformer with a .transform() and/or
                a .fit_transform() method. Some examples include a PCA() object or a
                RobustScaler().
            transform: maxent model transformation type. select from
                ["raw", "logistic", "cloglog"].

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels, preprocessor=preprocessor)
        predictions = self.predict(x, transform=transform)

        return predictions

    def initialize_glmnet_model(
        self,
        lambdas: np.array,
        alpha: float = 1,
        standardize: bool = False,
        fit_intercept: bool = True,
    ) -> None:
        """Creates the Logistic Regression with elastic net penalty model object.

        Args:
            lambdas: array of model lambda values. get from elapid.features.compute_lambdas()
            alpha: elasticnet mixing parameter. alpha=1 for lasso, alpha=0 for ridge
            standardize: specify coefficient normalization
            fit_intercept: include an intercept parameter
        """
        self.estimator = LogitNet(
            alpha=alpha,
            lambda_path=lambdas,
            standardize=standardize,
            fit_intercept=fit_intercept,
            scoring=self.scorer,
            n_jobs=self.n_cpus,
            tol=self.convergence_tolerance,
        )

    def initialize_sklearn_model(self, C: float, fit_intercept: bool = True) -> None:
        """Creates an sklearn Logisticregression estimator with L1 penalties.

        Args:
            C: the regularization parameter
            fit_intercept: include an intercept parameter
        """
        self.estimator = LogisticRegression(
            C=C,
            fit_intercept=fit_intercept,
            penalty="l1",
            solver="liblinear",
            tol=self.convergence_tolerance,
            max_iter=self.n_lambdas,
        )


class NicheEnvelopeModel(BaseEstimator):
    """Model estimator for niche envelope-style models."""

    percentile_range: Tuple[float, float] = None
    feature_mins_: np.ndarray = None
    feature_maxs_: np.ndarray = None
    categorical_estimator: BaseEstimator = None
    categorical_: list = None
    continuous_: list = None
    categorical_pd_: list = None
    continuous_pd_: list = None
    in_categorical_: np.ndarray = None

    def __init__(self, percentile_range: Tuple[float, float] = NicheEnvelopeConfig.percentile_range):
        """Create a niche envelope model estimator.

        Args:
            percentile_range: covariate values within this range are flagged as suitable habitat
                using a narrow range like [10, 90] drops more areas from suitability maps
                while [0, 100] creates an envelope around the full range of observed
                covariates at all y==1 locations.
        """
        self.percentile_range = percentile_range
        self.categorical_estimator = _features.CategoricalTransformer()

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

    def fit(self, x: ArrayLike, y: ArrayLike, categorical: list = None, labels: list = None) -> None:
        """Fits a niche envelope model using a set of covariates and presence/background points.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: indices for which columns are categorical
            labels: covariate labels. ignored if x is a pandas DataFrame
        """
        # format the input x/y data
        self._format_labels_and_dtypes(x, categorical=categorical, labels=labels)
        con, cat = self._format_covariate_data(x)
        y = format_occurrence_data(y)

        # estimate the feature range of the continuous data
        self.feature_mins_ = np.percentile(con[y == 1], self.percentile_range[0], axis=0)
        self.feature_maxs_ = np.percentile(con[y == 1], self.percentile_range[1], axis=0)

        # one-hot encode the categorical data and label the classes with
        if cat is not None:
            ohe = self.categorical_estimator.fit_transform(cat)
            self.in_categorical_ = np.any(ohe[y == 1], axis=0)

    def predict(self, x: ArrayLike, overlay: str = "average") -> np.ndarray:
        """Applies a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            overlay: niche envelope overlap type.
                select from ["average", "intersection", "union"]

        Returns:
            array-like of shape (n_samples,) with model predictions
        """
        overlay = overlay.lower()
        overlay_options = ["average", "intersection", "union"]
        assert overlay in overlay_options, f"overlay must be one of {', '.join(overlay_options)}"

        # format the input data
        con, cat = self._format_covariate_data(x)
        nrows, ncols = x.shape

        # any value within the transformed range is considered within the envelope
        in_range = (con >= self.feature_mins_) * (con <= self.feature_maxs_)

        # map the class locations where the species has been observed
        if cat is not None:
            ohe = self.categorical_estimator.transform(cat)
            should_be_here = ohe[:, self.in_categorical_].any(axis=1).reshape((nrows, 1))
            shouldnt_be_here = (~ohe[:, ~self.in_categorical_].any(axis=1)).reshape((nrows, 1))
            in_range = np.concatenate((in_range, should_be_here, shouldnt_be_here), axis=1)

        # comput envelope based on the overlay method
        if overlay == "average":
            ypred = np.mean(in_range, axis=1, dtype="float32")

        elif overlay == "intersection":
            ypred = np.all(in_range, axis=1).astype("uint8")

        elif overlay == "union":
            ypred = np.any(in_range, axis=1).astype("uint8")

        return ypred

    def fit_predict(
        self, x: ArrayLike, y: ArrayLike, categorical: list = None, labels: list = None, overlay: str = "average"
    ) -> np.ndarray:
        """Trains and applies a model to x/y data.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: column indices indicating which columns are categorical
            labels: Covariate labels. Ignored if x is a pandas DataFrame
            overlay: maxent model transformation type.
                select from ["average", "intersection", "union"]

        Returns:
            array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels)
        return self.predict(x, overlay=overlay)


def maxent_alpha(raw: np.ndarray) -> float:
    """Compute the sum-to-one alpha maxent model parameter.

    Args:
        raw: uncalibrated maxent raw (exponential) model output

    Returns:
        alpha: the output sum-to-one scaling factor
    """
    return -np.log(np.sum(raw))


def maxent_entropy(raw: np.ndarray) -> float:
    """Compute the maxent model entropy score for scaling the logistic output

    Args:
        raw: uncalibrated maxent raw (exponential) model output

    Returns:
        entropy: background distribution entropy score
    """
    scaled = raw / np.sum(raw)
    return -np.sum(scaled * np.log(scaled))


def maxent_raw_transform(engma: np.ndarray) -> np.ndarray:
    """Compute maxent's raw suitability score

    Args:
        engma: calibrated maxent linear model output

    Returns:
        the log-linear raw scores for each sample
    """
    return np.exp(engma)


def maxent_logistic_transform(engma: np.ndarray, entropy: float, tau: float = MaxentConfig.tau) -> np.ndarray:
    """Compute maxent's logistic suitability score

    Args:
        engma: calibrated maxent linear model output
        entropy: the calibrated model entropy score
        tau: the prevalence scaler. lower values indicate rarer species.

    Returns:
        the tau-scaled logistic scores for each sample
    """
    # maxnet's (tau-free) logistic formulation:
    # return 1 / (1 + np.exp(-entropy - engma))
    # use java's formulation instead
    logratio = np.exp(engma) * np.exp(entropy)
    return (tau * logratio) / ((1 - tau) + (tau * logratio))


def maxent_cloglog_transform(engma: np.ndarray, entropy: float) -> np.ndarray:
    """Compute maxent's cumulative log-log suitability score

    Args:
        engma: calibrated maxent linear model output
        entropy: the calibrated model entropy score

    Returns:
        the cloglog scores for each sample
    """
    return 1 - np.exp(-np.exp(engma) * np.exp(entropy))


def format_occurrence_data(y: ArrayLike) -> ArrayLike:
    """Reads input y data and formats it to consistent 1d array dtypes.

    Args:
        y: array-like of shape (n_samples,) or (n_samples, 1)

    Returns:
        formatted uint8 ndarray of shape (n_samples,)

    Raises:
        np.AxisError: an array with 2 or more columns is passed
    """
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if y.ndim > 1:
        if y.shape[1] > 1 or y.ndim > 2:
            raise np.AxisError(f"Multi-column y data passed of shape {y.shape}. Must be 1d or 1 column.")
        y = y.flatten()

    return y.astype("uint8")


def estimate_C_from_betas(beta_multiplier: float) -> float:
    """Convert the maxent-format beta_multiplier to an sklearn-format C regularization parameter.

    Args:
        beta_multiplier: the maxent beta regularization scaler

    Returns:
        a C factor approximating the level of regularization passed to glmnet
    """
    return 2 / (1 - np.exp(-beta_multiplier))
