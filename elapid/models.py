"""Classes for training species distribution models."""
from typing import List, Union

import numpy as np
import pandas as pd
from glmnet.logistic import LogitNet
from sklearn.base import BaseEstimator

from elapid import creatures as _features
from elapid.config import MaxentConfig
from elapid.types import ArrayLike, Number, validate_feature_types
from elapid.utils import n_cpus


class MaxentModel(BaseEstimator):
    """Creates a model estimator for Maxent-style species distribution models."""

    feature_types_: list = MaxentConfig.feature_types
    tau_: float = MaxentConfig.tau
    clamp_: bool = MaxentConfig.clamp
    convergence_tolerance_: float = MaxentConfig.tolerance
    beta_multiplier_: float = MaxentConfig.beta_multiplier
    beta_hinge_: float = MaxentConfig.beta_hinge
    beta_lqp_: float = MaxentConfig.beta_lqp
    beta_threshold_: float = MaxentConfig.beta_threshold
    beta_categorical_: float = MaxentConfig.beta_categorical
    n_hinge_features_: int = MaxentConfig.n_hinge_features
    n_threshold_features_: int = MaxentConfig.n_threshold_features
    scorer_: str = MaxentConfig.scorer
    use_lambdas_: str = MaxentConfig.use_lambdas
    n_lambdas_: str = MaxentConfig.n_lambdas
    initialized_: bool = False
    beta_scores_: np.array = None
    entropy_: float = 0.0
    alpha_: float = 0.0
    estimator: BaseEstimator = None
    transformer: _features.MaxentFeatureTransformer = None
    weights_: np.ndarray = None
    regularization_: np.ndarray = None
    lambdas_: np.ndarray = None
    weight_strategy_: Union[str, float] = None
    n_cpus_ = n_cpus

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
        weights: Union[str, float] = MaxentConfig.weights,
        n_cpus: int = n_cpus,
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
            convergence_tolerance: model convergence tolerance level
            use_lambdas: guide for which model lambdas to select (either "best" or "last")
            n_lambdas: number of lamba values to fit models with
            weights: strategy for weighting presence samples.
                pass "balance" to compute the ratio based on sample frequency
                or pass a float for the presence:background weight ratio
            n_cpus: threads to use during model training
        """
        self.feature_types_ = validate_feature_types(feature_types)
        self.tau_ = tau
        self.clamp_ = clamp
        self.convergence_tolerance_ = convergence_tolerance
        self.beta_multiplier_ = beta_multiplier
        self.beta_hinge_ = beta_hinge
        self.beta_lqp_ = beta_lqp
        self.beta_threshold_ = beta_threshold
        self.beta_categorical_ = beta_categorical
        self.n_hinge_features_ = n_hinge_features
        self.n_threshold_features_ = n_threshold_features
        self.n_cpus_ = n_cpus
        self.scorer_ = scorer
        self.use_lambdas_ = use_lambdas
        self.n_lambdas_ = n_lambdas
        self.weight_strategy_ = weights

    def fit(
        self, x: ArrayLike, y: ArrayLike, categorical: List[int] = None, labels: list = None, is_features: bool = False
    ) -> None:
        """Trains a maxent model using a set of covariates and presence/background points.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: indices for which columns are categorical
            labels: covariate labels. ignored if x is a pandas DataFrame
            is_features: specify that x data has been transformed from covariates to features

        Returns:
            None: Updates the model object
        """
        # data pre-processing
        if is_features:
            features = x
        else:
            self.transformer = _features.MaxentFeatureTransformer(
                feature_types=self.feature_types_,
                clamp=self.clamp_,
                n_hinge_features=self.n_hinge_features_,
                n_threshold_features=self.n_threshold_features_,
            )
            features = self.transformer.fit_transform(x, categorical=categorical, labels=labels)

        # compute sample weights
        if self.weight_strategy_ == "balance":
            pbr = len(y) / y.sum()
        else:
            pbr = self.weight_strategy_

        self.weights_ = _features.compute_weights(y, pbr=pbr)

        # set feature regularization parameters
        self.regularization_ = _features.compute_regularization(
            y,
            features,
            feature_labels=self.transformer.feature_names_,
            beta_multiplier=self.beta_multiplier_,
            beta_threshold=self.beta_threshold_,
            beta_hinge=self.beta_hinge_,
            beta_categorical=self.beta_categorical_,
        )

        # get model lambda scores to initialize the glm
        self.lambdas_ = _features.compute_lambdas(y, self.weights_, self.regularization_)

        # model fitting
        self.initialize_model(lambdas=self.lambdas_)
        self.estimator.fit(
            features,
            y,
            sample_weight=self.weights_,
            relative_penalties=self.regularization_,
        )

        if self.use_lambdas_ == "last":
            self.beta_scores_ = self.estimator.coef_path_[0, :, -1]
        elif self.use_lambdas_ == "best":
            self.beta_scores_ = self.estimator.coef_path_[0, :, self.estimator.lambda_max_inx_]

        # maxent-specific transformations
        raw = self.predict(features[y == 0], transform="raw", is_features=True)

        # alpha is a normalizing constant that ensures that f1(z) integrates (sums) to 1
        self.alpha_ = maxent_alpha(raw)

        # the distance from f(z) is considered the relative entropy of f1(z) WRT f(z)
        self.entropy_ = maxent_entropy(raw)

    def predict(self, x: ArrayLike, transform: str = "logistic", is_features: bool = False) -> ArrayLike:
        """Applies a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            transform: maxent model transformation type. select from
                ["raw", "logistic", "cloglog"].
            is_features: flag that x data has already been transformed from covariates to features

        Returns:
            predictions: array-like of shape (n_samples,) with model predictions
        """
        assert self.initialized_, "Model must be fit first"

        # feature transformations
        features = x if is_features else self.transformer.transform(x)

        # apply the model
        engma = np.matmul(features, self.beta_scores_) + self.alpha_

        # scale based on the transform type
        if transform == "raw":
            return np.exp(engma)

        elif transform == "logistic":
            # below is R's maxnet (tau-free) logistic formulation
            # return 1 / (1 + np.exp(-self.entropy_ - raw))
            # use the java formulation instead
            logratio = np.exp(engma) * np.exp(self.entropy_)
            return (self.tau_ * logratio) / ((1 - self.tau_) + (self.tau_ * logratio))

        elif transform == "cloglog":
            # below is R's maxent cloglog formula
            # return 1 - np.exp(0 - np.exp(self.entropy_ - raw))
            # use java again
            return 1 - np.exp(-np.exp(engma) * np.exp(self.entropy_))

    def fit_predict(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categorical: list = None,
        labels: list = None,
        transform: str = "logistic",
        is_features: bool = False,
    ) -> ArrayLike:
        """Trains and applies a model to x/y data.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: column indices indicating which columns are categorical
            labels: Covariate labels. Ignored if x is a pandas DataFrame
            transform: maxent model transformation type. select from
                ["raw", "exponential", "logistic", "cloglog"].
            is_features: specify that x data has already been transformed from covariates to features

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels)
        predictions = self.predict(x, transform=transform, is_features=is_features)

        return predictions

    def initialize_model(
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

        Returns:
            None. updates the self.estimator with an sklearn-style model estimator
        """
        self.estimator = LogitNet(
            alpha=alpha,
            lambda_path=lambdas,
            standardize=standardize,
            fit_intercept=fit_intercept,
            scoring=self.scorer_,
            n_jobs=self.n_cpus_,
            tol=self.convergence_tolerance_,
        )

        self.initialized_ = True


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
