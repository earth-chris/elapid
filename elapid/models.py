"""Classes for training species distribution models."""

import numpy as np
import pandas as pd
from glmnet.logistic import LogitNet

from elapid import features as _features
from elapid.utils import MAXENT_DEFAULTS, _ncpus


class MaxentModel(object):
    """Creates a model estimator for Maxent-style species distribution models.

    Attributes:
        feature_types_: List of maxent feature types to compute
        tau_: Float species prevalence scalar
        clamp_: Boolean for whether to constrain feature min/max ranges
        convergence_tolerance_: Float model convergence tolerance threshold
        beta_multiplier_: Float scalar for all feature beta values
        beta_hinge_: Float scalar for hinge beta values
        beta_lqp_: Float scalar for linear/quadratic/product beta values
        beta_threshold_: Float scalar for threshold beta values
        beta_categorical_: Float scalar for categorical beta values
        n_hinge_features_: Int number of hinge features to fit
        n_threshold_features_: Int number of threshold features to fit
        n_cpus_: Int of cpu threads ot use during model training
        scorer_: Str of model training metric (from `sklearn.metrics`)
        use_lambdas_: Str of lambda type to use (in `['best', 'last']`)
        initialized_: Bool tracking model fit status
        beta_scores_: Array of trained model betas
        entropy_: Float of trained model entropy score
        alpha_: Float of trained model alpha score 0.0
        estimator: Object of trained model estimator
        transformer: MaxentFeatureTransformer object
    """

    def __init__(
        self,
        feature_types=MAXENT_DEFAULTS["feature_types"],
        tau=MAXENT_DEFAULTS["tau"],
        clamp=MAXENT_DEFAULTS["clamp"],
        scorer=MAXENT_DEFAULTS["scorer"],
        beta_multiplier=MAXENT_DEFAULTS["beta_multiplier"],
        beta_lqp=MAXENT_DEFAULTS["beta_lqp"],
        beta_hinge=MAXENT_DEFAULTS["beta_hinge"],
        beta_threshold=MAXENT_DEFAULTS["beta_lqp"],
        beta_categorical=MAXENT_DEFAULTS["beta_categorical"],
        n_hinge_features=MAXENT_DEFAULTS["n_hinge_features"],
        n_threshold_features=MAXENT_DEFAULTS["n_threshold_features"],
        convergence_tolerance=MAXENT_DEFAULTS["tolerance"],
        use_lambdas=MAXENT_DEFAULTS["use_lambdas"],
        n_cpus=_ncpus,
    ):
        """Instatiate a maxent model object.

        Args:
            feature_types: List or string of maxent feature types to fit. must be in
                ["linear", "quadratic", "product", "hinge", "threshold", "auto"]
                or string "lqphta"
            tau: Float of the maxent tau (prevalence) value for scaling logistic output
            clamp: Boolean of whether to clamp features during inference
            scorer: Str sklearn scoring function for model training
            beta_multiplier: Float scalar for all regularization parameters,
                where higher values exclude more features
            beta_lqp: Float scalar for linear, quadratic and product feature regularization parameters
            beta_hinge: Float scalar for hinge feature regularization parameters
            beta_threshold: Float scalar for threshold feature regularization parameters
            beta_categorical: Float scalar for categorical feature regularization parameters
            convergence_tolerance: Float scalar for the model convergence tolerance level
            use_lambdas: guide for which model lambdas to select,
                from options ["best", "last"]
            n_cpus: Int of cpu threads to use during model training

        Returns:
            self: a MaxentModel object
        """
        self.feature_types_ = _features.validate_feature_types(feature_types)
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

        self.initialized_ = False
        self.beta_scores_ = None
        self.entropy_ = 0.0
        self.alpha_ = 0.0
        self.estimator = None
        self.transformer = None

    def fit(self, x, y, categorical=None, labels=None, is_features=False):
        """Trains a maxent model using a set of covariates and presence/background points.

        Args:
            x: Array-like of shape (n_samples, n_features) with covariate data
            y: Array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: Array-like of column indices for which columns are categorical
            labels: List of covariate labels. Ignored if x is a pandas dataframe
            is_features: Boolean specifying the x data has been transformed from covariates to features

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

        weights = _features.compute_weights(y)
        regularization = _features.compute_regularization(features, y)
        lambdas = _features.compute_lambdas(y, weights, regularization)

        # model fitting
        self.initialize_model(lambdas=lambdas)

        self.estimator.fit(
            features,
            y,
            sample_weight=weights,
            relative_penalties=regularization,
        )

        if self.use_lambdas_ == "last":
            self.beta_scores_ = self.estimator.coef_path_[0, :, -1]
        elif self.use_lambdas_ == "best":
            self.beta_scores_ = self.estimator.coef_path_[0, :, self.estimator.lambda_best_inx_]

        # maxent specific transformations
        rr = self.predict(features[y == 0], transform="exponential", is_features=True)
        raw = rr / np.sum(rr)
        self.entropy_ = -np.sum(raw * np.log(raw))
        self.alpha_ = -np.log(np.sum(rr))

    def predict(self, x, transform="logistic", is_features=False):
        """Applies a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: Array-like of shape (n_samples, n_features) with covariate data
            transform: Str maxent model transformation type. Select from
                ["raw", "exponential", "logistic", "cloglog"].
            is_features: Boolean specifying the x data has already been transformed from covariates to features

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        assert self.initialized_, "Model must be fit first"

        # feature transformations
        if is_features:
            features = x
        else:
            features = self.transformer.transform(x)

        # apply the model
        link = np.matmul(features, self.beta_scores_) + self.alpha_

        if transform == "raw":
            return link
        elif transform == "exponential":
            return np.exp(link)
        elif transform == "logistic":
            return 1 / (1 + np.exp(-self.entropy_ - link))
        elif transform == "cloglog":
            return 1 - np.exp(0 - np.exp(self.entropy_ + link))

    def fit_predict(self, x, y, categorical=None, labels=None, transform="logistic", is_features=False):
        """Trains and applies a model to x/y data.

        Args:
            x: Array-like of shape (n_samples, n_features) with covariate data
            y: Array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: Array-like of column indices for which columns are categorical
            labels: List of covariate labels. Ignored if x is a pandas dataframe
            transform: Str of maxent model transformation type. Select from
                ["raw", "exponential", "logistic", "cloglog"].
            is_features: Boolean specifying the x data has already been transformed from covariates to features

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels)
        predictions = self.predict(x, transform=transform, is_features=is_features)

        return predictions

    def initialize_model(
        self,
        lambdas,
        alpha=1,
        standardize=False,
        fit_intercept=True,
    ):
        """Creates the Logistic Regression with elastic net penalty model object.

        Args:
            lambdas: Array of model lambda values. Get from features.compute_lambdas()
            alpha: Float elasticnet mixing parameter. alpha=1 for lasso, alpha=0 for ridge
            standardize: Boolean to specify coefficient normalization
            fit_intercept: Boolean flag to include an intercept parameter

        Returns:
            None: Updates the self.estimator with an sklearn-style model estimator
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
