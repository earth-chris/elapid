"""Classes for training species distribution models."""

import numpy as np
import pandas as pd
from glmnet.logistic import LogitNet

from elapid import features as _features
from elapid.utils import MAXENT_DEFAULTS, _ncpus


class MaxentModel(object):
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
        """
        Creates a model estimator for Maxent-style species distribution models.

        :param feature_types: list or string of maxent feature types to fit. must be in ["linear", "quadratic", "product", "hinge", "threshold", "auto"] or string "lqphta"
        :param tau: float of the maxent tau (prevalence) value for scaling logistic output
        :param clamp: bool of whether to clamp features during inference
        :param scorer: the sklearn scoring function for model training
        :param beta_multiplier: scalar for all regularization parameters, where higher values exclude more features
        :param beta_lqp: scalar for linear, quadratic and product feature regularization parameters
        :param beta_hinge: scalar for hinge feature regularization parameters
        :param beta_threshold: scalar for threshold feature regularization parameters
        :param beta_categorical: scalar for categorical feature regularization parameters
        :param convergence_tolerance: scalar for the model convergence tolerance level
        :param use_lambdas: guide for which model lambdas to select, from options ["best", "last"]
        :param n_cpus: integer number of cpu threads to use during model training
        :returns: none
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
        """
        Trains a maxent model using a set of covariates and presence/background points.

        :param x: array-like of shape (n_samples, n_features) with covariate data
        :param y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        :param categorical: either a 2D a array-like akin to "x", or a 1d array-like of column indices indicating which columns are categorical
        :param labels: covariate labels. Ignored if x is a pandas dataframe.
        :param is_features: boolean to specify that the x data has already been transformed from covariates to features
        :returns: none. updates the model object
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
        """
        Applies a maxent model to a set of covariates or features. Requires that a model has been fit.

        :param x: array-like of shape (n_samples, n_features) with covariate data
        :param transform: the maxent model transformation type. Select from ["raw", "exponential", "logistic", "cloglog"].
        :param is_features: boolean to specify that the x data has already been transformed from covariates to features
        :returns predictions: array-like of shape (n_samples,) with model predictions
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
        """
        :param x: array-like of shape (n_samples, n_features) with covariate data
        :param y: array-like of shape (n_samples,) with binary presence/background (1/0) values
        :param categorical: either a 2D a array-like akin to "x", or a 1d array-like of column indices indicating which columns are categorical
        :param labels: covariate labels. Ignored if x is a pandas dataframe.
        :param transform: the maxent model transformation type. Select from ["raw", "exponential", "logistic", "cloglog"].
        :param is_features: boolean to specify that the x data has already been transformed from covariates to features
        :returns predictions: array-like of shape (n_samples,) with model predictions
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
        """
        Creates the Logistic Regression with elastic net penalty model object.

        :param lambdas: the lambda values for the model. Get from features.compute_lambdas()
        :param alpha: the elasticnet mixing parameter. alpha=1 for lasso, alpha=0 for ridge
        :param standardize: boolean flag to specify coefficient normalization
        :param fit_intercept: boolean flag to include an intercept parameter
        :return none: updates the self.estimator with an sklearn model estimator
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
