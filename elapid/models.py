"""Classes for training species distribution models."""
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as scistats
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from elapid.config import EnsembleConfig, MaxentConfig, NicheEnvelopeConfig
from elapid.features import (
    CategoricalTransformer,
    FeaturesMixin,
    MaxentFeatureTransformer,
    compute_lambdas,
    compute_regularization,
    compute_weights,
)
from elapid.types import ArrayLike, validate_feature_types
from elapid.utils import NCPUS, make_band_labels, square_factor

# handle windows systems without functioning gfortran compilers
FORCE_SKLEARN = False
try:
    from glmnet.logistic import LogitNet

except ModuleNotFoundError:
    FORCE_SKLEARN = True


class SDMMixin:
    """Mixin class for SDM classifiers."""

    _estimator_type = "classifier"
    classes_ = [0, 1]

    def score(self, x: ArrayLike, y: ArrayLike, sample_weight: ArrayLike = None) -> float:
        """Return the mean AUC score on the given test data and labels.

        Args:
            x: test samples. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)

        Returns:
            AUC score of `self.predict(x)` w.r.t. `y`.
        """
        return roc_auc_score(y, self.predict(x), sample_weight=sample_weight)

    def _more_tags(self):
        return {"requires_y": True}

    def permutation_importance_scores(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        n_repeats: int = 10,
        n_jobs: int = -1,
    ) -> np.ndarray:
        """Compute a generic feature importance score by modifying feature values
            and computing the relative change in model performance.

        Permutation importance measures how much a model score decreases when a
            single feature value is randomly shuffled. This score doesn't reflect
            the intrinsic predictive value of a feature by itself, but how important
            feature is for a particular model.

        Args:
            x: test samples. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)
            n_repeats: number of permutation iterations.
            n_jobs: number of parallel compute tasks. set to -1 for all cpus.

        Returns:
            importances: an array of shape (n_features, n_repeats).
        """
        pi = permutation_importance(self, x, y, sample_weight=sample_weight, n_jobs=n_jobs, n_repeats=n_repeats)

        return pi.importances

    def permutation_importance_plot(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        n_repeats: int = 10,
        labels: list = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Create a box plot with bootstrapped permutation importance scores for each covariate.

        Permutation importance measures how much a model score decreases when a
            single feature value is randomly shuffled. This score doesn't reflect
            the intrinsic predictive value of a feature by itself, but how important
            feature is for a particular model.

        It is often appropriate to compute permuation importance scores using both
            training and validation sets. Large differences between the two may
            indicate overfitting.

        This implementation does not necessarily match the implementation in Maxent.
            These scores may be difficult to interpret if there is a high degree
            of covariance between features or if the model estimator includes any
            non-linear feature transformations (e.g. 'hinge' features).

        Reference:
            https://scikit-learn.org/stable/modules/permutation_importance.html

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
            y: presence/absence labels. array-like of shape (n_samples,).
            sample_weight: array-like of shape (n_samples,)
            n_repeats: number of permutation iterations.
            labels: list of band names to label the plots.
            **kwargs: additional arguments to pass to `plt.subplots()`.

        Returns:
            fig, ax: matplotlib subplot figure and axes.
        """
        importance = self.permutation_importance_scores(x, y, sample_weight=sample_weight, n_repeats=n_repeats)
        rank_order = importance.mean(axis=-1).argsort()

        if labels is None:
            try:
                labels = x.columns.tolist()
            except AttributeError:
                labels = make_band_labels(x.shape[-1])
        labels = [labels[idx] for idx in rank_order]

        plot_defaults = {"dpi": 150, "figsize": (5, 4)}
        plot_defaults.update(**kwargs)
        fig, ax = plt.subplots(**plot_defaults)
        ax.boxplot(
            importance[rank_order].T,
            vert=False,
            labels=labels,
        )
        fig.tight_layout()

        return fig, ax

    def partial_dependence_scores(
        self,
        x: ArrayLike,
        percentiles: tuple = (0.025, 0.975),
        n_bins: int = 100,
        categorical_features: tuple = [None],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute partial dependence scores for each feature.

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
                used to constrain the range of values to evaluate for each feature.
            percentiles: lower and upper percentiles used to set the range to plot.
            n_bins: the number of bins spanning the lower-upper percentile range.
            categorical_features: a 0-based index of which features are categorical.

        Returns:
            bins, mean, stdv: the binned feature values and the mean/stdv of responses.
        """
        ncols = x.shape[1]
        mean = np.zeros((ncols, n_bins))
        stdv = np.zeros_like(mean)
        bins = np.zeros_like(mean)

        for idx in range(ncols):
            if idx in categorical_features:
                continue
            pd = partial_dependence(
                self,
                x,
                [idx],
                percentiles=percentiles,
                grid_resolution=n_bins,
                kind="individual",
            )
            mean[idx] = pd["individual"][0].mean(axis=0)
            stdv[idx] = pd["individual"][0].std(axis=0)
            bins[idx] = pd["values"][0]

        return bins, mean, stdv

    def partial_dependence_plot(
        self,
        x: ArrayLike,
        percentiles: tuple = (0.025, 0.975),
        n_bins: int = 50,
        categorical_features: tuple = None,
        labels: list = None,
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot the response of an estimator across the range of feature values.

        Args:
            x: evaluation features. array-like of shape (n_samples, n_features).
                used to constrain the range of values to evaluate for each feature.
            percentiles: lower and upper percentiles used to set the range to plot.
            n_bins: the number of bins spanning the lower-upper percentile range.
            categorical_features: a 0-based index of which features are categorical.
            labels: list of band names to label the plots.
            **kwargs: additional arguments to pass to `plt.subplots()`.

        Returns:
            fig, ax: matplotlib subplot figure and axes.
        """
        # skip categorical features for now
        if categorical_features is None:
            try:
                categorical_features = self.transformer.categorical_ or [None]
            except AttributeError:
                categorical_features = [None]

        bins, mean, stdv = self.partial_dependence_scores(
            x, percentiles=percentiles, n_bins=n_bins, categorical_features=categorical_features
        )

        if labels is None:
            try:
                labels = x.columns.tolist()
            except AttributeError:
                labels = make_band_labels(x.shape[-1])

        ncols = x.shape[1]
        figx = int(np.ceil(np.sqrt(ncols)))
        figy = int(np.ceil(ncols / figx))
        fig, ax = plt.subplots(figx, figy, **kwargs)
        ax = ax.flatten()

        for idx in range(ncols):
            ax[idx].fill_between(bins[idx], mean[idx] - stdv[idx], mean[idx] + stdv[idx], alpha=0.25)
            ax[idx].plot(bins[idx], mean[idx])
            ax[idx].set_title(labels[idx])

        # turn off empty plots
        for axi in ax:
            if not axi.lines:
                axi.set_visible(False)

        fig.tight_layout()

        return fig, ax


class MaxentModel(BaseEstimator, SDMMixin):
    """Model estimator for Maxent-style species distribution models."""

    def __init__(
        self,
        feature_types: Union[list, str] = MaxentConfig.feature_types,
        tau: float = MaxentConfig.tau,
        transform: float = MaxentConfig.transform,
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
            transform: maxent model transformation type. select from
                ["raw", "logistic", "cloglog"].
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
                pass "balanced" to compute the ratio based on sample frequency
                or pass a float for the presence:background weight ratio
                the R `maxnet` package uses a value of 100 as default.
                set to None to ignore.
            n_cpus: threads to use during model training
            use_sklearn: force using `sklearn` for fitting logistic regression.
                turned off by default to use `glmnet` for fitting.
                this feature was turned on to support Windows users
                who install the package without a fortran compiler.
        """
        self.feature_types = feature_types
        self.tau = tau
        self.transform = transform
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

        # computed during model fitting
        self.initialized_ = False
        self.estimator = None
        self.preprocessor = None
        self.transformer = None
        self.regularization_ = None
        self.lambdas_ = None
        self.beta_scores_ = None
        self.entropy_ = 0.0
        self.alpha_ = 0.0

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike = None,
        categorical: List[int] = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
    ) -> None:
        """Trains a maxent model using a set of covariates and presence/background points.

        Args:
            x: array of shape (n_samples, n_features) with covariate data
            y: array of shape (n_samples,) with binary presence/background (1/0) values
            sample_weight: array of weights assigned to each sample with shape (n_samples,).
                this is modified by the `class_weights` model parameter unless
                you set `class_weights=None`.
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
        self.feature_types = validate_feature_types(self.feature_types)
        self.transformer = MaxentFeatureTransformer(
            feature_types=self.feature_types,
            clamp=self.clamp,
            n_hinge_features=self.n_hinge_features,
            n_threshold_features=self.n_threshold_features,
        )
        features = self.transformer.fit_transform(x, categorical=categorical, labels=labels)
        feature_labels = self.transformer.feature_names_

        # compute class weights
        if self.class_weights is not None:
            pbr = len(y) / y.sum() if self.class_weights == "balanced" else self.class_weights
            class_weight = compute_weights(y, pbr=pbr)

            # scale the sample weight
            if sample_weight is None:
                sample_weight = class_weight
            else:
                sample_weight *= class_weight

        # model fitting with sklearn
        if self.use_sklearn:
            C = estimate_C_from_betas(self.beta_multiplier)
            self.initialize_sklearn_model(C)
            self.estimator.fit(features, y, sample_weight=sample_weight)
            self.beta_scores_ = self.estimator.coef_[0]

        # model fitting with glmnet
        else:
            # set feature regularization parameters
            self.regularization_ = compute_regularization(
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
            self.lambdas_ = compute_lambdas(y, sample_weight, self.regularization_, n_lambdas=self.n_lambdas)

            # model fitting
            self.initialize_glmnet_model(lambdas=self.lambdas_)
            self.estimator.fit(
                features,
                y,
                sample_weight=sample_weight,
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
        class_transform = self.get_params()["transform"]
        self.set_params(transform="raw")
        raw = self.predict(x[y == 0])
        self.set_params(transform=class_transform)

        # alpha is a normalizing constant that ensures that f1(z) integrates (sums) to 1
        self.alpha_ = maxent_alpha(raw)

        # the distance from f(z) is the relative entropy of f1(z) WRT f(z)
        self.entropy_ = maxent_entropy(raw)

        return self

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Apply a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

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
        if self.transform == "raw":
            return maxent_raw_transform(engma)

        elif self.transform == "logistic":
            return maxent_logistic_transform(engma, self.entropy_, self.tau)

        elif self.transform == "cloglog":
            return maxent_cloglog_transform(engma, self.entropy_)

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Compute prediction probability scores for the 0/1 classes.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            predictions: array-like of shape (n_samples, 2) with model predictions
        """
        ypred = self.predict(x).reshape(-1, 1)
        predictions = np.hstack((1 - ypred, ypred))

        return predictions

    def fit_predict(
        self,
        x: ArrayLike,
        y: ArrayLike,
        categorical: list = None,
        labels: list = None,
        preprocessor: BaseEstimator = None,
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

        Returns:
            predictions: Array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels, preprocessor=preprocessor)
        predictions = self.predict(x)

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


class NicheEnvelopeModel(BaseEstimator, SDMMixin, FeaturesMixin):
    """Model estimator for niche envelope-style models."""

    def __init__(
        self,
        percentile_range: Tuple[float, float] = NicheEnvelopeConfig.percentile_range,
        overlay: str = NicheEnvelopeConfig.overlay,
    ):
        """Create a niche envelope model estimator.

        Args:
            percentile_range: covariate values within this range are flagged as suitable habitat
                using a narrow range like [10, 90] drops more areas from suitability maps
                while [0, 100] creates an envelope around the full range of observed
                covariates at all y==1 locations.
            overlay: niche envelope overlap type.
                select from ["average", "intersection", "union"]
        """
        self.percentile_range = percentile_range
        self.overlay = overlay
        self.feature_mins_ = None
        self.feature_maxs_ = None
        self.categorical_estimator = None
        self.categorical_ = None
        self.continuous_ = None
        self.categorical_pd_ = None
        self.continuous_pd_ = None
        self.in_categorical_ = None

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
            self.categorical_estimator = CategoricalTransformer()
            ohe = self.categorical_estimator.fit_transform(cat)
            self.in_categorical_ = np.any(ohe[y == 1], axis=0)

        return self

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Applies a model to a set of covariates or features. Requires that a model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            array-like of shape (n_samples,) with model predictions
        """
        overlay = self.overlay.lower()
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

    def predict_proba(self, x: ArrayLike) -> ArrayLike:
        """Compute prediction probability scores for the 0/1 classes.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            predictions: array-like of shape (n_samples, 2) with model predictions
        """
        ypred = self.predict(x).reshape(-1, 1)
        predictions = np.hstack((1 - ypred, ypred))

        return predictions

    def fit_predict(self, x: ArrayLike, y: ArrayLike, categorical: list = None, labels: list = None) -> np.ndarray:
        """Trains and applies a model to x/y data.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data
            y: array-like of shape (n_samples,) with binary presence/background (1/0) values
            categorical: column indices indicating which columns are categorical
            labels: Covariate labels. Ignored if x is a pandas DataFrame

        Returns:
            array-like of shape (n_samples,) with model predictions
        """
        self.fit(x, y, categorical=categorical, labels=labels)
        return self.predict(x)


class EnsembleModel(BaseEstimator, SDMMixin):
    """Barebones estimator for ensembling multiple model predictions."""

    models: list = None
    reducer: str = None

    def __init__(self, models: List[BaseEstimator], reducer: str = EnsembleConfig.reducer):
        """Create a model ensemble from a set of trained models.

        Args:
            models: iterable of models with `.predict()` and/or `.predict_proba()` methods
            reducer: method for reducing/ensembling each model's predictions.
                select from ['mean', 'median', 'mode']
        """
        self.models = models
        self.reducer = reducer

    def reduce(self, preds: List[np.ndarray]) -> np.ndarray:
        """Reduce multiple model predictions into ensemble prediction/probability scores.

        Args:
            preds: list of model predictions from .predict() or .predict_proba()

        Returns:
            array-like of shape (n_samples, n_classes) with model predictions
        """
        reducer = self.reducer.lower()

        if reducer == "mean":
            reduced = np.nanmean(preds, axis=0)

        elif reducer == "median":
            reduced = np.nanmedian(preds, axis=0)

        elif reducer == "mode":
            try:
                summary = scistats.mode(preds, axis=0, nan_policy="omit", keepdims=False)
                reduced = summary.mode
            except TypeError:
                summary = scistats.mode(preds, axis=0, nan_policy="omit")
                reduced = np.squeeze(summary.mode)

        return reduced

    def predict(self, x: ArrayLike) -> np.ndarray:
        """Applies models to a set of covariates or features. Requires each model has been fit.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            array-like of shape (n_samples,) with model predictions
        """
        preds = [model.predict(x) for model in self.models]
        ensemble = self.reduce(preds)
        return ensemble

    def predict_proba(self, x: ArrayLike) -> np.ndarray:
        """Compute prediction probability scores for each class.

        Args:
            x: array-like of shape (n_samples, n_features) with covariate data

        Returns:
            array-like of shape (n_samples, n_classes) with model predictions
        """
        probas = [model.predict_proba(x) for model in self.models]
        ensemble = self.reduce(probas)
        return ensemble


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
