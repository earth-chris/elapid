"""Model classes for species distribution modeling"""

import numpy as np
import pandas as pd

from elapid import features as _features
from elapid.utils import _validate_feature_types

MAXENT_DEFAULTS = {
    "clamp": True,
    "beta_multiplier": 1.0,
    "beta_hinge": 1.0,
    "beta_lqp": 1.0,
    "beta_threshold": 1.0,
    "beta_categorical": 1.0,
    "feature_types": ["linear", "hinge", "product"],
    "n_hinge_features": 50,
    "n_threshold_features": 50,
    "tau": 0.5,
}


class Maxent(object):
    def __init__(
        self,
        feature_types=MAXENT_DEFAULTS["feature_types"],
        tau=MAXENT_DEFAULTS["tau"],
        clamp=MAXENT_DEFAULTS["clamp"],
        beta_multiplier=MAXENT_DEFAULTS["beta_multiplier"],
        beta_lqp=MAXENT_DEFAULTS["beta_lqp"],
        beta_hinge=MAXENT_DEFAULTS["beta_hinge"],
        beta_threshold=MAXENT_DEFAULTS["beta_lqp"],
        beta_categorical=MAXENT_DEFAULTS["beta_categorical"],
        n_hinge_features=MAXENT_DEFAULTS["n_hinge_features"],
        n_threshold_features=MAXENT_DEFAULTS["n_threshold_features"],
    ):
        """
        Creates a model estimator for Maxent-style species distribution models.
        """
        self.feature_types_ = _validate_feature_types(feature_types)
        self.tau_ = tau
        self.clamp_ = clamp
        self.beta_multiplier_ = beta_multiplier
        self.beta_hinge_ = beta_hinge
        self.beta_lqp_ = beta_lqp
        self.beta_threshold_ = beta_threshold
        self.beta_categorical_ = beta_categorical
        self.n_hinge_features_ = n_hinge_features
        self.n_threshold_features_ = n_threshold_features

    def compute_features(self, df):
        """
        Transforms input data into the features used for model training.

        :param df: a pandas dataframe encoded with numeric and categorical covariates
        :param features: a dataframe with the feature transformations applied to each column
        """
        categorical = df.select_dtypes(include="category")
        continuous = df.select_dtypes(exclude="category")

        categorical_covariates = list(categorical.columns)
        continuous_covariates = list(continuous.columns)

        feature_list = list()
        for covariate in categorical_covariates:

            series = categorical[covariate]
            classes = list(series.unique())
            classes.sort()
            feature_names = [f"{covariate}_class_{clas}" for clas in classes]
            one_hot_encoded = _features.categorical(series.to_numpy())
            feature_df = pd.DataFrame(one_hot_encoded, columns=feature_names)
            feature_list.append(feature_df)

        for covariate in continuous_covariates:
            series = continuous[covariate]

            if "linear" in self.feature_types_:

                feature_list.append(series.rename(f"{covariate}_linear"))

            if "quadratic" in self.feature_types_:

                feature_list.append((series ** 2).rename(f"{covariate}_squared"))

            if "hinge" in self.feature_types_:

                feature_names = [f"{covariate}_hinge_{i+1:03d}" for i in range((self.n_hinge_features_ - 1) * 2)]
                hinges = _features.hinge(series.to_numpy(), n_hinges=self.n_hinge_features_)
                feature_df = pd.DataFrame(hinges, columns=feature_names)
                feature_list.append(feature_df)

            if "threshold" in self.feature_types_:

                feature_names = [f"{covariate}_threshold_{i+1:03d}" for i in range(self.n_threshold_features_ - 2)]
                thresholds = _features.threshold(series.to_numpy(), n_thresholds=self.n_threshold_features_)
                feature_df = pd.DataFrame(thresholds, columns=feature_names)
                feature_list.append(feature_df)

            if "product" in self.feature_types_:

                idx_cov = continuous_covariates.index(covariate)
                for i in range(idx_cov, len(continuous_covariates) - 1):
                    feature_name = f"{covariate}_x_{continuous_covariates[i+1]}"
                    product = series * continuous[continuous_covariates[i + 1]]
                    feature_df = pd.DataFrame(product, columns=[feature_name])
                    feature_list.append(feature_df)

        features = pd.concat(feature_list, axis=1)
        return features

    def compute_regularization(self, f, y):
        """
        Applies variable regularization to all feature data.

        :param f: pandas dataframe with feature transformations applied
        :param y: pandas series with binary presence/background (1/0) values
        :returns reg: a numpy array with per-feature regularization parameters
        """

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
                multiplier = self.beta_lqp_
            elif "_hinge" in feature:
                freg = [[0, 1], [0.5, 0.5]]
                multiplier = self.beta_hinge_
            elif "_threshold" in feature:
                freg = [[0, 100], [2, 1]]
                multiplier = self.beta_threshold_
            elif "_class" in feature:
                freg = [[0, 10, 17], [0.65, 0.5, 0.25]]
                multiplier = self.beta_categorical_

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

        # and scale it
        max_reg *= self.beta_multiplier_

        return max_reg

    def compute_lambdas(self, y, weights, reg, n_lambda=200):
        """
        Computes lambda parameter values for elastic lasso fits.

        :param y: pandas series with binary presence/background (1/0) values
        :param weights: per-sample model weights
        :param reg: per-feature regularization coefficients
        :param n_lambda: the number of lambda values to estimate
        :return lambdas: a numpy array of lambda scores of length n_lambda
        """
        n_presence = np.sum(y)
        mean_regularization = np.mean(reg)
        total_weight = np.sum(weights)
        seed_range = np.linspace(4, 0, n_lambda)
        lambdas = 10 ** (seed_range) * mean_regularization * (n_presence / total_weight)

        return lambdas

    def compute_weights(self, y):
        """
        Uses Maxent's weight formulation to compute per-sample model weights.

        :param y: pandas series or 1d array with binary presence/background (1/0) values
        """
        weights = np.array(y + (1 - y) * 100)

        return weights
