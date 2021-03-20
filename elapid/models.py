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
    "feature_types": ["linear", "hinge", "product"],
    "n_hinge_features": 30,
    "n_threshold_features": 30,
    "tau": 0.5,
}


class Maxent(object):
    def __init__(
        self,
        feature_types=MAXENT_DEFAULTS["feature_types"],
        tau=MAXENT_DEFAULTS["tau"],
        clamp=MAXENT_DEFAULTS["clamp"],
        beta_multiplier=MAXENT_DEFAULTS["beta_multiplier"],
        beta_hinge=MAXENT_DEFAULTS["beta_hinge"],
        beta_lqp=MAXENT_DEFAULTS["beta_lqp"],
        beta_threshold=MAXENT_DEFAULTS["beta_lqp"],
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

        features = pd.concat(feature_list, axis=1)
        return features
