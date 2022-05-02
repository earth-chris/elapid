"""SDM model configuration parameters."""
from typing import Tuple, Union


class MaxentConfig:
    # constrain feature min/max ranges
    clamp: bool = True

    # beta regularization drops feature cofficients if they don't improve model
    # performance. these scaler modify feature beta regularization behavior.
    # increase this number to drop more coefficients, decrease to keep more
    # coefficients. `beta_multiplier` scales regularization for all features,
    # but regularization can be tuned by feature type
    beta_multiplier: float = 1.0
    beta_hinge: float = 1.0
    beta_lqp: float = 1.0
    beta_threshold: float = 1.0
    beta_categorical: float = 1.0

    # default maxent feature types to compute
    feature_types: list = ["linear", "hinge", "product"]

    # the number of hinge and threshold features to compute per-covariate
    n_hinge_features: int = 10
    n_threshold_features: int = 10

    # model training metric (from `sklearn.metrics`)
    scorer: str = "roc_auc"

    # species prevalence scalar
    tau: float = 0.5

    # model convergence tolerance threshold
    tolerance: float = 1e-7

    # elasticnet lambda type to use ('best' or 'last')
    use_lambdas: str = "best"
    n_lambdas: int = 100

    # method for weighting presence samples
    weights: Union[str, float] = "balance"


# maxent default regularization parameters (from the maxnet R package)
class RegularizationConfig:
    linear: list = [[0, 10, 30, 100], [1, 1, 0.2, 0.05]]
    quadratic: list = [[0, 10, 17, 30, 100], [1.3, 0.8, 0.5, 0.25, 0.05]]
    product: list = [[0, 10, 17, 30, 100], [2.6, 1.6, 0.9, 0.55, 0.05]]
    hinge: list = [[0, 1], [0.5, 0.5]]
    threshold: list = [[0, 100], [2, 1]]
    categorical: list = [[0, 10, 17], [0.65, 0.5, 0.25]]


class NicheEnvelopeConfig:
    percentile_range: Tuple[float, float] = [2.5, 97.5]
