"""SDM model configuration parameters."""


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
    n_hinge_features: int = 30
    n_threshold_features: int = 20

    # model training metric (from `sklearn.metrics`)
    scorer: str = "roc_auc"

    # species prevalence scalar
    tau: float = 0.5

    # model convergence tolerance threshold
    tolerance: float = 1e-7

    # elasticnet lambda type to use ('best' or 'last')
    use_lambdas: str = "best"
