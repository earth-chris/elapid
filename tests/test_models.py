from sklearn import metrics

from elapid import models
from elapid.utils import load_sample_data

x, y = load_sample_data()


def test_MaxentModel_flow():
    model = models.MaxentModel()
    model.fit(x, y)
    ypred = model.predict(x)
    assert len(ypred) == len(y)
    print(ypred)
    print(ypred.max())
    assert ypred.max() <= 1.0
    assert ypred.min() >= 0.0


def test_MaxentModel_performance():
    model = models.MaxentModel(use_lambdas="last", scorer="roc_auc")
    model.fit(x, y)
    ypred = model.predict(x)
    auc_score = metrics.roc_auc_score(y, ypred)

    # check that model is not nonsense
    assert 0.5 <= auc_score <= 1.0

    # check that full model close to cross-val performance (+- stderr)
    assert abs(auc_score - model.estimator.cv_mean_score_[-1]) < 0.1 + model.estimator.cv_standard_error_[-1]


def test_MaxentModel_best_lambdas():
    model = models.MaxentModel(use_lambdas="best")
    model.fit(x, y)
    ypred = model.predict(x, transform="logistic")
    auc_score = metrics.roc_auc_score(y, ypred)
    assert 0.5 <= auc_score <= 1.0
    assert 0.48 < ypred[y == 1].mean() < 0.52


def test_tau_scaler():
    model = models.MaxentModel(tau=0.5)
    model.fit(x, y)
    ypred = model.predict(x, transform="logistic")
    assert 0.48 < ypred[y == 1].mean() < 0.52

    model = models.MaxentModel(tau=0.25)
    model.fit(x, y)
    ypred = model.predict(x, transform="logistic")
    assert 0.23 < ypred[y == 1].mean() < 0.27


def test_MaxentModel_feature_types():
    all = models.MaxentModel(feature_types="lqpht", beta_multiplier=1.5)
    linear = models.MaxentModel(feature_types=["linear"])
    qp = models.MaxentModel(feature_types=["quadratic", "product"], convergence_tolerance=2e-7, beta_lqp=1.5)
    ht = models.MaxentModel(
        feature_types=["hinge", "threshold"],
        beta_threshold=1.5,
        beta_hinge=0.75,
        n_threshold_features=10,
        n_hinge_features=10,
    )
    auto = models.MaxentModel(feature_types=["auto"])

    for model in [all, linear, qp, ht, auto]:
        model.fit(x, y)
        ypred = model.predict(x)
        auc_score = metrics.roc_auc_score(y, ypred)
        assert 0.5 <= auc_score <= 1.0
