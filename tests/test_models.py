import numpy as np
import pytest
from numpy.testing import assert_almost_equal
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

from elapid import models
from elapid.utils import load_sample_data

x, y = load_sample_data(name="ariolimax", drop_geometry=True)


def test_MaxentModel_flow():
    model = models.MaxentModel()
    model.fit(x, y)
    ypred = model.predict(x)
    assert len(ypred) == len(y)
    assert ypred.max() <= 1.0
    assert ypred.min() >= 0.0

    proba = model.predict_proba(x)
    assert len(proba) == len(y)
    assert len(proba.shape) == 2
    assert proba.sum(axis=1).sum() == len(y)


def test_MaxentModel_performance():
    model = models.MaxentModel(use_lambdas="last", scorer="roc_auc")
    model.fit(x, y)
    ypred = model.predict(x)
    auc_score = metrics.roc_auc_score(y, ypred)

    # check that model is not nonsense
    assert 0.5 <= auc_score <= 1.0

    # check that full model close to cross-val performance (+- stderr)
    if not isinstance(model.estimator, LogisticRegression):
        assert abs(auc_score - model.estimator.cv_mean_score_[-1]) < 0.1 + model.estimator.cv_standard_error_[-1]


def test_MaxentModel_best_lambdas():
    model = models.MaxentModel(use_lambdas="best", transform="logistic")
    model.fit(x, y)
    ypred = model.predict(x)
    auc_score = metrics.roc_auc_score(y, ypred)
    assert 0.5 <= auc_score <= 1.0
    assert 0.48 < ypred[y == 1].mean() < 0.75


def test_MaxentModel_sample_weight():
    # no class weights
    model = models.MaxentModel(class_weights=None)
    sample_weight = np.ones_like(y, dtype="float32")
    sample_weight[y == 1] = 3
    model.fit(x, y, sample_weight=sample_weight)

    # with class weights
    model = models.MaxentModel(class_weights="balanced")
    model.fit(x, y, sample_weight=sample_weight)
    yw = model.predict(x)
    ywscore = metrics.roc_auc_score(y, yw)

    # unweighted
    model.fit(x, y)
    yu = model.predict(x)
    yuscore = metrics.roc_auc_score(y, yu)
    assert yuscore != ywscore


def test_tau_scaler():
    model = models.MaxentModel(tau=0.5, transform="logistic")
    model.fit(x, y)
    ypred = model.predict(x)
    fit_mean_pt5 = ypred[y == 1].mean()
    assert 0.48 < ypred[y == 1].mean() < 0.75

    model = models.MaxentModel(tau=0.25, transform="logistic")
    model.fit(x, y)
    ypred = model.predict(x)
    fit_mean_pt25 = ypred[y == 1].mean()
    assert fit_mean_pt25 < fit_mean_pt5


def test_MaxentModel_feature_types():
    all = models.MaxentModel(feature_types="lqpht", beta_multiplier=1.5)
    auto = models.MaxentModel(feature_types=["auto"])
    linear = models.MaxentModel(feature_types=["linear"])
    qp = models.MaxentModel(feature_types=["quadratic", "product"], convergence_tolerance=2e-7, beta_lqp=1.5)
    ht = models.MaxentModel(
        feature_types=["hinge", "threshold"],
        beta_threshold=1.5,
        beta_hinge=0.75,
        n_threshold_features=10,
        n_hinge_features=10,
    )

    all.fit(x, y)
    auto.fit(x, y)
    linear.fit(x, y)
    qp.fit(x, y)
    ht.fit(x, y)

    for model in [all, linear, qp, ht, auto]:
        ypred = model.predict(x)
        auc_score = metrics.roc_auc_score(y, ypred)
        assert 0.5 <= auc_score <= 1.0


def test_sklearn_MaxentModel():
    skl = models.MaxentModel(use_sklearn=True)
    ypred = skl.fit_predict(x, y)
    assert ypred.max() <= 1.0
    assert ypred.min() >= 0.0


def test_format_occurrence_data():
    # add a trailing dimension
    yt = np.expand_dims(y.to_numpy(), axis=1)
    model = models.MaxentModel()
    model.fit(x, yt)

    # fail on >2 dims
    with pytest.raises(np.AxisError):
        ytt = np.concatenate((yt, yt), axis=1)
        model.fit(x, ytt)


def test_preprocessor():
    # not-fitted transformer
    pca = PCA()
    model = models.MaxentModel(feature_types="l")
    model.fit(x, y, preprocessor=pca)

    # pre-fitted
    pca = PCA()
    pca.fit(x)
    model = models.MaxentModel(feature_types="l")
    model.fit(x, y, preprocessor=pca)


def test_NicheEnvelopeModel():
    # test the full range of values, ensuring all y=1 points are included
    ne = models.NicheEnvelopeModel(percentile_range=[0, 100], overlay="union")
    ne.fit(x, y)
    union = ne.predict(x)
    ne.set_params(overlay="intersection")
    intersection = ne.predict(x)
    ne.set_params(overlay="average")
    average = ne.predict(x)
    assert np.min([union, intersection, average]) <= 1
    assert np.max([union, intersection, average]) >= 0
    assert y.sum() == union[y == 1].sum()
    assert intersection[y == 1].mean() <= average[y == 1].mean() <= union[y == 1].mean()

    # test a narrower feature data percentile range
    narrow = models.NicheEnvelopeModel(percentile_range=[20, 80], overlay="average")
    narrow.fit(x, y)
    average_narrow = narrow.predict(x)
    assert average_narrow[y == 1].sum() < average[y == 1].sum()

    # test passing numpy arrays
    xt = x.to_numpy()
    ne = models.NicheEnvelopeModel()
    ne.fit_predict(xt, y)

    # test prediction probabilities
    proba = ne.predict_proba(xt)
    assert len(proba) == len(y)
    assert len(proba.shape) == 2
    assert proba.sum(axis=1).sum() == len(y)


def test_EnsembleModel():
    ne = models.NicheEnvelopeModel()
    me = models.MaxentModel()
    ne.fit(x, y)
    me.fit(x, y)

    ypred = ne.predict(x)
    yprob = ne.predict_proba(x)

    ensemble = models.EnsembleModel((ne, me), reducer="mean")
    epred = ensemble.predict(x)
    eprob = ensemble.predict_proba(x)

    assert not np.all(ypred == epred)
    assert eprob.shape == yprob.shape
    assert len(eprob.shape) == 2
    assert eprob.shape[-1] == 2

    ensemble.set_params(reducer="mode")
    mpred = ensemble.predict(x)
    assert not np.all(mpred == epred)
    assert epred.shape == mpred.shape
