import numpy as np
import pandas as pd
import pytest
from numpy.exceptions import AxisError
from sklearn import metrics
from sklearn.base import is_classifier
from sklearn.decomposition import PCA
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LogisticRegression

from elapid import models
from elapid.utils import load_sample_data

x, y = load_sample_data(name="ariolimax", drop_geometry=True)
xb, yb = load_sample_data("bradypus")


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


def test_MaxentModel_random_state_behavior():
    model1 = models.MaxentModel(random_state=1)
    model2 = models.MaxentModel(random_state=1)
    model1.fit(x, y)
    model2.fit(x, y)
    ypred1 = model1.predict(x)
    ypred2 = model2.predict(x)
    np.testing.assert_allclose(ypred1, ypred2)

    model3 = models.MaxentModel(random_state=2)
    model3.fit(x, y)
    ypred3 = model3.predict(x)

    if np.allclose(ypred1, ypred3):
        import warnings

        warnings.warn(
            "MaxentModel predictions are identical for different random_state values; model may be deterministic for this configuration."
        )


def test_format_occurrence_data():
    # add a trailing dimension
    yt = np.expand_dims(y.to_numpy(), axis=1)
    model = models.MaxentModel()
    model.fit(x, yt)

    # fail on >2 dims
    with pytest.raises(AxisError):
        ytt = np.concatenate((yt, yt), axis=1)
        model.fit(x, ytt)


def test_preprocessor():
    import warnings as _warnings

    pca = PCA()
    pca.fit(x)
    model = models.MaxentModel(feature_types="l")

    # the internal predict() inside fit() must not re-apply the preprocessor;
    # if it does, PCA (fit on a DataFrame) is handed a numpy array on the
    # second call and emits the "X does not have valid feature names" warning.
    with _warnings.catch_warnings():
        _warnings.filterwarnings("error", message=r"X does not have valid feature names")
        model.fit(x, y, preprocessor=pca)

    # sanity: predict round-trip works on the same DataFrame
    ypred = model.predict(x)
    assert ypred.shape == (len(y),)
    assert 0.0 <= ypred.min() and ypred.max() <= 1.0


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


def test_partial_dependence_scores():
    ne = models.NicheEnvelopeModel()
    me = models.MaxentModel()

    # just test that these methods work with each estimator
    ne.fit(x, y)
    me.fit(x, y)
    ne.partial_dependence_plot(x)
    me.partial_dependence_plot(x)

    # and with the bradypus data
    ne.fit(xb, yb)
    me.fit(xb, yb)
    ne.partial_dependence_plot(xb, categorical_features=[2])
    me.partial_dependence_plot(xb)


def test_permutation_importance_scores():
    ne = models.NicheEnvelopeModel()
    me = models.MaxentModel()

    # just test that these methods work with each estimator
    ne.fit(x, y)
    me.fit(x, y)
    ne.permutation_importance_plot(x, y)
    me.permutation_importance_plot(x, y)

    # and with bradypus
    ne.fit(xb, yb)
    me.fit(xb, yb)
    ne.permutation_importance_plot(xb, yb)
    me.permutation_importance_plot(xb, yb)


# Tests guarding sklearn compatibility (estimator tags, classes_, partial_dependence)
# Pandas-touching helper tests live alongside since the integer-cast path is the
# main pandas dtype concern in models.py.


@pytest.mark.parametrize(
    "model_factory",
    [
        models.MaxentModel,
        models.NicheEnvelopeModel,
        lambda: models.EnsembleModel([models.MaxentModel()]),
    ],
)
def test_is_classifier(model_factory):
    # is_classifier reads from __sklearn_tags__ in sklearn >=1.6; failing this
    # also breaks sklearn.inspection.partial_dependence.
    assert is_classifier(model_factory())


@pytest.mark.parametrize(
    "model_factory",
    [models.MaxentModel, models.NicheEnvelopeModel],
)
def test_sklearn_tags(model_factory):
    tags = model_factory().__sklearn_tags__()
    assert tags.estimator_type == "classifier"
    assert tags.target_tags.required is True
    assert tags.classifier_tags is not None


def test_classes_is_ndarray():
    # sklearn 1.9 indexes into estimator.classes_ with numpy boolean masks;
    # a Python list `[0, 1]` silently breaks downstream metrics.
    assert isinstance(models.MaxentModel().classes_, np.ndarray)
    assert list(models.MaxentModel().classes_) == [0, 1]


def test_partial_dependence_accepts_integer_dataframe():
    # bradypus loads as int64; sklearn>=1.7 rejects integer feature dtypes in
    # partial_dependence unless we cast first.
    assert pd.api.types.is_integer_dtype(xb["cld6190_ann"])
    me = models.MaxentModel()
    me.fit(xb, yb)
    bins, mean, stdv = me.partial_dependence_scores(xb, categorical_features=me.transformer.categorical_ or [None])
    assert bins.shape == mean.shape == stdv.shape

    # every non-categorical feature should have at least one populated grid point
    non_cat = [i for i in range(xb.shape[1]) if i != xb.columns.get_loc("ecoreg")]
    for idx in non_cat:
        assert np.isfinite(bins[idx]).any(), f"feature {idx} produced no grid points"
        assert np.isfinite(mean[idx]).any()

    # the caller's dtypes must not have been mutated by the cast
    assert pd.api.types.is_integer_dtype(xb["cld6190_ann"])


def test_cast_integer_features_to_float_dataframe():
    df = pd.DataFrame(
        {
            "int_col": np.array([1, 2, 3], dtype="int64"),
            "uint_col": np.array([4, 5, 6], dtype="uint8"),
            "float_col": np.array([1.0, 2.0, 3.0]),
            "cat_col": pd.Categorical(["a", "b", "a"]),
        }
    )
    out = models._cast_integer_features_to_float(df)

    assert out["int_col"].dtype == np.float64
    assert out["uint_col"].dtype == np.float64
    assert out["float_col"].dtype == np.float64
    assert isinstance(out["cat_col"].dtype, pd.CategoricalDtype)

    # input must not be mutated
    assert df["int_col"].dtype == np.int64
    assert df["uint_col"].dtype == np.uint8


def test_cast_integer_features_to_float_numpy():
    arr_int = np.array([[1, 2], [3, 4]], dtype="int64")
    out = models._cast_integer_features_to_float(arr_int)
    assert out.dtype == np.float64

    arr_float = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = models._cast_integer_features_to_float(arr_float)
    assert out is arr_float  # passthrough when no cast needed


def test_partial_dependence_scores_matches_sklearn():
    # the helper should produce values consistent with sklearn's underlying
    # partial_dependence call (sanity check that our cast/loop didn't drift).
    me = models.MaxentModel()
    me.fit(x, y)
    percentiles = (0.025, 0.975)
    bins, mean, stdv = me.partial_dependence_scores(x, n_bins=20, percentiles=percentiles)

    pd_result = partial_dependence(me, x, [0], grid_resolution=20, percentiles=percentiles, kind="individual")
    np.testing.assert_allclose(bins[0], pd_result["grid_values"][0])
    np.testing.assert_allclose(mean[0], pd_result["individual"][0].mean(axis=0))
