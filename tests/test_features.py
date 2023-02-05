from elapid import features
from elapid.utils import load_sample_data

x, y = load_sample_data(name="ariolimax", drop_geometry=True)
nrows, ncols = x.shape


def test_LinearTransformer():
    # feature range testing
    tmin, tmax = (0, 2)
    lt = features.LinearTransformer(clamp=True, feature_range=(tmin, tmax))
    t = lt.fit_transform(x)
    assert t.min() == tmin
    assert t.max() == tmax

    # clamp testing
    mid_row = int(nrows // 2)
    lt = features.LinearTransformer(clamp=False)
    t1 = lt.fit_transform(x.iloc[:mid_row])
    t2 = lt.transform(x.iloc[mid_row:])
    assert t2.max() > t1.max()


def test_QuadraticTransformer():
    # feature range testing
    tmin, tmax = (0, 2)
    qt = features.QuadraticTransformer(clamp=True, feature_range=(tmin, tmax))
    t = qt.fit_transform(x)
    assert t.min() == tmin
    assert t.max() == tmax

    # clamp testing
    mid_row = int(nrows // 2)
    qt = features.QuadraticTransformer(clamp=False)
    t1 = qt.fit_transform(x.iloc[:mid_row])
    t2 = qt.transform(x.iloc[mid_row:])
    assert t2.max() > t1.max()


def test_ProductTransformer():
    # feature range testing
    tmin, tmax = (0, 2)
    pt = features.ProductTransformer(clamp=True, feature_range=(tmin, tmax))
    t = pt.fit_transform(x)
    assert t.min() == tmin
    assert t.max() == tmax

    # should be more features than covariates
    trows, tcols = t.shape
    assert trows == nrows
    assert tcols > ncols

    # clamp testing
    mid_row = int(nrows // 2)
    pt = features.ProductTransformer(clamp=False)
    t1 = pt.fit_transform(x.iloc[:mid_row])
    t2 = pt.transform(x.iloc[mid_row:])
    assert t2.max() > t1.max()


def test_ThresholdTransformer():
    n_thresholds = 5
    tt = features.ThresholdTransformer(n_thresholds=n_thresholds)
    t = tt.fit_transform(x)
    trows, tcols = t.shape
    assert tcols == n_thresholds * ncols


def test_HingeTransformer():
    n_hinges = 5
    ht = features.HingeTransformer(n_hinges=n_hinges)
    t = ht.fit_transform(x)
    trows, tcols = t.shape
    assert tcols == (n_hinges - 1) * 2 * ncols


def test_CategoricalTransformer():
    x, y = load_sample_data("bradypus")
    cat = x["ecoreg"]
    n_unique = len(cat.unique())
    ct = features.CategoricalTransformer()
    t = ct.fit_transform(cat)
    trows, tcols = t.shape
    assert trows == len(y)
    assert tcols == n_unique


# TODO
def test_CumulativeTransformer():
    pass


def test_MaxentFeatureTransformer():
    types = ["l", "q", "p", "t", "a"]
    for ftype in types:
        ft = features.MaxentFeatureTransformer(feature_types=ftype)
        t = ft.fit_transform(x)
        trows, tcols = t.shape
        assert t.ndim == 2
        assert trows == nrows
        assert tcols >= ncols


def test_compute_weights():
    pbr = 10
    weights = features.compute_weights(y, pbr=pbr)
    assert weights[y == 0].max() == pbr
    assert weights[y == 1].max() == 1


def test_compute_regularization():
    ft = features.MaxentFeatureTransformer()
    t = ft.fit_transform(x)
    trows, tcols = t.shape
    reg = features.compute_regularization(y, t, feature_labels=ft.feature_names_)
    assert len(reg) == tcols
    assert reg.min() >= 0.0
    assert reg.max() <= 1.0


def test_compute_lambdas():
    n_lambdas = 50
    ft = features.MaxentFeatureTransformer()
    t = ft.fit_transform(x)
    reg = features.compute_regularization(y, t, feature_labels=ft.feature_names_)
    weights = features.compute_weights(y)
    lambdas = features.compute_lambdas(y, weights, reg, n_lambdas)
    assert len(lambdas) == n_lambdas
    assert lambdas.min() >= 0
