import numpy as np

from elapid import stats


def test_normalize_sample_probabilities():
    input = np.array((2, 1, 1))
    normed = stats.normalize_sample_probabilities(input)
    assert normed.sum() == 1
    assert normed.max() == 0.5
    assert normed.min() == 0.25
