"""Testing for Bag-of-SFA Symbols."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from pyts.transformation import BOSS
from pyts.approximation import SymbolicFourierApproximation


n_samples, n_timestamps, n_classes = 8, 200, 2
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(n_classes, size=n_samples)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'word_size': "3"}, TypeError, "'word_size' must be an integer."),

     ({'window_size': {}}, TypeError,
      "'window_size' must be an integer or a float."),

     ({'window_step': {}}, TypeError,
      "'window_step' must be an integer or a float."),

     ({'word_size': 0}, ValueError, "'word_size' must be a positive integer."),

     ({'window_size': 0, 'drop_sum': True}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to (n_timestamps - 1) if 'drop_sum=True'."),

     ({'window_size': n_timestamps, 'drop_sum': True}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to (n_timestamps - 1) if 'drop_sum=True'."),

     ({'window_size': 0}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps if 'drop_sum=False'."),

     ({'window_size': n_timestamps + 1}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps if 'drop_sum=False'."),

     ({'window_size': 1.5}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1."),

     ({'window_step': 0}, ValueError,
      "If 'window_step' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps."),

     ({'window_step': n_timestamps + 1}, ValueError,
      "If 'window_step' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps."),

     ({'window_step': 0.}, ValueError,
      "If 'window_step' is a float, it must be greater than 0 and lower than "
      "or equal to 1."),

     ({'window_step': 1.2}, ValueError,
      "If 'window_step' is a float, it must be greater than 0 and lower than "
      "or equal to 1."),

     ({'window_size': 4, 'drop_sum': True}, ValueError,
      "'word_size' must be lower than or equal to (window_size - 1) if "
      "'drop_sum=True'."),

     ({'window_size': 3}, ValueError,
      "'word_size' must be lower than or equal to window_size if "
      "'drop_sum=False'.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    boss = BOSS(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        boss.fit(X, y)


@pytest.mark.parametrize(
    'sparse, instance', [(True, csr_matrix), (False, np.ndarray)])
def test_sparse_dense(sparse, instance):
    """Test that the expected type is returned."""
    weasel = BOSS(sparse=sparse)
    assert isinstance(weasel.fit(X, y).transform(X), instance)
    assert isinstance(weasel.fit_transform(X, y), instance)


def test_accurate_results_without_numerosity_reduction():
    """Test that the actual results are the expected ones."""
    boss = BOSS(
        word_size=4, n_bins=3, window_size=100, window_step=100,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=False
    )

    X_windowed = X.reshape(8, 2, 100).reshape(16, 100)
    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=False, anova=False, norm_mean=False,
        norm_std=False, n_bins=3, strategy='quantile', alphabet=None
    )
    y_repeated = np.repeat(y, 2)
    X_sfa = sfa.fit_transform(X_windowed, y_repeated)
    X_word = np.asarray([''.join(X_sfa[i]) for i in range(16)])
    X_word = X_word.reshape(8, 2)
    X_bow = np.asarray([' '.join(X_word[i]) for i in range(8)])

    vectorizer = CountVectorizer()
    arr_desired = vectorizer.fit_transform(X_bow).toarray()
    vocabulary_desired = {value: key for key, value in
                          vectorizer.vocabulary_.items()}

    arr_actual = boss.fit_transform(X, y).toarray()
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired

    arr_actual = boss.fit(X, y).transform(X).toarray()
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired


def test_accurate_results_floats():
    """Test that the actual results are the expected ones."""
    boss = BOSS(
        word_size=4, n_bins=3, window_size=0.5, window_step=0.5,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=True
    )

    X_windowed = X.reshape(8, 2, 100).reshape(16, 100)
    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=False, anova=False, norm_mean=False,
        norm_std=False, n_bins=3, strategy='quantile', alphabet=None
    )
    y_repeated = np.repeat(y, 2)
    X_sfa = sfa.fit_transform(X_windowed, y_repeated)
    X_word = np.asarray([''.join(X_sfa[i]) for i in range(16)])
    X_word = X_word.reshape(8, 2)
    not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1], np.full(8, True)]
    X_bow = np.asarray([' '.join(X_word[i, not_equal[i]]) for i in range(8)])

    vectorizer = CountVectorizer()
    arr_desired = vectorizer.fit_transform(X_bow).toarray()
    vocabulary_desired = {value: key for key, value in
                          vectorizer.vocabulary_.items()}

    arr_actual_1 = boss.fit_transform(X, None).toarray()
    np.testing.assert_allclose(arr_actual_1, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired

    arr_actual_2 = boss.fit(X, None).transform(X).toarray()
    np.testing.assert_allclose(arr_actual_2, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired
