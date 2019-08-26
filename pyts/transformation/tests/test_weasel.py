"""Testing for Word ExtrAction for time SEries cLassification."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from pyts.approximation import SymbolicFourierApproximation
from pyts.transformation import WEASEL


n_samples, n_timestamps, n_classes = 8, 200, 2
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_timestamps)
y = rng.randint(n_classes, size=n_samples)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'word_size': "3"}, TypeError, "'word_size' must be an integer."),

     ({'window_sizes': {}}, TypeError, "'window_sizes' must be array-like."),

     ({'window_steps': "3"}, TypeError,
      "'window_steps' must be None or array-like."),

     ({'chi2_threshold': "3"}, TypeError,
      "'chi2_threshold' must be a float or an integer."),

     ({'word_size': 0}, ValueError, "'word_size' must be a positive integer."),

     ({'window_sizes': np.ones((2, 4))}, ValueError,
      "'window_sizes' must be one-dimensional."),

     ({'window_sizes': ['a', 'b', 'c']}, ValueError,
      "The elements of 'window_sizes' must be integers or floats."),

     ({'window_sizes': [0.5, 2.]}, ValueError,
      "If the elements of 'window_sizes' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [300]}, ValueError,
      "All the elements in 'window_sizes' must be lower than or equal to "
      "n_timestamps."),

     ({'word_size': 4, 'window_sizes': [4, 6], 'drop_sum': True}, ValueError,
      "If 'drop_sum=True', 'word_size' must be lower than the minimum value "
      "in 'window_sizes'."),

     ({'word_size': 5, 'window_sizes': [4, 6], 'drop_sum': False}, ValueError,
      "If 'drop_sum=False', 'word_size' must be lower than or equal to the "
      "minimum value in 'window_sizes'."),

     ({'window_steps': np.ones((2, 4))}, ValueError,
      "'window_steps' must be one-dimensional."),

     ({'window_sizes': [8, 10], 'window_steps': [1, 2, 3]}, ValueError,
      "If 'window_steps' is not None, it must have the same size as "
      "'window_sizes'."),

     ({'window_sizes': [8, 10], 'window_steps': ['a', 'b']}, ValueError,
      "If 'window_steps' is not None, the elements of 'window_steps' must be "
      "integers or floats."),

     ({'window_sizes': [8, 10], 'window_steps': [0.5, 2.]}, ValueError,
      "If the elements of 'window_steps' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [8], 'window_steps': [0]}, ValueError,
      "All the elements in 'window_steps' must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps."),

     ({'window_sizes': [8], 'window_steps': [250]}, ValueError,
      "All the elements in 'window_steps' must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps."),

     ({'chi2_threshold': -1}, ValueError,
      "'chi2_threshold' must be positive.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    weasel = WEASEL(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        weasel.fit(X, y)


@pytest.mark.parametrize(
    'sparse, instance', [(True, csr_matrix), (False, np.ndarray)])
def test_sparse_dense(sparse, instance):
    """Test that the expected type is returned."""
    weasel = WEASEL(strategy='quantile', sparse=sparse)
    assert isinstance(weasel.fit(X, y).transform(X), instance)
    assert isinstance(weasel.fit_transform(X, y), instance)


def test_accurate_results():
    """Test that the actual results are the expected ones."""
    X_features = csr_matrix((n_samples, 0), dtype=np.int64)
    vocabulary_ = {}

    weasel = WEASEL(
        word_size=4, n_bins=3, window_sizes=[5, 10],
        window_steps=None, anova=True, drop_sum=True, norm_mean=True,
        norm_std=True, strategy='entropy', chi2_threshold=2, alphabet=None
    )

    for window_size, n_windows in zip([5, 10], [40, 20]):
        X_windowed = X.reshape(n_samples, n_windows, window_size)
        X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

        sfa = SymbolicFourierApproximation(
            n_coefs=4, drop_sum=True, anova=True, norm_mean=True,
            norm_std=True, n_bins=3, strategy='entropy', alphabet=None
        )
        y_repeated = np.repeat(y, n_windows)
        X_sfa = sfa.fit_transform(X_windowed, y_repeated)
        X_word = np.asarray([''.join(X_sfa[i])
                             for i in range((n_samples * n_windows))])
        X_word = X_word.reshape(n_samples, n_windows)
        X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        vectorizer = CountVectorizer(ngram_range=(1, 2))
        X_counts = vectorizer.fit_transform(X_bow)
        chi2_statistics, _ = chi2(X_counts, y)
        relevant_features = np.where(
            chi2_statistics > 2)[0]
        X_features = hstack([X_features, X_counts[:, relevant_features]])

        old_length_vocab = len(vocabulary_)
        vocabulary = {value: key
                      for (key, value) in vectorizer.vocabulary_.items()}
        for i, idx in enumerate(relevant_features):
            vocabulary_[i + old_length_vocab] = \
                str(window_size) + " " + vocabulary[idx]

    arr_desired = X_features.toarray()

    # Accuracte results for fit followed by transform
    arr_actual_1 = weasel.fit_transform(X, y).toarray()
    np.testing.assert_allclose(arr_actual_1, arr_desired, atol=1e-5, rtol=0)
    assert weasel.vocabulary_ == vocabulary_

    # Accuracte results for fit_transform
    arr_actual_2 = weasel.fit(X, y).transform(X).toarray()
    np.testing.assert_allclose(arr_actual_2, arr_desired, atol=1e-5, rtol=0)
    assert weasel.vocabulary_ == vocabulary_
