"""Testing for Word ExtrAction for time SEries cLassification."""

import numpy as np
import pytest
from scipy.sparse import csc_matrix, hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from ...approximation import SymbolicFourierApproximation
from ..weasel import WEASEL


def test_WEASEL():
    """Test 'WEASEL' class."""
    rng = np.random.RandomState(42)
    X = rng.randn(8, 20)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])
    n_samples = 8

    # Parameter check
    msg_error = "'word_size' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        weasel = WEASEL(word_size="3", window_sizes=[0.5],
                        window_steps=[1], drop_sum=False)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'window_sizes' must be array-like."
    with pytest.raises(TypeError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes={},
                        window_steps=[1], drop_sum=False)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'window_steps' must be None or array-like."
    with pytest.raises(TypeError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=[6],
                        window_steps="3", drop_sum=False)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'chi2_threshold' must be a float or an integer."
    with pytest.raises(TypeError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=[6],
                        window_steps=[1], chi2_threshold="3")
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'word_size' must be a positive integer."
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=0)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'window_sizes' must be one-dimensional."
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=np.ones((2, 4)))
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "The elements of 'window_sizes' must be integers or floats."
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=['a', 'b', 'c'])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If the elements of 'window_sizes' are floats, they all "
        "must be greater than 0 and lower than or equal to 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=[0.5, 2.])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "All the elements in 'window_sizes' must be "
        "lower than or equal to n_timestamps."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=[30])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If 'drop_sum=True', 'word_size' must be lower than "
        "the minimum value in 'window_sizes'."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=4, window_sizes=[4, 6], drop_sum=True)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If 'drop_sum=False', 'word_size' must be lower than or "
        "equal to the minimum value in 'window_sizes'."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[4, 6], drop_sum=False)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'window_steps' must be one-dimensional."
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=np.ones((2, 4)))
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If 'window_steps' is not None, it must have "
        "the same size as 'window_sizes'."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=[1, 2, 3])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If 'window_steps' is not None, the elements of 'window_steps' "
        "must be integers or floats."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=['a', 'b'])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "If the elements of 'window_steps' are floats, they "
        "all must be greater than 0 and lower than or equal to 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=[0.5, 2.])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "All the elements in 'window_steps' must be greater than or equal "
        "to 1 and lower than or equal to n_timestamps."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=[0, 2])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = (
        "All the elements in 'window_steps' must be greater than or equal "
        "to 1 and lower than or equal to n_timestamps."
    )
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        window_steps=[2, 25])
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    msg_error = "'chi2_threshold' must be positive."
    with pytest.raises(ValueError, match=msg_error):
        weasel = WEASEL(word_size=5, window_sizes=[8, 10],
                        chi2_threshold=-1)
        weasel.fit_transform(X, y)
        weasel.fit(X, y).transform(X)

    # Test 1:
    X_features = csc_matrix((n_samples, 0), dtype=np.int64)
    vocabulary_ = {}

    weasel = WEASEL(
        word_size=4, n_bins=3, window_sizes=[5, 10],
        window_steps=None, anova=True, drop_sum=True, norm_mean=True,
        norm_std=True, strategy='entropy', chi2_threshold=2, alphabet=None
    )

    for window_size, n_windows in zip([5, 10], [4, 2]):
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
        if old_length_vocab == 0:
            old_length_vocab = -1
        vocabulary = {value: key
                      for (key, value) in vectorizer.vocabulary_.items()}
        for i, idx in enumerate(relevant_features):
            vocabulary_[i + 1 + old_length_vocab] = \
                str(window_size) + " " + vocabulary[idx]

    arr_desired = X_features.toarray()

    arr_actual = weasel.fit_transform(X, y).toarray()
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert weasel.vocabulary_ == vocabulary_

    arr_actual = weasel.fit(X, y).transform(X).toarray()
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert weasel.vocabulary_ == vocabulary_
