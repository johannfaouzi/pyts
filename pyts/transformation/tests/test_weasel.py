"""Testing for Word ExtrAction for time SEries cLassification."""

import numpy as np
from itertools import product
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
    def type_error_list():
        type_error_list_ = [
            "'word_size' must be an integer.",
            "'window_sizes' must be array-like.",
            "'window_steps' must be None or array-like.",
            "'chi2_threshold' must be a float or an integer."
        ]
        return type_error_list_

    def value_error_list():
        value_error_list_ = [
            "'word_size' must be a positive integer.",
            "'window_sizes' must be one-dimensional.",
            "The elements of 'window_sizes' must be integers or floats.",
            "If the elements of 'window_sizes' are floats, they all "
            "must be greater than 0 and lower than or equal to 1.",
            "All the elements in 'window_sizes' must be "
            "lower than or equal to n_timestamps.",
            "If 'drop_sum=True', 'word_size' must be lower than "
            "the minimum value in 'window_sizes'.",
            "If 'drop_sum=False', 'word_size' must be lower than or "
            "equal to the minimum value in 'window_sizes'.",
            "'window_steps' must be one-dimensional.",
            "If 'window_steps' is not None, it must have "
            "the same size as 'window_sizes'.",
            "If 'window_steps' is not None, the elements of 'window_steps' "
            "must be integers or floats.",
            "If the elements of 'window_steps' are floats, they "
            "all must be greater than 0 and lower than or equal to 1.",
            "All the elements in 'window_steps' must be greater than or "
            "equal to 1.",
            "All the elements in 'window_steps' must be "
            "lower than or equal to n_timestamps.",
            "'chi2_threshold' must be positive."
        ]
        return value_error_list_

    word_size_list = [-1, 2, 40, 8, None]
    window_sizes_list = [['a', 'b'], [-1, 1], [0.5, 2.], [0.5], [5]]
    window_steps_list = [['a', 'b'], [-1, 1], [0.5, 2.], [0.5], [5], None]
    drop_sum_list = [True, False]

    for (word_size, window_sizes, window_steps, drop_sum) in product(
        word_size_list, window_sizes_list, window_steps_list, drop_sum_list
    ):
        weasel = WEASEL(word_size=word_size, window_sizes=window_sizes,
                        window_steps=window_steps, drop_sum=drop_sum)
        try:
            weasel.fit_transform(X, y)
            weasel.fit(X, y).transform(X)
        except ValueError as e:
            if str(e) in value_error_list():
                pass
            else:
                raise ValueError("Unexpected ValueError: {}".format(e))
        except TypeError as e:
            if str(e) in type_error_list():
                pass
            else:
                raise TypeError("Unexpected TypeError: {}".format(e))

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
