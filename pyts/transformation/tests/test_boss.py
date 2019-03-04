"""Testing for Bag-of-SFA Symbols."""

import numpy as np
from itertools import product
from sklearn.feature_extraction.text import CountVectorizer
from ..boss import BOSS
from ...approximation import SymbolicFourierApproximation


def test_BOSS():
    """Test 'BOSS' class."""
    rng = np.random.RandomState(42)
    X = rng.randn(8, 20)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    # Parameter check
    def type_error_list():
        type_error_list_ = [
            "'word_size' must be an integer.",
            "'window_size' must be an integer or a float.",
            "'window_step' must be an integer or a float.",
        ]
        return type_error_list_

    def value_error_list():
        value_error_list_ = [
            "'word_size' must be a positive integer.",
            "If 'window_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to "
            "(n_timestamps - 1) if 'drop_sum=True'.",
            "If 'window_size' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to "
            "n_timestamps if 'drop_sum=False'.",
            "If 'window_size' is a float, it must be greater "
            "than 0 and lower than or equal to 1.",
            "If 'window_step' is an integer, it must be greater "
            "than or equal to 1 and lower than or equal to "
            "n_timestamps.",
            "If 'window_step' is a float, it must be greater "
            "than 0 and lower than or equal to 1.",
            "'word_size' must be lower than or equal to "
            "(window_size - 1) if 'drop_sum=True'.",
            "'word_size' must be lower than or equal to "
            "window_size if 'drop_sum=False'."
        ]
        return value_error_list_

    word_size_list = [-1, 2, 40, 8, None]
    window_size_list = [-1, 2, 5, 7, 8, 0.5, None]
    window_step_list = [-1, 2, 5, 7, 8, 0.5, None]
    drop_sum_list = [True, False]

    for (word_size, window_size, window_step, drop_sum) in product(
        word_size_list, window_size_list, window_step_list, drop_sum_list
    ):
        boss = BOSS(word_size=word_size, window_size=window_size,
                    window_step=window_step, drop_sum=drop_sum)
        try:
            boss.fit_transform(X)
            boss.fit(X).transform(X)
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

    # Test 1: numerosity_reduction=False
    boss = BOSS(
        word_size=4, n_bins=3, window_size=10, window_step=10,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=False
    )

    X_windowed = X.reshape(8, 2, 10).reshape(16, 10)
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

    arr_actual = boss.fit_transform(X, y)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired

    arr_actual = boss.fit(X, y).transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired

    # Test 2: numerosity_reduction=True
    boss = BOSS(
        word_size=4, n_bins=3, window_size=10, window_step=10,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=True
    )

    X_windowed = X.reshape(8, 2, 10).reshape(16, 10)
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

    arr_actual = boss.fit_transform(X, y)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired

    arr_actual = boss.fit(X, y).transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)
    assert boss.vocabulary_ == vocabulary_desired
