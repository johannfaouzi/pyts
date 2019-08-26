"""Testing for Bag-of-SFA Symbols in Vector Space."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from pyts.classification import BOSSVS
from pyts.approximation import SymbolicFourierApproximation


rng = np.random.RandomState(42)
X = rng.randn(8, 20)
y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'word_size': '4'}, TypeError, "'word_size' must be an integer."),

     ({'window_size': None}, TypeError,
      "'window_size' must be an integer or a float."),

     ({'window_step': '2'}, TypeError,
      "'window_step' must be an integer or a float."),

     ({'word_size': 0}, ValueError, "'word_size' must be a positive integer."),

     ({'window_size': 0}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps if 'drop_sum=False'."),

     ({'window_size': 0, 'drop_sum': True}, ValueError,
      "If 'window_size' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to (n_timestamps - 1) if 'drop_sum=True'."),

     ({'window_size': 2.}, ValueError,
      "If 'window_size' is a float, it must be greater than 0 and lower than "
      "or equal to 1."),

     ({'window_step': 0}, ValueError,
      "If 'window_step' is an integer, it must be greater than or equal to 1 "
      "and lower than or equal to n_timestamps."),

     ({'window_step': 2.}, ValueError,
      "If 'window_step' is a float, it must be greater than 0 and lower than "
      "or equal to 1."),

     ({'window_size': 4, 'word_size': 4, 'drop_sum': True}, ValueError,
      "'word_size' must be lower than or equal to (window_size - 1) if "
      "'drop_sum=True'."),

     ({'window_size': 4, 'word_size': 5}, ValueError,
      "'word_size' must be lower than or equal to window_size if "
      "'drop_sum=False'.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    clf = BOSSVS(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X, y)


def test_actual_results_no_numerosity_reduction():
    """Test that the actual results are the expected ones."""
    bossvs = BOSSVS(
        word_size=4, n_bins=3, window_size=10, window_step=10,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=False,
        use_idf=True, smooth_idf=False, sublinear_tf=True
    )

    X_windowed = X.reshape(8, 2, 10).reshape(16, 10)
    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=False, anova=False, norm_mean=False,
        norm_std=False, n_bins=3, strategy='quantile', alphabet=None
    )
    y_repeated = np.repeat(y, 2)
    X_sfa = sfa.fit_transform(X_windowed, y_repeated)
    X_word = np.asarray([''.join(X_sfa[i])
                         for i in range(16)])
    X_word = X_word.reshape(8, 2)
    X_bow = np.asarray([' '.join(X_word[i]) for i in range(8)])
    X_class = np.array([' '.join(X_bow[y == i]) for i in range(2)])

    tfidf = TfidfVectorizer(
        norm=None, use_idf=True, smooth_idf=False, sublinear_tf=True
    )
    tfidf_desired = tfidf.fit_transform(X_class).toarray()

    # Vocabulary
    vocabulary_desired = {value: key for key, value in
                          tfidf.vocabulary_.items()}

    # Tf-idf
    tfidf_actual = bossvs.fit(X, y).tfidf_

    # Decision function
    decision_function_actual = bossvs.decision_function(X)
    decision_function_desired = cosine_similarity(
        tfidf.transform(X_bow), tfidf_desired)

    # Predictions
    y_pred_actual = bossvs.predict(X)
    y_pred_desired = decision_function_desired.argmax(axis=1)

    # Testing
    assert bossvs.vocabulary_ == vocabulary_desired
    np.testing.assert_allclose(tfidf_actual, tfidf_desired, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        decision_function_actual, decision_function_desired, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        y_pred_actual, y_pred_desired, atol=1e-5, rtol=0)


def test_actual_results_numerosity_reduction():
    """Test that the actual results are the expected ones."""
    bossvs = BOSSVS(
        word_size=4, n_bins=3, window_size=10, window_step=10,
        anova=False, drop_sum=False, norm_mean=False, norm_std=False,
        strategy='quantile', alphabet=None, numerosity_reduction=True,
        use_idf=True, smooth_idf=False, sublinear_tf=True
    )

    X_windowed = X.reshape(8, 2, 10).reshape(16, 10)
    sfa = SymbolicFourierApproximation(
        n_coefs=4, drop_sum=False, anova=False, norm_mean=False,
        norm_std=False, n_bins=3, strategy='quantile', alphabet=None
    )
    y_repeated = np.repeat(y, 2)
    X_sfa = sfa.fit_transform(X_windowed, y_repeated)
    X_word = np.asarray([''.join(X_sfa[i])
                         for i in range(16)])
    X_word = X_word.reshape(8, 2)
    not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1], np.full(8, True)]
    X_bow = np.asarray([' '.join(X_word[i, not_equal[i]]) for i in range(8)])
    X_class = np.array([' '.join(X_bow[y == i]) for i in range(2)])

    tfidf = TfidfVectorizer(
        norm=None, use_idf=True, smooth_idf=False, sublinear_tf=True
    )
    tfidf_desired = tfidf.fit_transform(X_class).toarray()
    vocabulary_desired = {value: key for key, value in
                          tfidf.vocabulary_.items()}

    # Tf-idf
    tfidf_actual = bossvs.fit(X, y).tfidf_

    # Decision function
    decision_function_actual = bossvs.decision_function(X)
    decision_function_desired = cosine_similarity(
        tfidf.transform(X_bow), tfidf_desired)

    # Predictions
    y_pred_actual = bossvs.predict(X)
    y_pred_desired = decision_function_desired.argmax(axis=1)

    # Testing
    assert bossvs.vocabulary_ == vocabulary_desired
    np.testing.assert_allclose(tfidf_actual, tfidf_desired, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        decision_function_actual, decision_function_desired, atol=1e-5, rtol=0)
    np.testing.assert_allclose(
        y_pred_actual, y_pred_desired, atol=1e-5, rtol=0)
