"""Testing for Bag-of-SFA Symbols in Vector Space."""

import numpy as np
import pytest
import re
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from ..bossvs import BOSSVS
from ...approximation import SymbolicFourierApproximation


def test_BOSSVS():
    """Test 'BOSSVS' class."""
    rng = np.random.RandomState(42)
    X = rng.randn(8, 20)
    y = np.asarray([0, 0, 0, 0, 1, 1, 1, 1])

    # Parameter check
    msg_error = "'word_size' must be an integer."
    with pytest.raises(TypeError, match=msg_error):
        bossvs = BOSSVS(word_size="3", window_size=4,
                        window_step=1, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = "'window_size' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size="3",
                        window_step=1, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = "'window_step' must be an integer or a float."
    with pytest.raises(TypeError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size=4,
                        window_step=None, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = "'word_size' must be a positive integer."
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=0, window_size=4,
                        window_step=1, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = re.escape(
        "If 'window_size' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to "
        "(n_timestamps - 1) if 'drop_sum=True'."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size=0,
                        window_step=1, drop_sum=True)
        bossvs.fit(X, y).predict(X)

    msg_error = (
        "If 'window_size' is a float, it must be greater "
        "than 0 and lower than or equal to 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size=2.,
                        window_step=1, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = (
        "If 'window_step' is an integer, it must be greater "
        "than or equal to 1 and lower than or equal to "
        "n_timestamps."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size=4,
                        window_step=0, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = (
        "If 'window_step' is a float, it must be greater "
        "than 0 and lower than or equal to 1."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=2, window_size=4,
                        window_step=2., drop_sum=False)
        bossvs.fit(X, y).predict(X)

    msg_error = re.escape(
        "'word_size' must be lower than or equal to "
        "(window_size - 1) if 'drop_sum=True'."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=4, window_size=4,
                        window_step=1, drop_sum=True)
        bossvs.fit(X, y).predict(X)

    msg_error = (
        "'word_size' must be lower than or equal to "
        "window_size if 'drop_sum=False'."
    )
    with pytest.raises(ValueError, match=msg_error):
        bossvs = BOSSVS(word_size=5, window_size=4,
                        window_step=1, drop_sum=False)
        bossvs.fit(X, y).predict(X)

    # Test 1: numerosity_reduction=False
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
    vocabulary_desired = {value: key for key, value in
                          tfidf.vocabulary_.items()}

    # Testing for tfidf_
    tfidf_actual = bossvs.fit(X, y).tfidf_
    np.testing.assert_allclose(tfidf_actual, tfidf_desired, atol=1e-5, rtol=0)

    # Testing for vocabulary_
    assert bossvs.vocabulary_ == vocabulary_desired

    # Testing for decision_function
    decision_function_actual = bossvs.decision_function(X)
    decision_function_desired = cosine_similarity(
        tfidf.transform(X_bow), tfidf_desired)
    np.testing.assert_allclose(
        decision_function_actual, decision_function_desired, atol=1e-5, rtol=0)

    # Testing for predict
    y_pred_actual = bossvs.predict(X)
    y_pred_desired = decision_function_desired.argmax(axis=1)
    np.testing.assert_allclose(
        y_pred_actual, y_pred_desired, atol=1e-5, rtol=0)

    # Test 2: numerosity_reduction=True
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

    # Testing for tfidf_
    tfidf_actual = bossvs.fit(X, y).tfidf_
    np.testing.assert_allclose(tfidf_actual, tfidf_desired, atol=1e-5, rtol=0)

    # Testing for vocabulary_
    assert bossvs.vocabulary_ == vocabulary_desired

    # Testing for decision_function
    decision_function_actual = bossvs.decision_function(X)
    decision_function_desired = cosine_similarity(
        tfidf.transform(X_bow), tfidf_desired)
    np.testing.assert_allclose(
        decision_function_actual, decision_function_desired, atol=1e-5, rtol=0)

    # Testing for predict
    y_pred_actual = bossvs.predict(X)
    y_pred_desired = decision_function_desired.argmax(axis=1)
    np.testing.assert_allclose(
        y_pred_actual, y_pred_desired, atol=1e-5, rtol=0)
