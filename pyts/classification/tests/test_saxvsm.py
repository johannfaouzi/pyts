"""Testing for SAX-VSM."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import log
from sklearn.metrics.pairwise import cosine_similarity
from pyts.classification import SAXVSM


X = [[0, 0, 0, 1, 0, 0, 1, 1, 1],
     [0, 1, 1, 1, 0, 0, 1, 1, 1],
     [0, 0, 0, 1, 0, 0, 0, 1, 0]]
y = [0, 0, 1]


def test_actual_results_strategy_uniform():
    """Test that the actual results are the expected ones."""
    # Data
    X = [[0, 0, 0, 1, 0, 0, 1, 1, 1],
         [0, 1, 1, 1, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 0, 0, 0, 1, 0]]
    y = [0, 0, 1]

    clf = SAXVSM(window_size=4, word_size=4, n_bins=2, strategy='uniform',
                 numerosity_reduction=False, sublinear_tf=False)
    decision_function_actual = clf.fit(X, y).decision_function(X)

    # X_bow = ["aaab aaba abaa baab aabb abbb",
    #          "abbb bbba bbaa baab aabb abbb",
    #          "aaab aaba abaa baaa aaab aaba"]

    assert clf.vocabulary_ == {0: 'aaab', 1: 'aaba', 2: 'aabb', 3: 'abaa',
                               4: 'abbb', 5: 'baaa', 6: 'baab', 7: 'bbaa',
                               8: 'bbba'}

    freq = np.asarray([[1, 1, 1, 1, 1, 0, 1, 0, 0],
                       [0, 0, 1, 0, 2, 0, 1, 1, 1],
                       [2, 2, 0, 1, 0, 1, 0, 0, 0]])
    tf = np.asarray([[1, 1, 2, 1, 3, 0, 2, 1, 1],
                     [2, 2, 0, 1, 0, 1, 0, 0, 0]])
    idf = np.asarray([1, 1, log(2) + 1, 1, log(2) + 1, log(2) + 1, log(2) + 1,
                      log(2) + 1, log(2) + 1])
    decision_function_desired = cosine_similarity(freq, tf * idf[None, :])
    np.testing.assert_allclose(decision_function_actual,
                               decision_function_desired, atol=1e-5, rtol=0.)

    pred_actual = clf.predict(X)
    pred_desired = cosine_similarity(freq, tf * idf[None, :]).argmax(axis=1)
    np.testing.assert_array_equal(pred_actual, pred_desired)


def test_actual_results_strategy_quantile():
    """Test that the actual results are the expected ones."""
    # Data
    X = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
         [0.0, 0.3, 0.2, 0.4, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9],
         [0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]]
    y = [0, 0, 1]

    clf = SAXVSM(window_size=4, word_size=4, n_bins=2, strategy='quantile',
                 numerosity_reduction=False, sublinear_tf=False)
    decision_function_actual = clf.fit(X, y).decision_function(X)

    # X_bow = ["aabb aabb aabb aabb aabb aabb aabb",
    #          "abab baba abab aabb aabb aabb aabb",
    #          "abab baba abab baba abab baba abab"]

    assert clf.vocabulary_ == {0: 'aabb', 1: 'abab', 2: 'baba'}

    freq = np.asarray([[7, 0, 0],
                       [4, 2, 1],
                       [0, 4, 3]])
    tf = np.asarray([[11, 2, 1],
                     [0, 4, 3]])
    idf = np.asarray([log(2) + 1, 1, 1])
    decision_function_desired = cosine_similarity(freq, tf * idf[None, :])
    np.testing.assert_allclose(decision_function_actual,
                               decision_function_desired, atol=1e-5, rtol=0.)

    pred_actual = clf.fit(X, y).predict(X)
    pred_desired = cosine_similarity(freq, tf * idf[None, :]).argmax(axis=1)
    np.testing.assert_array_equal(pred_actual, pred_desired)
