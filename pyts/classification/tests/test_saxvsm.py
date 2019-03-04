"""Testing for SAX-VSM."""

import numpy as np
from math import log
from sklearn.metrics.pairwise import cosine_similarity
from ..saxvsm import SAXVSM


def test_SAXVSM():
    """Test 'SAXVSM' class."""
    # Test 1
    X = [[0, 0, 0, 1, 0, 0, 1, 1, 1],
         [0, 1, 1, 1, 0, 0, 1, 1, 1],
         [0, 0, 0, 1, 0, 0, 0, 1, 0]]
    y = [0, 0, 1]

    clf = SAXVSM(n_bins=2, strategy='uniform', window_size=2,
                 numerosity_reduction=False, sublinear_tf=False)
    arr_actual = clf.fit(X, y).decision_function(X)

    # X_sax = np.array(['a', 'b'])[np.asarray(X)]
    # X_bow = ["aa aa ab ba aa ab bb bb",
    #          "ab bb bb ba aa ab bb bb",
    #          "aa aa ab ba aa aa ab ba"]
    freq = np.asarray([[3, 2, 1, 2],
                       [1, 2, 1, 4],
                       [4, 2, 2, 0]])
    tf = np.asarray([[4, 4, 2, 6],
                     [4, 2, 2, 0]])
    idf = np.asarray([1, 1, 1, log(2) + 1])
    arr_desired = cosine_similarity(freq, tf * idf[None, :])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    arr_actual = clf.fit(X, y).predict(X)
    arr_desired = cosine_similarity(freq, tf * idf[None, :]).argmax(axis=1)
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2
    X = [[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
         [0.0, 0.3, 0.2, 0.4, 0.1, 0.5, 0.6, 0.7, 0.8, 0.9],
         [0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.5]]
    y = [0, 0, 1]
    clf = SAXVSM(n_bins=2, strategy='quantile', window_size=2,
                 numerosity_reduction=False, sublinear_tf=False)
    arr_actual = clf.fit(X, y).decision_function(X)

    # X_sax = [['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
    #          ['a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'],
    #          ['a', 'b', 'a', 'b', 'a', 'b', 'a', 'b', 'a', 'b']]
    # X_bow = ["aa aa aa aa ab bb bb bb bb",
    #          "aa aa aa aa ab bb bb bb bb",
    #          "ab ba ab ba ab ba ab ba ab"]
    freq = np.asarray([[4, 1, 0, 4],
                       [4, 1, 0, 4],
                       [0, 5, 4, 0]])
    tf = np.asarray([[8, 2, 0, 8],
                     [0, 5, 4, 0]])
    idf = np.asarray([log(2) + 1, 1, log(2) + 1, log(2) + 1])
    arr_desired = cosine_similarity(freq, tf * idf[None, :])
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    arr_actual = clf.fit(X, y).predict(X)
    arr_desired = cosine_similarity(freq, tf * idf[None, :]).argmax(axis=1)
    np.testing.assert_array_equal(arr_actual, arr_desired)
