"""Tests for :mod:`pyts.classification` module."""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
from itertools import product
import numpy as np
from ..classification import SAXVSMClassifier, KNNClassifier, BOSSVSClassifier


standard_library.install_aliases()


def test_KNNClassifier():
    """Testing 'KNNClassifier'."""
    # Parameter
    x1 = np.array([1, -1, 1, 1, 1, -1])
    x2 = np.ones(6)
    x3 = - np.ones(6)
    X = np.vstack([x1, x2, x3])

    # DTW metric
    clf = KNNClassifier(n_neighbors=1, metric='dtw')

    # Test 1
    indices = [0, 1]
    y = np.array([0, 1])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[2].reshape(1, -1))
    np.testing.assert_array_equal(0, y_pred)

    # Test 2
    indices = [0, 2]
    y = np.array([1, 0])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[1].reshape(1, -1))
    np.testing.assert_array_equal(1, y_pred)

    # Test 3
    indices = [1, 2]
    y = np.array([1, 0])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[1].reshape(1, -1))
    np.testing.assert_array_equal(1, y_pred)

    # FastDTW metric
    clf = KNNClassifier(n_neighbors=1,
                        metric='fast_dtw',
                        metric_params={'window_size': 2})

    # Test 1
    indices = [0, 1]
    y = np.array([0, 1])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[2].reshape(1, -1))
    np.testing.assert_array_equal(0, y_pred)

    # Test 2
    indices = [0, 2]
    y = np.array([1, 0])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[1].reshape(1, -1))
    np.testing.assert_array_equal(1, y_pred)

    # Test 3
    indices = [1, 2]
    y = np.array([1, 0])
    clf.fit(X[indices], y)
    y_pred = clf.predict(X[1].reshape(1, -1))
    np.testing.assert_array_equal(1, y_pred)


def test_SAXVSMClassifier():
    """Testing 'SAXVSMClassifier'."""
    # Parameter
    x1 = np.array(["aaa", "bbb", "bbb"])
    x2 = np.array(["aaa", "bbb", "ccc"])
    x3 = np.array(["aaa", "ddd", "ddd"])
    x4 = np.array(["aaa", "ddd", "eee"])
    X = np.vstack([x1, x2, x3, x4])
    y = np.array([0, 0, 1, 1])

    # Test 1
    clf = SAXVSMClassifier(use_idf=True,
                           smooth_idf=True,
                           sublinear_tf=False)
    clf.fit(X, y)
    test_index = 2
    np.testing.assert_equal(y[test_index],
                            clf.predict(X[test_index][np.newaxis, :]))
    test_index = np.array([2, 3])
    np.testing.assert_equal(y[test_index], clf.predict(X[test_index]))

    # Test 2: loop
    use_idf_list = [True, False]
    smooth_idf_list = [True, False]
    sublinear_tf_list = [True, False]
    for use_idf, smooth_idf, sublinear_tf in product(*[use_idf_list,
                                                       smooth_idf_list,
                                                       sublinear_tf_list]):
        clf = SAXVSMClassifier(use_idf=use_idf,
                               smooth_idf=smooth_idf,
                               sublinear_tf=sublinear_tf)
        clf.fit(X, y)
        clf.predict(X[:2])


def test_BOSSVS():
    """Testing 'BOSSVS'."""
    # Parameters
    X = np.arange(1, 21).reshape(20, 1) * np.ones((20, 10))
    y = np.repeat([0, 0, 1, 1, 0, 0, 1, 1, 2, 2], 2)

    # Test 1
    bossvs = BOSSVSClassifier(n_coefs=4, window_size=5, norm_mean=False,
                              norm_std=False)
    bossvs.fit(X, y).predict(X)

    # Test 2
    bossvs = BOSSVSClassifier(n_coefs=2, window_size=5, norm_mean=False,
                              norm_std=False, quantiles='empirical')
    bossvs.fit(X, y).predict(X)

    # Test 3: loop
    rng = np.random.RandomState(41)
    X_noise = rng.randn(20, 10)

    n_coefs_list = [4, None]
    window_size_list = [6, 10]
    norm_mean_list = [True, False]
    norm_std_list = [True, False]
    n_bins_list = [3, 5]
    quantiles_list = ['gaussian', 'empirical']
    variance_selection_list = [True, False]
    variance_threshold_list = [0., 0.001]
    numerosity_reduction_list = [True, False]
    smooth_idf_list = [True, False]
    sublinear_tf_list = [True, False]

    for (n_coefs, window_size, norm_mean, norm_std, n_bins, quantiles,
         variance_selection, variance_threshold, numerosity_reduction,
         smooth_idf, sublinear_tf) in product(*[n_coefs_list,
                                                window_size_list,
                                                norm_mean_list,
                                                norm_std_list,
                                                n_bins_list,
                                                quantiles_list,
                                                variance_selection_list,
                                                variance_threshold_list,
                                                numerosity_reduction_list,
                                                smooth_idf_list,
                                                sublinear_tf_list]):
        bossvs = BOSSVSClassifier(n_coefs, window_size, norm_mean, norm_std,
                                  n_bins, quantiles, variance_selection,
                                  variance_threshold, numerosity_reduction,
                                  smooth_idf, sublinear_tf)
        bossvs.fit(X_noise, y).predict(X_noise[:2])
