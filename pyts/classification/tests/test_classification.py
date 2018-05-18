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

    # Test: loop
    n_neighbors_list = [1, 2]
    weights_list = ['uniform', 'distance']
    algorithm_list = ['auto', 'ball_tree', 'kd_tree', 'brute']
    leaf_size_list = [5, 30]
    metric_list = ['euclidean', 'minkowski', 'manhattan']
    p_list = [1, 2, 3]
    n_jobs_list = [1, -1]
    for (n_neighbors, weights, algorithm, leaf_size, metric, p,
         n_jobs) in product(*[n_neighbors_list, weights_list, algorithm_list,
                              leaf_size_list, metric_list, p_list,
                              n_jobs_list]):
        clf = KNNClassifier(n_neighbors=n_neighbors,
                            weights=weights,
                            algorithm=algorithm,
                            leaf_size=leaf_size,
                            metric=metric,
                            p=p,
                            n_jobs=n_jobs)
        clf.fit(X[indices], y).predict(X)


def test_SAXVSMClassifier():
    """Testing 'SAXVSMClassifier'."""
    # Parameter
    rng = np.random.RandomState(41)
    X = rng.randn(4, 10)
    y = np.array([0, 0, 1, 1])

    # Test: loop
    n_bins_list = [3, 4]
    quantiles_list = ['empirical', 'gaussian']
    window_size_list = [1, 4, 6]
    numerosity_reduction_list = [True, False]
    use_idf_list = [True, False]
    smooth_idf_list = [True, False]
    sublinear_tf_list = [True, False]
    for (n_bins, quantiles, window_size, numerosity_reduction, use_idf,
         smooth_idf, sublinear_tf) in product(*[n_bins_list,
                                                quantiles_list,
                                                window_size_list,
                                                numerosity_reduction_list,
                                                use_idf_list,
                                                smooth_idf_list,
                                                sublinear_tf_list]):
        print(numerosity_reduction)
        clf = SAXVSMClassifier(n_bins=n_bins,
                               quantiles=quantiles,
                               window_size=window_size,
                               numerosity_reduction=numerosity_reduction,
                               use_idf=use_idf,
                               smooth_idf=smooth_idf,
                               sublinear_tf=sublinear_tf)
        clf.fit(X, y)
        clf.predict(X[:2])


def test_BOSSVSClassifier():
    """Testing 'BOSSVSClassifier'."""
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
        bossvs.fit(X_noise, y, overlapping=True).predict(X_noise[:2])
        bossvs.fit(X_noise, y, overlapping=False).predict(X_noise[:2])
