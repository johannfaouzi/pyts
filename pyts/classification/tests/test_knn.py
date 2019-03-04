"""Testing for k-nearest-neighbors."""

import numpy as np
from ..knn import KNeighborsClassifier


def test_KNeighborsClassifier():
    """Test 'KNeighborsClassifier' class."""
    X = np.arange(40).reshape(8, 5)
    y = [0, 0, 0, 1, 1, 0, 1, 1]
    X_test = X + 0.1

    # Test 1: metric='euclidean'
    knn = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 2: metric='dtw'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 3: metric='dtw_classic'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw_classic')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 4: metric='dtw_sakoechiba'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw_sakoechiba')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 5: metric='dtw_itakura'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw_itakura')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 6: metric='dtw_multiscale'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw_multiscale')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 6: metric='dtw_fast'
    knn = KNeighborsClassifier(n_neighbors=1, metric='dtw_fast')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)

    # Test 6: metric='boss'
    knn = KNeighborsClassifier(n_neighbors=1, metric='boss')
    arr_actual = knn.fit(X, y).predict(X_test)
    arr_desired = y
    np.testing.assert_array_equal(arr_actual, arr_desired)
