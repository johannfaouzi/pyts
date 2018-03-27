import numpy as np
import unittest
from pyts.classification import SAXVSMClassifier, KNNClassifier


class pytsClassificationTest(unittest.TestCase):

    """Unit tests for pyts.classification module"""

    def test_SAXVSMClassifier(self):
        """Testing 'SAXVSMClassifier'"""

        # Parameter
        x1 = np.array(["aaa", "bbb", "bbb"])
        x2 = np.array(["aaa", "bbb", "ccc"])
        x3 = np.array(["aaa", "ddd", "ddd"])
        x4 = np.array(["aaa", "ddd", "eee"])
        X = np.vstack([x1, x2, x3, x4])
        y = np.array([0, 0, 1, 1])

        # Test
        clf = SAXVSMClassifier(norm='l2',
                               use_idf=True,
                               smooth_idf=True,
                               sublinear_tf=False)
        clf.fit(X, y)
        test_index = 2
        np.testing.assert_equal(y[test_index], clf.predict(X[test_index][np.newaxis, :]))
        test_index = np.array([2, 3])
        np.testing.assert_equal(y[test_index], clf.predict(X[test_index]))

    def test_KNNClassifier(self):
        """Testing 'KNNClassifier'"""

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
