import numpy as np
import unittest
from pyts.utils import (bin_allocation_integers, bin_allocation_alphabet,
    segmentation, mean, arange, paa, sax, num_red, vsm, gaf, mtf,
    dtw, fast_dtw, recurrence_plot)


class pytsUtilsTest(unittest.TestCase):

    """Unit tests for pyts.utils module"""

    def test_bin_allocation_integers(self):
        """Testing 'bin_allocation_integers'"""

        # Test 1
        x = 0
        n_bins = 3
        quantiles = [-0.2, 0.2, 0.3]
        res_actual = bin_allocation_integers(x, n_bins, quantiles)
        res_desired = 1
        np.testing.assert_equal(res_actual, res_desired)

        # Test 2
        x = 0.25
        n_bins = 3
        quantiles = [-0.2, 0.2, 0.3]
        res_actual = bin_allocation_integers(x, n_bins, quantiles)
        res_desired = 2
        np.testing.assert_equal(res_actual, res_desired)

    def test_bin_allocation_alphabet(self):
        """Testing 'bin_allocation_alphabet'"""

        # Test 1
        x = 0
        n_bins = 3
        alphabet = "abc"
        quantiles = [-0.2, 0.2, 0.3]
        res_actual = bin_allocation_alphabet(x, n_bins, alphabet, quantiles)
        res_desired = 'b'
        np.testing.assert_equal(res_actual, res_desired)

        # Test 2
        x = 0.25
        n_bins = 3
        quantiles = [-0.2, 0.2, 0.3]
        res_actual = bin_allocation_alphabet(x, n_bins, alphabet, quantiles)
        res_desired = 'c'
        np.testing.assert_equal(res_actual, res_desired)

    def test_segmentation(self):
        """Testing 'segmentation'"""

        # Test 1
        bounds = np.array([0, 4, 8, 12, 16, 20])
        window_size = 4
        overlapping = False
        res_actual = segmentation(bounds, window_size, overlapping)
        res_desired = np.arange(20).reshape((-1, window_size))
        np.testing.assert_array_equal(res_actual, res_desired)

    def test_mean(self):
        """Testing 'mean'"""

        # Test 1
        ts = np.arange(9)
        indices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        overlapping = True
        res_actual = mean(ts, indices, overlapping)
        res_desired = np.array([1, 4, 7])
        np.testing.assert_array_equal(res_actual, res_desired)

        # Test 2
        ts = np.arange(9)
        indices = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
        overlapping = False
        res_actual = mean(ts, indices, overlapping)
        res_desired = np.array([1, 4, 7])
        np.testing.assert_array_equal(res_actual, res_desired)

    def test_arange(self):
        """Testing 'arange'"""

        # Test 1
        array = np.array([2, 8])
        res_actual = arange(array)
        res_desired = np.arange(2, 8)
        np.testing.assert_array_equal(res_actual, res_desired)

    def test_paa(self):
        """Testing 'paa'"""

        # Parameter
        X = np.arange(30)
        X_size = 30

        # Test 1
        arr_actual = paa(X, X_size, window_size=2, overlapping=0.)
        arr_desired = np.arange(0.5, 30, 2)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        arr_actual = paa(X, X_size, window_size=3, overlapping=0.)
        arr_desired = np.arange(1, 30, 3)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        arr_actual = paa(X, X_size, window_size=5, overlapping=0.)
        arr_desired = np.arange(2, 30, 5)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_sax(self):
        """Testing 'sax'"""

        # Test 1
        X = np.tile(np.arange(3), 5)
        str_actual = sax(X, n_bins=3, quantiles='empirical', alphabet='abc')
        str_desired = ''.join(["a", "b", "c"] * 5)
        np.testing.assert_string_equal(str_actual, str_desired)

        # Test 2
        X = np.repeat(np.arange(-0.75, 1, 0.5), 3)
        str_actual = sax(X, n_bins=4, quantiles='gaussian', alphabet='abcd')
        str_desired = ''.join([a for a in "abcd" for _ in range(3)])
        np.testing.assert_string_equal(str_actual, str_desired)

    def test_num_red(self):
        """Testing 'num_red'"""

        array = np.array(["aaa", "aaa", "aaa", "bbb", "bbb", "ccc", "aaa"])
        arr_actual = num_red(array)
        arr_desired = ["aaa", "bbb", "ccc", "aaa"]
        np.testing.assert_array_equal(arr_actual, arr_desired)

    def test_vsm(self):
        """Testing 'vsm'"""

        # Parameter
        ts_sax = "aaabbbcccddd"
        ts_sax_size = len(ts_sax)

        # Test 1
        window_size = 4
        arr_actual = vsm(ts_sax, ts_sax_size, window_size, numerosity_reduction=False)
        arr_desired = np.array([ts_sax[i: i + window_size] for i in range(ts_sax_size - window_size + 1)])
        np.testing.assert_array_equal(arr_actual, arr_desired)

        # Test 2
        window_size = 2
        arr_actual = vsm(ts_sax, ts_sax_size, window_size, numerosity_reduction=True)
        arr_desired = []
        for i in range(ts_sax_size - window_size + 1):
            substring = ts_sax[i: i + window_size]
            if i == 0:
                arr_desired.append(substring)
            else:
                substring = ts_sax[i: i + window_size]
                if substring != arr_desired[-1]:
                    arr_desired.append(substring)
        np.testing.assert_array_equal(arr_actual, arr_desired)

    def test_gaf(self):
        """Testing 'gaf'"""

        # Parameter
        size = 9
        ts = np.linspace(-1, 1, size)
        overlapping = False
        ones = np.ones(size)

        # Test 1
        arr_actual = gaf(ts, size, size, overlapping, method='s', scale='-1')
        arr_desired = np.outer(ts, ts) - np.outer(np.sqrt(ones - ts ** 2), np.sqrt(ones - ts ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        arr_actual = gaf(ts, size, size, overlapping, method='d', scale='-1')
        arr_desired = np.outer(np.sqrt(ones - ts ** 2), ts) - np.outer(ts, np.sqrt(ones - ts ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_mtf(self):
        """Testing 'mtf'"""

        # Parameter
        size = 9
        ts = np.linspace(-1, 1, size)

        # Test 1
        arr_actual = mtf(ts, size, size, n_bins=3, quantiles='empirical', overlapping=False)
        MTF = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
        MTF = np.multiply(MTF.T, (MTF.sum(axis=1) ** (-1))).T
        arr_desired = np.repeat(np.tile(MTF, 3), 3).reshape(9, 9)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        ts_reversed = ts[::-1]
        arr_actual = mtf(ts_reversed, size, size, n_bins=3,
                                    quantiles='empirical', overlapping=False)
        MTF = np.array([[2., 0., 0.], [1., 2., 0.], [0., 1., 2.]])
        MTF = np.multiply(MTF.T, (MTF.sum(axis=1) ** (-1))).T
        arr_desired = np.repeat(np.tile(MTF[::-1, ::-1], 3), 3).reshape(9, 9)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        size_new = 3
        arr_actual = mtf(ts, size, size_new, n_bins=3, quantiles='empirical', overlapping=False)
        MTF = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
        arr_desired = np.multiply(MTF.T, (MTF.sum(axis=1) ** (-1))).T
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_dtw(self):
        """Testing 'dtw'"""

        # Parameter 1
        x1 = np.array([1, -1, -1, 1, 1])
        x2 = np.array([1, 1, -1, -1, 1])

        # Test 1
        cost_actual = dtw(x1, x2)
        cost_desired = 0
        np.testing.assert_equal(cost_actual, cost_desired)

        # Parameter 2
        x1 = np.array([1, -1, 1, 1, 1, -1])
        x2 = np.ones(6)
        x3 = - np.ones(6)

        # Test 1
        cost_actual = dtw(x1, x2)
        cost_desired = 4
        np.testing.assert_equal(cost_actual, cost_desired)

        # Test 2
        cost_actual = dtw(x1, x3)
        cost_desired = 8
        np.testing.assert_equal(cost_actual, cost_desired)

        # Test 3
        cost_actual = dtw(x2, x3)
        cost_desired = 12
        np.testing.assert_equal(cost_actual, cost_desired)

    def test_fast_dtw(self):
        """Testing 'fast_dtw'"""

        # Parameter
        x1 = np.array([1, -1, 1, 1, 1, -1])
        x2 = np.ones(6)
        x3 = - np.ones(6)

        # Test 1
        cost_actual = fast_dtw(x1, x2, window_size=2)
        cost_desired = 2
        np.testing.assert_equal(cost_actual, cost_desired)

        # Test 2
        cost_actual = fast_dtw(x1, x3, window_size=2)
        cost_desired = 4
        np.testing.assert_equal(cost_actual, cost_desired)

        # Test 3
        cost_actual = fast_dtw(x2, x3, window_size=2)
        cost_desired = 6
        np.testing.assert_equal(cost_actual, cost_desired)

    def test_recurrence_plot(self):
        """Testing 'recurrence_plot'"""

        # Parameter
        size = 9
        X = np.linspace(-1, 1, size)

        # Test 1
        arr_actual = recurrence_plot(X, dimension=1, epsilon=None, percentage=10)
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j])
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        percentage = 50
        epsilon = 'percentage_distance'
        arr_actual = recurrence_plot(X, dimension=1, epsilon=epsilon, percentage=percentage)
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j]) < percentage * (X.max() - X.min()) / 100
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        percentage = 50
        epsilon = 'percentage_points'
        arr_actual = recurrence_plot(X, dimension=1, epsilon=epsilon, percentage=percentage)
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j])
        arr_desired = arr_desired < np.percentile(arr_desired, q=50)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
