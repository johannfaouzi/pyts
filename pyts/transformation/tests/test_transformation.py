import numpy as np
import unittest
from pyts.transformation import (StandardScaler, PAA, SAX,
    VSM, GASF, GADF, MTF, RecurrencePlots)


class pytsTransformationTest(unittest.TestCase):

    """Unit tests for pyts.transformation module"""

    def test_StandardScaler(self):
        """Testing 'StandardScaler'"""

        # Parameter
        X = np.arange(0, 7)

        # Test 1
        standardscaler = StandardScaler(epsilon=0.)
        arr_actual = standardscaler.transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(-1.5, 2., 0.5)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        standardscaler = StandardScaler(epsilon=2.)
        arr_actual = standardscaler.transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(-.75, 1., 0.25)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        standardscaler = StandardScaler(epsilon=2.)
        arr_actual = standardscaler.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(-.75, 1., 0.25)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_PAA(self):
        """Testing 'PAA'"""

        # Parameter
        X = np.arange(30)

        # Test 1
        paa = PAA(window_size=2)
        arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(0.5, 30, 2)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        paa = PAA(window_size=3)
        arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(1, 30, 3)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        paa = PAA(window_size=5)
        arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(2, 30, 5)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 4
        paa = PAA(output_size=10)
        arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.arange(1, 30, 3)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 5
        paa = PAA(window_size=4, overlapping=True)
        arr_actual = paa.fit_transform(X[np.newaxis, :])[0]
        arr_desired = np.array([1.5, 4.5, 8.5, 12.5, 15.5, 19.5, 23.5, 27.5])
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_SAX(self):
        """Testing 'SAX'"""

        # Test 1
        X = np.tile(np.arange(3), 5)
        sax = SAX(n_bins=3, quantiles='empirical')
        str_actual = sax.fit_transform(X[np.newaxis, :])[0]
        str_desired = ''.join(["a", "b", "c"] * 5)
        np.testing.assert_string_equal(str_actual, str_desired)

        # Test 2
        X = np.repeat(np.arange(-0.75, 1, 0.5), 3)
        sax = SAX(n_bins=4, quantiles='gaussian')
        str_actual = sax.fit_transform(X[np.newaxis, :])[0]
        str_desired = ''.join([a for a in "abcd" for _ in range(3)])
        np.testing.assert_string_equal(str_actual, str_desired)

    def test_VSM(self):
        """Testing 'VSM'"""

        # Parameter
        X = np.array(["aaabbbcccddd"])

        # Test 1
        window_size = 4
        vsm = VSM(window_size=window_size, numerosity_reduction=False)
        arr_actual = vsm.fit_transform(X)[0]
        arr_desired = np.array([X[0][i: i + window_size] for i in range(len(X[0]) - window_size + 1)])
        np.testing.assert_array_equal(arr_actual, arr_desired)

        # Test 2
        window_size = 2
        vsm = VSM(window_size=window_size, numerosity_reduction=True)
        arr_actual = vsm.fit_transform(X)[0]
        arr_desired = []
        for i in range(len(X[0]) - window_size + 1):
            substring = X[0][i: i + window_size]
            if i == 0:
                arr_desired.append(substring)
            else:
                substring = X[0][i: i + window_size]
                if substring != arr_desired[-1]:
                    arr_desired.append(substring)
        np.testing.assert_array_equal(arr_actual, arr_desired)

    def test_GASF(self):
        "Testing 'GASF'"

        # Parameter
        size = 9
        X = np.linspace(-1, 1, size)

        # Test 1
        ones = np.ones(size)
        gasf = GASF(image_size=size)
        arr_actual = gasf.transform(X[np.newaxis, :])[0]
        arr_desired = np.outer(X, X) - np.outer(np.sqrt(ones - X ** 2),
                                                np.sqrt(ones - X ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        size_new = 3
        ones_new = np.ones(size_new)
        gasf = GASF(image_size=size_new)
        arr_actual = gasf.transform(X[np.newaxis, :])[0]
        X_new = np.linspace(-1, 1, size_new)
        arr_desired = np.outer(X_new, X_new) - np.outer(np.sqrt(ones_new - X_new ** 2),
                                                        np.sqrt(ones_new - X_new ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_GADF(self):
        "Testing 'GADF'"

        # Parameter
        size = 9
        X = np.linspace(-1, 1, size)

        # Test 1
        ones = np.ones(size)
        gadf = GADF(image_size=size)
        arr_actual = gadf.transform(X[np.newaxis, :])[0]
        arr_desired = np.outer(np.sqrt(ones - X ** 2), X) - np.outer(X, np.sqrt(ones - X ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        size_new = 3
        ones_new = np.ones(size_new)
        gadf = GADF(image_size=size_new)
        arr_actual = gadf.transform(X[np.newaxis, :])[0]
        X_new = np.linspace(-1, 1, size_new)
        arr_desired = np.outer(np.sqrt(ones_new - X_new ** 2), X_new)\
                    - np.outer(X_new, np.sqrt(ones_new - X_new ** 2))
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_MTF(self):
        "Testing 'MTF'"

        # Parameter
        size = 9
        X = np.linspace(-1, 1, size)

        # Test 1
        mtf = MTF(image_size=size, n_bins=3)
        arr_actual = mtf.transform(X[np.newaxis, :])[0]
        MTF_arr = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
        MTF_arr = np.multiply(MTF_arr.T, (MTF_arr.sum(axis=1) ** (-1))).T
        arr_desired = np.repeat(np.tile(MTF_arr, 3), 3).reshape(9, 9)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        size_new = 3
        mtf = MTF(image_size=size_new, n_bins=3)
        arr_actual = mtf.transform(X[np.newaxis, :])[0]
        MTF_arr = np.array([[2., 1., 0.], [0., 2., 1.], [0., 0., 2.]])
        arr_desired = np.multiply(MTF_arr.T, (MTF_arr.sum(axis=1) ** (-1))).T
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

    def test_ReccurencePlots(self):
        "Testing 'ReccurencePlots'"

        # Parameter
        size = 9
        X = np.linspace(-1, 1, size)

        # Test 1
        rp = RecurrencePlots(dimension=1)
        arr_actual = rp.transform(X[np.newaxis, :])[0]
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j])
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 2
        percentage = 50
        rp = RecurrencePlots(dimension=1,
                             epsilon='percentage_distance',
                             percentage=percentage)
        arr_actual = rp.transform(X[np.newaxis, :])[0]
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j]) < percentage * (X.max() - X.min()) / 100
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)

        # Test 3
        percentage = 50
        rp = RecurrencePlots(dimension=1,
                             epsilon='percentage_points',
                             percentage=percentage)
        arr_actual = rp.transform(X[np.newaxis, :])[0]
        arr_desired = np.empty((size, size))
        for i in range(size):
            for j in range(size):
                arr_desired[i, j] = abs(X[i] - X[j])
        arr_desired = arr_desired < np.percentile(arr_desired, q=50)
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
