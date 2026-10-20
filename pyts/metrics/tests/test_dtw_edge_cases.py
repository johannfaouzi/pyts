"""Edge cases of Dynamic Time Warping: methods, regions and short series."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest

from pyts.metrics import dtw, itakura_parallelogram, sakoe_chiba_band, show_options
from pyts.metrics._dtw import _dtw_region


def _reference_dtw(x, y, region=None):
    """Brute-force DTW, optionally restricted to a region."""
    x, y = np.asarray(x, dtype='float64'), np.asarray(y, dtype='float64')
    acc = np.full((x.size, y.size), np.inf)
    for i in range(x.size):
        columns = range(y.size) if region is None else range(*region[:, i])
        for j in columns:
            cost = (x[i] - y[j]) ** 2
            if i == 0 and j == 0:
                acc[i, j] = cost
                continue
            previous = [acc[i - 1, j]] if i > 0 else []
            previous += [acc[i, j - 1]] if j > 0 else []
            previous += [acc[i - 1, j - 1]] if i > 0 and j > 0 else []
            acc[i, j] = cost + min(previous)
    return np.sqrt(acc[-1, -1])


def _region_mask(region, n_timestamps_1, n_timestamps_2):
    mask = np.zeros((n_timestamps_1, n_timestamps_2), dtype=bool)
    for i in range(n_timestamps_1):
        mask[i, region[0, i] : region[1, i]] = True
    return mask


rng = np.random.RandomState(0)
x, y = rng.randn(6), rng.randn(4)


@pytest.mark.parametrize(
    'region',
    [
        sakoe_chiba_band(6, 4, window_size=0.5),
        itakura_parallelogram(6, 4),
        [[0, 0, 1, 1, 2, 3], [2, 3, 3, 4, 4, 4]],
    ],
)
def test_dtw_region_method(region):
    """``method='region'`` uses the provided constraint region."""
    options = {'region': region}
    expected = _dtw_region(x, y, region=region)
    np.testing.assert_allclose(dtw(x, y, method='region', options=options), expected)
    np.testing.assert_allclose(
        _reference_dtw(x, y, np.asarray(region)), expected, rtol=1e-12
    )
    # A precomputed cost is summed along the path, without a square root.
    cost = np.subtract.outer(x, y) ** 2
    np.testing.assert_allclose(
        dtw(
            dist='precomputed', precomputed_cost=cost, method='region', options=options
        ),
        expected**2,
    )


def test_dtw_region_method_without_region():
    """Without a region, ``method='region'`` is the classic DTW."""
    assert dtw(x, y, method='region') == dtw(x, y)
    assert dtw(x, y, method='region', options={'region': None}) == dtw(x, y)


def test_dtw_region_method_validation():
    err_msg = "The shape of 'region' must be equal to (2, n_timestamps_1) (got (1, 2))."
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        dtw(x, y, method='region', options={'region': [[0, 1]]})
    with pytest.raises(TypeError, match="unexpected keyword argument 'max_slope'"):
        dtw(x, y, method='region', options={'max_slope': 2})


def test_show_options_region():
    """The 'region' options are shown, not the Itakura ones."""
    text = show_options('region', disp=False)
    assert text.startswith('\nregion\n======\n')
    assert 'region : None or array-like' in text
    assert 'max_slope' not in text


@pytest.mark.parametrize('max_slope', [1, 1.5, 2, 4])
@pytest.mark.parametrize('n_timestamps_2', range(2, 12))
def test_itakura_two_timestamps(n_timestamps_2, max_slope):
    """A first series with two time points has a feasible parallelogram."""
    region = itakura_parallelogram(2, n_timestamps_2, max_slope=max_slope)
    if n_timestamps_2 == 2:
        np.testing.assert_array_equal(region, [[0, 1], [1, 2]])
    else:
        # The parallelogram is symmetric with respect to swapping the series.
        swapped = itakura_parallelogram(n_timestamps_2, 2, max_slope=max_slope)
        np.testing.assert_array_equal(
            _region_mask(region, 2, n_timestamps_2),
            _region_mask(swapped, n_timestamps_2, 2).T,
        )
    x_short, y_long = rng.randn(2), rng.randn(n_timestamps_2)
    options = {'max_slope': max_slope}
    actual = dtw(x_short, y_long, method='itakura', options=options)
    assert np.isfinite(actual)
    np.testing.assert_allclose(actual, _reference_dtw(x_short, y_long, region))
    np.testing.assert_allclose(
        actual, dtw(y_long, x_short, method='itakura', options=options)
    )


def test_itakura_two_by_one():
    """Both rows of a 2x1 parallelogram contain the only column."""
    np.testing.assert_array_equal(itakura_parallelogram(2, 1), [[0, 0], [1, 1]])


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'n_timestamps_1': 2, 'n_timestamps_2': 1},
            ValueError,
            "'n_timestamps_2' must be an integer greater than or equal to 2.",
        ),
        (
            {'n_timestamps_1': 5, 'n_timestamps_2': 0},
            ValueError,
            "'n_timestamps_2' must be an integer greater than or equal to 2.",
        ),
        (
            {'n_timestamps_1': 5, 'n_timestamps_2': 2.5},
            TypeError,
            "'n_timestamps_2' must be an integer.",
        ),
    ],
)
def test_sakoe_chiba_band_n_timestamps_2(params, error, err_msg):
    """``n_timestamps_2`` is validated like ``n_timestamps_1``."""
    with pytest.raises(error, match=re.escape(err_msg)):
        sakoe_chiba_band(**params)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [
        (
            {'n_timestamps_1': 5, 'n_timestamps_2': 0},
            ValueError,
            "'n_timestamps_2' must be an integer greater than or equal to 1.",
        ),
        (
            {'n_timestamps_1': 5, 'n_timestamps_2': 2.5},
            TypeError,
            "'n_timestamps_2' must be an integer.",
        ),
    ],
)
def test_itakura_parallelogram_n_timestamps_2(params, error, err_msg):
    with pytest.raises(error, match=re.escape(err_msg)):
        itakura_parallelogram(**params)


def test_sakoechiba_length_one_series_error():
    err_msg = "'n_timestamps_2' must be an integer greater than or equal to 2."
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        dtw([1.0, 2.0, 3.0], [2.0], method='sakoechiba')


@pytest.mark.parametrize('method', ['classic', 'fast'])
@pytest.mark.parametrize(
    'x_short, y_short',
    [([1.0, 2.0, 3.0], [2.0]), ([2.0], [1.0, 2.0, 3.0]), ([2.0], [5.0])],
)
def test_length_one_series(method, x_short, y_short):
    """DTW is defined when a series has a single time point."""
    actual, path = dtw(x_short, y_short, method=method, return_path=True)
    np.testing.assert_allclose(actual, _reference_dtw(x_short, y_short))
    n_timestamps_1, n_timestamps_2 = len(x_short), len(y_short)
    np.testing.assert_array_equal(path[:, 0], [0, 0])
    np.testing.assert_array_equal(path[:, -1], [n_timestamps_1 - 1, n_timestamps_2 - 1])


def test_length_one_series_precomputed():
    """A precomputed cost is summed along the path (no square root)."""
    cost = np.array([[1.0], [0.0], [1.0]])
    assert dtw(dist='precomputed', precomputed_cost=cost) == 2.0


@pytest.mark.parametrize(
    'n_timestamps_1, n_timestamps_2, resolution',
    [(6, 6, 6), (10, 10, 10), (6, 3, 3), (3, 3, 3), (10, 4, 20), (4, 10, 20)],
)
def test_multiscale_resolution_down_to_one_point(
    n_timestamps_1, n_timestamps_2, resolution
):
    """A resolution that reduces a series to a single point is valid.

    When it reduces both series to a single point, the region is the whole
    cost matrix, which is the classic DTW.
    """
    x_ms, y_ms = rng.randn(n_timestamps_1), rng.randn(n_timestamps_2)
    actual = dtw(x_ms, y_ms, method='multiscale', options={'resolution': resolution})
    classic = dtw(x_ms, y_ms)
    assert np.isfinite(actual) and actual >= classic - 1e-12
    if resolution >= max(n_timestamps_1, n_timestamps_2):
        np.testing.assert_allclose(actual, classic)
