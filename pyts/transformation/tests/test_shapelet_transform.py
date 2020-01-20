import numpy as np
import pytest
import re
from pyts.transformation import ShapeletTransform
from pyts.transformation.shapelet_transform import (
    _extract_all_shapelets, _derive_shapelet_distances, _derive_all_distances,
    _remove_similar_shapelets
)

rng = np.random.RandomState(42)
X = rng.randn(4, 8)
y = rng.randint(2, size=4)
X_large = rng.randn(11, 8)
y_large = rng.randint(2, size=11)


@pytest.mark.parametrize(
    'params, res_desired',
    [({'x': np.arange(5), 'window_sizes': (2,),
       'window_steps': (1,), 'n_timestamps': 5},
      {'shapelets': ([[0, 1], [1, 2], [2, 3], [3, 4]],),
       'lengths': (np.full(4, 2),), 'start_idx': [np.arange(4)],
       'end_idx': [np.arange(2, 6)]}),

     ({'x': np.arange(5), 'window_sizes': (2, 3),
       'window_steps': (1, 2), 'n_timestamps': 5},
      {'shapelets': ([[0, 1], [1, 2], [2, 3], [3, 4]], [[0, 1, 2], [2, 3, 4]]),
       'lengths': ([2, 2, 2, 2], [3, 3]), 'start_idx': [(0, 1, 2, 3), (0, 2)],
       'end_idx': [(2, 3, 4, 5), (3, 5)]}),

     ({'x': np.arange(4)[::-1], 'window_sizes': (1, 3),
       'window_steps': (2, 1), 'n_timestamps': 4},
      {'shapelets': ([[3], [1]], [[3, 2, 1], [2, 1, 0]]),
       'lengths': ([1, 1], [3, 3]), 'start_idx': [(0, 2), (0, 1)],
       'end_idx': [(1, 3), (3, 4)]})]
)
def test_extract_all_shapelets(params, res_desired):
    """Test that the actual results are the expected ones."""
    res_actual = _extract_all_shapelets(**params)

    for arr_actual, arr_desired in zip(res_actual[0],
                                       res_desired['shapelets']):
        np.testing.assert_array_equal(arr_actual, arr_desired)

    for arr_actual, arr_desired in zip(res_actual[1], res_desired['lengths']):
        np.testing.assert_array_equal(arr_actual, arr_desired)

    for arr_actual, arr_desired in zip(res_actual[2],
                                       res_desired['start_idx']):
        np.testing.assert_array_equal(arr_actual, arr_desired)

    for arr_actual, arr_desired in zip(res_actual[3], res_desired['end_idx']):
        np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'X, shapelet, arr_desired',
    [(np.ones((5, 3, 2)), np.ones(2), np.zeros(5)),
     (np.ones((5, 3, 2)), np.zeros(2), np.ones(5)),
     (2 * np.ones((5, 3, 2)), np.zeros(2), 4 * np.ones(5)),
     (np.arange(30).reshape(2, 3, 5), np.arange(-2, 3), [4, 289]),
     (np.arange(30).reshape(2, 3, 5), np.zeros(5), [6, 291])]
)
def test_derive_shapelet_distances(X, shapelet, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _derive_shapelet_distances(X, shapelet)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': np.arange(6, dtype='float64').reshape(2, 3), 'window_sizes': (2,),
       'shapelets': (np.array([[1, 2], [2, 1]]),),
       'lengths': (np.array([2, 2]),), 'fit': True},
      np.sqrt([[0, 1], [4, 5]])),

     ({'X': np.arange(8, dtype='float64').reshape(2, 4),
       'window_sizes': (2, 3), 'lengths': np.array([2, 3]), 'fit': False,
       'shapelets': (np.array([1, 2]), np.array([2, 1, 2]))},
      np.sqrt([[0, 1], [9, 12]]))
     ]
)
def test_derive_all_distances(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _derive_all_distances(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'scores': np.arange(10), 'start_idx': np.arange(10),
       'end_idx': np.arange(2, 12)}, [9, 7, 5, 3, 1]),
     ({'scores': np.arange(10), 'start_idx': np.arange(10),
       'end_idx': np.arange(3, 13)}, [9, 6, 3, 0]),
     ({'scores': np.arange(10), 'start_idx': np.arange(10),
       'end_idx': np.arange(4, 14)}, [9, 5, 1]),
     ({'scores': np.arange(10)[::-1], 'start_idx': np.arange(10),
       'end_idx': np.arange(4, 14)}, [0, 4, 8])]
)
def test_remove_similar_shapelets(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _remove_similar_shapelets(**params)
    np.testing.assert_array_equal(arr_actual, arr_desired)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_shapelets': 'yolo'}, TypeError,
      "'n_shapelets' must be 'auto' or an integer."),

     ({'n_shapelets': -2}, ValueError,
      "If 'n_shapelets' is an integer, it must be a positive integer "
      "(got -2)."),

     ({'criterion': 'mse'}, ValueError,
      "'criterion' must be either 'mutual_info' or 'anova' (got mse)."),

     ({'window_sizes': 'mse'}, TypeError,
      "'window_sizes' must be 'auto', a list, a tuple or a numpy.ndarray."),

     ({'window_sizes': [[2, 3, 4]]}, ValueError,
      "'window_sizes' must be one-dimensional."),

     ({'window_sizes': np.array([2, 3]) + 1j}, ValueError,
      "The elements of 'window_sizes' must be integers or floats."),

     ({'window_sizes': [0.5, 2.0]}, ValueError,
      "If the elements of 'window_sizes' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [0.5, -0.5]}, ValueError,
      "If the elements of 'window_sizes' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [3, 40]}, ValueError,
      "If the elements of 'window_sizes' are integers, they all must be "
      "greater than 0 and lower than or equal to n_timestamps."),

     ({'window_sizes': [3, -3]}, ValueError,
      "If the elements of 'window_sizes' are integers, they all must be "
      "greater than 0 and lower than or equal to n_timestamps."),

     ({'window_steps': {0: 0}}, TypeError,
      "'window_steps' must be None or array-like."),

     ({'window_steps': [1, 2]}, ValueError,
      "'window_steps' must be None if window_sizes='auto'."),

     ({'window_sizes': [3, 5], 'window_steps': [[1, 2]]}, ValueError,
      "'window_steps' must be one-dimensional."),

     ({'window_sizes': [3, 5], 'window_steps': [1, 2, 3]}, ValueError,
      "If 'window_steps' is not None, it must have the same size as "
      "'window_sizes'."),

     ({'window_sizes': [3, 5], 'window_steps': [[1, 2]]}, ValueError,
      "'window_steps' must be one-dimensional."),

     ({'window_sizes': [3, 5], 'window_steps': [1, 2 + 1j]}, ValueError,
      "If 'window_steps' is not None, the elements of 'window_steps' must "
      "be integers or floats."),

     ({'window_sizes': [3, 5], 'window_steps': [0.5, 2.0]}, ValueError,
      "If the elements of 'window_steps' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [3, 5], 'window_steps': [0.5, -0.5]}, ValueError,
      "If the elements of 'window_steps' are floats, they all must be greater "
      "than 0 and lower than or equal to 1."),

     ({'window_sizes': [3, 5], 'window_steps': [3, 45]}, ValueError,
      "If the elements of 'window_steps' are integers, they all must be "
      "greater than 0 and lower than or equal to n_timestamps."),

     ({'window_sizes': [3, 5], 'window_steps': [3, -3]}, ValueError,
      "If the elements of 'window_steps' are integers, they all must be "
      "greater than 0 and lower than or equal to n_timestamps."),

     ({'verbose': '0'}, ValueError,
      "'verbose' must be a positive integer (got 0)."),

     ({'verbose': -1}, ValueError,
      "'verbose' must be a positive integer (got -1)."),

     ({'n_jobs': '0'}, TypeError,
      "'n_jobs' must be None or an integer.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    shapelet = ShapeletTransform(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        shapelet._check_params(X, y)


@pytest.mark.parametrize(
    'params',
    [{'window_sizes': [3, 5], 'sort': True},
     {'window_sizes': [0.5], 'window_steps': None, 'criterion': 'anova'},
     {}]
)
def test_fit_transform(params):
    """Test that 'fit_transform' and 'fit' then 'transform' yield same res."""
    shapelet_1 = ShapeletTransform(random_state=42, **params)
    shapelet_2 = ShapeletTransform(random_state=42, **params)

    X_fit_transform = shapelet_1.fit_transform(X, y)
    X_fit_then_transform = shapelet_2.fit(X, y).transform(X)

    # Test that the transformation are identical
    np.testing.assert_allclose(X_fit_transform, X_fit_then_transform,
                               atol=1e-5, rtol=0.)

    # Test that the shapelets are identical
    for (shap_1, shap_2) in zip(shapelet_1.shapelets_, shapelet_2.shapelets_):
        np.testing.assert_allclose(shap_1, shap_2, atol=1e-5, rtol=0.)

    # Test that the remaining attributes are identical
    np.testing.assert_allclose(shapelet_1.scores_, shapelet_2.scores_,
                               atol=1e-5, rtol=0.)
    np.testing.assert_array_equal(shapelet_1.indices_, shapelet_2.indices_)
    shapelets_not_none = ((shapelet_1.window_range_ is not None) and
                          (shapelet_2.window_range_ is not None))
    if shapelets_not_none:
        np.testing.assert_array_equal(shapelet_1.window_range_,
                                      shapelet_2.window_range_)


@pytest.mark.parametrize(
    'params, X, y, fewer_shapelets, attr_expected',
    [({'n_shapelets': 5, 'window_sizes': [3]}, X, y, False,
      {'n_shapelets': 5, 'indices_shape': (5, 3), 'scores_size': 5}),

     ({'n_shapelets': 500, 'window_sizes': [3]}, X, y, True,
      {'n_shapelets': 500, 'indices_shape': (500, 3), 'scores_size': 500}),

     ({'n_shapelets': 'auto', 'window_sizes': [3]}, X, y, False,
      {'n_shapelets': 4, 'indices_shape': (4, 3), 'scores_size': 4}),

     ({'n_shapelets': 'auto'}, X, y, False,
      {'n_shapelets': 4, 'indices_shape': (4, 3), 'scores_size': 4}),

     ({'n_shapelets': 'auto'}, X_large, y_large, False,
      {'n_shapelets': 4, 'indices_shape': (4, 3), 'scores_size': 4})]
)
def test_attributes_shape(params, X, y, fewer_shapelets, attr_expected):
    """Test the attributes of a ShapaletTransform instance."""
    shapelet = ShapeletTransform(**params)
    shapelet.fit(X, y)

    # Check 'window_range_'
    window_sizes_auto = (isinstance(shapelet.window_sizes, str) and
                         shapelet.window_sizes == 'auto')
    if window_sizes_auto:
        assert isinstance(shapelet.window_range_, tuple)
        assert (0 < shapelet.window_range_[0]
                <= shapelet.window_range_[1] <= 8)
    else:
        assert shapelet.window_range_ is None

    # Check other attributes
    n_shapelets_actual = len(shapelet.shapelets_)
    if fewer_shapelets:
        assert n_shapelets_actual < attr_expected['n_shapelets']
        assert shapelet.indices_.shape[0] < attr_expected['indices_shape'][0]
        assert shapelet.scores_.size < attr_expected['scores_size']
    else:
        assert n_shapelets_actual == attr_expected['n_shapelets']
        assert shapelet.indices_.shape == attr_expected['indices_shape']
        assert shapelet.scores_.size == attr_expected['scores_size']
