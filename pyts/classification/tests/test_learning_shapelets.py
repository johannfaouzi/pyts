"""Testing for Learning Time-Series Shapelets algorithm."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re

from sklearn.exceptions import ConvergenceWarning

from pyts.classification.learning_shapelets import (
    CrossEntropyLearningShapelets
)
from pyts.classification import LearningShapelets
from pyts.classification.learning_shapelets import (
    _expit, _xlogy, _softmin, _softmin_grad, _softmax,
    _derive_shapelet_distances, _derive_all_squared_distances,
    _reshape_list_shapelets, _reshape_array_shapelets,
    _loss, _grad_weights, _grad_shapelets
)


X_multi = np.arange(20, dtype='float64').reshape(4, 5)
y_multi = np.arange(4)
X_bin = X_multi[:2]
y_bin = y_multi[:2]
n_classes = 2
shapelets = (np.array([[4., 6.]]), np.array([[0., 1., 3., 3.]]))
lengths = (np.array([2]), np.array([4]))


@pytest.mark.parametrize(
    'x, arr_desired',
    [(0, 0.5),
     (np.arange(10), 1 / (1 + np.exp(- np.arange(10)))),
     (np.log(np.arange(1, 10)), 1 / (1 + 1 / np.arange(1, 10))),
     (np.log(1 / np.arange(1, 10)), 1 / (1 + np.arange(1, 10)))]
)
def test_expit(x, arr_desired):
    arr_actual = _expit(x)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'x, y, arr_desired',
    [(0, 3, 0),
     (3, 1, 0),
     (np.arange(6), np.arange(6, 12), np.arange(6) * np.log(np.arange(6, 12))),
     (np.array([0, 1]), np.array([2, 1]), [0, 0])]
)
def test_xlogy(x, y, arr_desired):
    arr_actual = _xlogy(x, y)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'arr, alpha, min_desired',
    [(np.arange(100), -50, 0),
     (np.arange(-100, 100), -50, -100),
     (np.arange(0, 10000, 100), -500, 0),
     (np.linspace(2, 3, 50), -500, 2)]
)
def test_softmin(arr, alpha, min_desired):
    min_actual = _softmin(arr, alpha)
    np.testing.assert_allclose(min_actual, min_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'arr, alpha, grad_desired',
    [(np.arange(100), -50, [1] + [0] * 99),
     (np.arange(-100, 100), -50, [1] + [0] * 199),
     (np.arange(0, 10000, 100), -500, [1] + [0] * 99),
     (np.linspace(2, 3, 50), -5000, [1] + [0] * 49)]
)
def test_softmin_grad(arr, alpha, grad_desired):
    grad_actual = _softmin_grad(arr, alpha)
    np.testing.assert_allclose(grad_actual, grad_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize(
    'arr, min_desired',
    [([np.log(np.arange(1, 5))], [np.arange(0.1, 0.5, 0.1)]),
     ([np.log(np.arange(10, 14))], [np.arange(10, 14) / 46]),
     ([np.log(np.arange(1, 10))], [np.arange(1, 10) / 45])]
)
def test_softmax(arr, min_desired):
    arr = np.asarray(arr)
    n_samples, n_classes = arr.shape
    min_actual = _softmax(arr, n_samples, n_classes)
    np.testing.assert_allclose(min_actual, min_desired, atol=1e-5, rtol=0)


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
    arr_actual = _derive_shapelet_distances(X, shapelet, alpha=-100)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': np.arange(6, dtype='float64').reshape(2, 3), 'n_samples': 2,
       'n_timestamps': 3, 'shapelets': (np.array([[1, 2], [2, 1]]),),
       'lengths': (np.array([2, 2]),)},
      [[0, 4], [1, 5]]),

     ({'X': np.arange(8, dtype='float64').reshape(2, 4), 'n_samples': 2,
       'n_timestamps': 4, 'lengths': (np.array([2]), np.array([3])),
       'shapelets': (np.array([[1, 2]]), np.array([[2, 1, 2]]))},
      [[0, 9], [1, 12]])
     ]
)
def test_derive_all_squared_distances(params, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _derive_all_squared_distances(**params, alpha=-100)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'shapelets, lengths, res_desired',
    [(np.arange(13), (np.array([3, 3, 3]), np.array([2, 2])),
      [np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
       np.array([[9, 10], [11, 12]])]),

     (np.arange(10), (np.array([5, 5]),),
      [np.arange(10).reshape(2, 5)])]
)
def test_reshape_list_shapelets(shapelets, lengths, res_desired):
    """Test that the actual results are the expected ones."""
    res_actual = _reshape_list_shapelets(shapelets, lengths)
    assert len(res_actual) == len(res_desired)
    for arr_actual, arr_desired in zip(res_actual, res_desired):
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'shapelets, lengths, arr_desired',
    [((np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
       np.array([[9, 10], [11, 12]])),
      (np.array([3, 3, 3]), np.array([2, 2])), np.arange(13)),

     ((np.arange(10).reshape(2, 5),),
      (np.array([5, 5]),), np.arange(10))]
)
def test_reshape_array_shapelets(shapelets, lengths, arr_desired):
    """Test that the actual results are the expected ones."""
    arr_actual = _reshape_array_shapelets(shapelets, lengths)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'shapelets, lengths',
    [(np.arange(13), (np.array([3, 3, 3]), np.array([2, 2]))),
     (np.arange(10), (np.array([5, 5]),))]
)
def test_reshape_list_shapelets_inverse(shapelets, lengths):
    """Test that the actual results are the expected ones."""
    shapelets_tuple = tuple(_reshape_list_shapelets(shapelets, lengths))
    shapelets_array = _reshape_array_shapelets(shapelets_tuple, lengths)
    np.testing.assert_allclose(shapelets, shapelets_array, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'shapelets, lengths',
    [((np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
       np.array([[9, 10], [11, 12]])),
      (np.array([3, 3, 3]), np.array([2, 2]))),
     ((np.arange(10).reshape(2, 5),), (np.array([5, 5]),))]
)
def test_reshape_array_shapelets_inverse(shapelets, lengths):
    """Test that the actual results are the expected ones."""
    shapelets_array = _reshape_array_shapelets(shapelets, lengths)
    shapelets_tuple = tuple(_reshape_list_shapelets(shapelets_array, lengths))
    assert len(shapelets) == len(shapelets_tuple)
    for arr_actual, arr_desired in zip(shapelets, shapelets_tuple):
        np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, loss_desired',
    [({'weights': np.zeros(2), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, np.log(2)),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, np.log(2)),
     ({'weights': np.zeros((3, 2)), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([2., 2.])}, 2 * np.log(2)),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])},
      0.00861448 / 2 + 5),
     ({'weights': np.array([1, -2, 1]), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])},
      0.02324546 / 2 + 6),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([3., 1.])},
      0.00861448 * 1.5 + 5),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 10, 'sample_weight': np.array([3., 1.])},
      0.00861448 * 1.5 + 0.5),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l1',
       'C': 1, 'sample_weight': np.array([3., 1.])},
      0.00861448 * 1.5 + 3),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l1',
       'C': 10, 'sample_weight': np.array([3., 1.])},
      0.00861448 * 1.5 + 0.3),
     ({'weights': np.array([[-2, -2], [1, 1]]), 'fit_intercept': False,
       'penalty': 'l2', 'C': 1, 'sample_weight': np.array([1., 1.])},
      np.log(2) + 10),
     ({'weights': np.array([[3, 3], [-2, -2], [1, 1]]), 'fit_intercept': True,
       'penalty': 'l2', 'C': 1, 'sample_weight': np.array([1., 1.])},
      np.log(2) + 28)]
)
def test_loss(params, loss_desired):
    """Test that the actual loss is equal to the expected loss."""
    loss_actual = _loss(X_bin, y_bin, n_classes, shapelets=shapelets,
                        lengths=lengths, alpha=-100, intercept_scaling=1,
                        **params)
    np.testing.assert_allclose(loss_actual, loss_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'weights': np.zeros(2), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, [0.5, -5.625]),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, [0, 0.5, -5.625]),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([2., 2.])}, [0, 1, -11.25]),
     ({'weights': np.zeros((3, 2)), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])},
      [[0.5, -0.5], [0.75, -0.75], [5.75, -5.75]]),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])},
      [-3.98927814, 2.00107218]),
     ({'weights': np.array([1, -2, 1]), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])},
      [2.01148868, -3.97127829, 2.00287217]),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([3., 1.])},
      [-3.96783443, 2.00321655]),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l2',
       'C': 10, 'sample_weight': np.array([3., 1.])},
      [-0.36783443, 0.20321655]),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l1',
       'C': 1, 'sample_weight': np.array([3., 1.])},
      [-0.96783443, 1.00321655]),
     ({'weights': np.array([-2, 1]), 'fit_intercept': False, 'penalty': 'l1',
       'C': 10, 'sample_weight': np.array([3., 1.])},
      [-0.06783443, 0.10321655]),
     ({'weights': np.array([[-2, -2], [1, 1]]), 'fit_intercept': False,
       'penalty': 'l2', 'C': 1, 'sample_weight': np.array([1., 1.])},
      [[-3.25, -4.75], [7.75, -3.75]]),
     ({'weights': np.array([[3, 3], [-2, -2], [1, 1]]), 'fit_intercept': True,
       'penalty': 'l2', 'C': 1, 'sample_weight': np.array([1., 1.])},
      [[6.5, 5.5], [-3.25, -4.75], [7.75, -3.75]])]
)
def test_grad_weights(params, arr_desired):
    """Test that the actual gradient is the expected gradient."""
    params['sample_weight'] = params['sample_weight'].reshape(-1, 1)
    arr_actual = _grad_weights(
        X_bin, y_bin, n_classes, shapelets=shapelets, lengths=lengths,
        alpha=-100, intercept_scaling=1, **params
    )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'weights': np.zeros(2), 'fit_intercept': False, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, np.zeros(6)),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, np.zeros(6)),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([2., 2.])}, np.zeros(6)),
     ({'weights': np.zeros((3, 2)), 'fit_intercept': True, 'penalty': 'l2',
       'C': 1, 'sample_weight': np.array([1., 1.])}, np.zeros(6)),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l1',
       'C': 1, 'sample_weight': np.array([2., 2.])}, np.zeros(6)),
     ({'weights': np.zeros(3), 'fit_intercept': True, 'penalty': 'l2',
       'C': 10, 'sample_weight': np.array([2., 2.])}, np.zeros(6)),
     ({'weights': np.array([3., 0., 0.]), 'fit_intercept': True,
       'penalty': 'l2', 'C': 10, 'sample_weight': np.array([1., 1.])},
      np.zeros(6))]
)
def test_grad_shapelets(params, arr_desired):
    """Test that the actual gradient is the expected gradient."""
    params['sample_weight'] = params['sample_weight'].reshape(-1, 1)
    arr_actual = _grad_shapelets(
        X_bin, y_bin, n_classes, shapelets=shapelets, lengths=lengths,
        alpha=-100, intercept_scaling=1, **params
    )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_shapelets_per_size': 'str'}, TypeError,
      "'n_shapelets_per_size' must be an integer or a float (got str)."),

     ({'n_shapelets_per_size': 8}, ValueError,
      "If 'n_shapelets_per_size' is an integer, it must be "
      "greater than or equal to 1 and lower than or equal to "
      "n_timestamps (got 8)."),

     ({'n_shapelets_per_size': -0.5}, ValueError,
      "If 'n_shapelets_per_size' is a float, it must be greater "
      "than 0 and lower than or equal to 1 (got -0.5)."),

     ({'min_shapelet_length': [6]}, TypeError,
      "'min_shapelet_length' must be an integer or a float (got [6])."),

     ({'min_shapelet_length': 6}, ValueError,
      "If 'min_shapelet_length' is an integer, it must be "
      "greater than or equal to 1 and lower than or equal to "
      "n_timestamps (got 6)."),

     ({'min_shapelet_length': 2.0}, ValueError,
      "If 'min_shapelet_length' is a float, it must be greater "
      "than 0 and lower than or equal to 1 (got 2.0)."),

     ({'shapelet_scale': 'yolo'}, ValueError,
      "'shapelet_scale' must be a positive integer (got yolo)."),

     ({'shapelet_scale': 0}, ValueError,
      "'shapelet_scale' must be a positive integer (got 0)."),

     ({'shapelet_scale': 12}, ValueError,
      "'shapelet_scale' and 'min_shapelet_length' must be "
      "such that shapelet_scale * min_shapelet_length is "
      "smaller than or equal to n_timestamps."),

     ({'penalty': 'l3'}, ValueError,
      "'penalty' must be either 'l2' or 'l1' (got l3)."),

     ({'C': 'l3'}, ValueError,
      "'C' must be a positive float (got l3)."),

     ({'C': -1}, ValueError,
      "'C' must be a positive float (got -1)."),

     ({'tol': 'l4'}, ValueError,
      "'tol' must be a positive float (got l4)."),

     ({'tol': 0}, ValueError,
      "'tol' must be a positive float (got 0)."),

     ({'learning_rate': 'l5'}, ValueError,
      "'learning_rate' must be a positive float (got l5)."),

     ({'learning_rate': 0}, ValueError,
      "'learning_rate' must be a positive float (got 0)."),

     ({'max_iter': '200'}, ValueError,
      "'max_iter' must be a non-negative integer (got 200)."),

     ({'max_iter': 200.}, ValueError,
      "'max_iter' must be a non-negative integer (got 200.0)."),

     ({'alpha': '200'}, ValueError,
      "'alpha' must be a negative float (got 200)."),

     ({'alpha': 200}, ValueError,
      "'alpha' must be a negative float (got 200)."),

     ({'intercept_scaling': '300'}, ValueError,
      "'intercept_scaling' must be a float (got 300)."),

     ({'class_weight': [3]}, ValueError,
      "'class_weight' must be None, a dictionary  or 'balanced' (got [3])."),

     ({'verbose': '200'}, ValueError,
      "'verbose' must be a non-negative integer (got 200)."),

     ({'verbose': -1}, ValueError,
      "'verbose' must be a non-negative integer (got -1)."),

     ({'n_shapelets_per_size': 5}, ValueError,
      "'n_shapelets_per_size' is too high given "
      "'min_shapelet_length' and 'shapelet_scale'.")]
)
def test_parameter_check_cross_entropy(params, error, err_msg):
    """Test parameter validation."""
    clf = CrossEntropyLearningShapelets(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X_bin, y_bin)


@pytest.mark.parametrize('X, y', [(X_bin, y_bin), (X_multi, y_multi)])
@pytest.mark.parametrize('max_iter', [0, 1, 2])
def test_convergence_warning(X, y, max_iter):
    """Test that the ConvergenceWarning is raised."""
    clf = CrossEntropyLearningShapelets(learning_rate=1e-3, tol=1e-9,
                                        random_state=42, max_iter=max_iter)
    msg = ('Maximum number of iterations reached without converging. Increase '
           'the maximum number of iterations.')
    with pytest.warns(ConvergenceWarning, match=msg):
        clf.fit(X, y)


@pytest.mark.filterwarnings("ignore:Maximum number of iterations")
@pytest.mark.parametrize('X, y', [(X_bin, y_bin), (X_multi, y_multi)])
@pytest.mark.parametrize(
    'params, n_shapelets_desired',
    [({'verbose': 1, 'learning_rate': 100, 'max_iter': 1}, 3),
     ({'min_shapelet_length': 1, 'fit_intercept': False}, 3),
     ({'n_shapelets_per_size': 3}, 9),
     ({'shapelet_scale': 5, 'tol': 10, 'max_iter': 2}, 5),
     ({'n_shapelets_per_size': 2, 'shapelet_scale': 3}, 6)]
)
def test_shapes_cross_entropy(X, y, params, n_shapelets_desired):
    """Test that the attributes and returned arrays have the expected shapes"""
    n_samples = X.shape[0]
    if 'max_iter' not in params.keys():
        params['max_iter'] = 0
    clf = CrossEntropyLearningShapelets(**params)

    # Training phase
    clf.fit(X, y)
    n_classes = 1 if len(clf.classes_) == 2 else len(clf.classes_)
    np.testing.assert_array_equal(clf.classes_, y)
    assert clf.shapelets_.shape == (n_shapelets_desired,)
    assert clf.coef_.shape == (n_classes, n_shapelets_desired)
    assert clf.intercept_.shape == (n_classes,)
    assert clf.n_iter_ <= clf.max_iter

    # Test phase
    X_new = clf.decision_function(X)
    y_proba = clf.predict_proba(X)
    y_pred = clf.predict(X)
    if n_classes == 1:
        assert X_new.shape == (n_samples,)
        assert y_proba.shape == (n_samples, 2)
    else:
        assert X_new.shape == (n_samples, n_classes)
        assert y_proba.shape == (n_samples, n_classes)
    assert y_pred.shape == (n_samples,)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'multi_class': 'yolo'}, ValueError,
      "'multi_class' must be either 'multinomial', 'ovr' or 'ovo' "
      "(got yolo)."),

     ({'class_weight': {0: 2, 1: 3}, 'multi_class': 'ovr'}, ValueError,
      "'class_weight' must be None or 'balanced' if 'multi_class' is "
      "either 'ovr' or 'ovo'."),

     ({'class_weight': {0: 2, 1: 3}, 'multi_class': 'ovo'}, ValueError,
      "'class_weight' must be None or 'balanced' if 'multi_class' is "
      "either 'ovr' or 'ovo'."),

     ({'n_jobs': 0}, ValueError,
      "'n_jobs' must be None or an integer not equal to zero (got 0)."),

     ({'n_jobs': 'oops'}, ValueError,
      "'n_jobs' must be None or an integer not equal to zero (got oops).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    clf = LearningShapelets(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X_multi, y_multi)


@pytest.mark.filterwarnings("ignore:Maximum number of iterations")
@pytest.mark.parametrize('X, y', [(X_bin, y_bin), (X_multi, y_multi)])
@pytest.mark.parametrize(
    'params, n_shapelets_desired, n_tasks',
    [({'verbose': 1, 'learning_rate': 100, 'max_iter': 1}, 3, 4),
     ({'multi_class': 'ovr'}, 3, 4),
     ({'multi_class': 'ovo'}, 3, 6),
     ({'fit_intercept': False}, 3, 4),
     ({'n_shapelets_per_size': 3}, 9, 4),
     ({'shapelet_scale': 5}, 5, 4),
     ({'n_shapelets_per_size': 2, 'shapelet_scale': 3}, 6, 4)]
)
def test_shapes(X, y, params, n_shapelets_desired, n_tasks):
    """Test that the attributes and returned arrays have the expected shapes"""
    n_samples = X.shape[0]
    if 'max_iter' not in params.keys():
        params['max_iter'] = 0
    clf = LearningShapelets(**params)

    # Training phase
    clf.fit(X, y)
    n_classes = len(clf.classes_) if len(clf.classes_) > 2 else 1
    np.testing.assert_array_equal(clf.classes_, y)
    if clf._multi_class in ('binary', 'multinomial'):
        assert clf.shapelets_.shape == (1, n_shapelets_desired)
        assert clf.coef_.shape == (n_classes, n_shapelets_desired)
        assert clf.intercept_.shape == (n_classes,)
        assert clf.n_iter_.shape == (1,)
    else:
        assert clf.shapelets_.shape == (n_tasks, n_shapelets_desired)
        assert clf.coef_.shape == (n_tasks, n_shapelets_desired)
        assert clf.intercept_.shape == (n_tasks,)
        assert clf.n_iter_.shape == (n_tasks,)

    # Test phase
    X_new = clf.decision_function(X)
    y_proba = clf.predict_proba(X)
    y_pred = clf.predict(X)
    if clf._multi_class == 'binary':
        assert X_new.shape == (n_samples,)
        assert y_proba.shape == (n_samples, 2)
    else:
        assert X_new.shape == (n_samples, n_classes)
        assert y_proba.shape == (n_samples, n_classes)
    assert y_pred.shape == (n_samples,)
