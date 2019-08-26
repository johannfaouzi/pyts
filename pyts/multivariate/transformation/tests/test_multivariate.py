"""Testing for MultivariateTransformer."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from scipy.sparse import csr_matrix
from pyts.classification import SAXVSM
from pyts.image import RecurrencePlot
from pyts.multivariate.transformation import MultivariateTransformer
from pyts.transformation import BOSS


n_samples, n_features, n_timestamps = 40, 3, 30
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'estimator': [BOSS(), RecurrencePlot(), SAXVSM()]},
      ValueError, "Estimator 2 must be a transformer."),

     ({'estimator': [BOSS()]}, ValueError,
      "If 'estimator' is a list, its length must be equal to "
      "the number of features (1 != 3)"),

     ({'estimator': None}, TypeError,
      "'estimator' must be a transformer that inherits from "
      "sklearn.base.BaseEstimator or a list thereof.")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    transformer = MultivariateTransformer(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        transformer.fit_transform(X)


@pytest.mark.parametrize(
    'X, arr_desired',
    [(csr_matrix(np.ones((5, 5))), np.ones((5, 5))),
     (np.ones((5, 5)), np.ones((5, 5)))]
)
def test_array_conversion(X, arr_desired):
    """Test the array conversion static method."""
    arr_actual = MultivariateTransformer._convert_to_array(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'X, err_msg',
    [({}, "Unexpected type for X: dict."),
     ({0}, "Unexpected type for X: set."),
     ([], "Unexpected type for X: list.")]
)
def test_array_conversion_error(X, err_msg):
    """Test the array conversion static method."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        MultivariateTransformer._convert_to_array(X)


@pytest.mark.parametrize(
    'params, shape_desired',
    [({'estimator': RecurrencePlot(dimension=1), 'flatten': False},
      (40, 3, 30, 30)),

     ({'estimator': RecurrencePlot(dimension=6), 'flatten': False},
      (40, 3, 25, 25)),

     ({'estimator': RecurrencePlot(dimension=6), 'flatten': True},
      (40, 3 * 25 * 25)),

     ({'estimator': [RecurrencePlot(dimension=6),
                     RecurrencePlot(dimension=4),
                     RecurrencePlot(dimension=2)], 'flatten': True},
      (40, (25 * 25) + (27 * 27) + (29 * 29)))]
)
def test_shapes(params, shape_desired):
    """Test that the shape of the output is the expected one."""
    transformer = MultivariateTransformer(**params)
    assert transformer.fit(X).transform(X).shape == shape_desired
    assert transformer.fit_transform(X).shape == shape_desired


@pytest.mark.parametrize(
    'params, ndim_desired',
    [({'estimator': RecurrencePlot(), 'flatten': False}, 4),
     ({'estimator': RecurrencePlot(dimension=6), 'flatten': True}, 2),
     ({'estimator': BOSS(), 'flatten': False}, 2),
     ({'estimator': BOSS(), 'flatten': True}, 2),
     ({'estimator': [RecurrencePlot(dimension=6),
                     RecurrencePlot(dimension=4),
                     RecurrencePlot(dimension=2)], 'flatten': True}, 2),
     ({'estimator': [RecurrencePlot(dimension=6),
                     RecurrencePlot(dimension=4),
                     RecurrencePlot(dimension=2)], 'flatten': False}, 2),
     ({'estimator': [RecurrencePlot(dimension=6), BOSS(),
                     RecurrencePlot(dimension=2)], 'flatten': True}, 2)]
)
def test_ndim(params, ndim_desired):
    """Test that the ndim of the output is the expected one."""
    transformer = MultivariateTransformer(**params)
    assert transformer.fit(X).transform(X).ndim == ndim_desired
    assert transformer.fit_transform(X).ndim == ndim_desired


def test_actual_results_without_flatten():
    """Test that the actual results are the expected ones."""
    params = {'estimator': RecurrencePlot(dimension=6), 'flatten': False}
    arr_actual = MultivariateTransformer(**params).fit_transform(X)
    arr_desired = []
    for i in range(n_features):
        arr_desired.append(params['estimator'].transform(X[:, i]))
    arr_desired = np.transpose(arr_desired, axes=(1, 0, 2, 3))
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


def test_actual_results_with_flatten():
    """Test that the actual results are the expected ones."""
    params = {'estimator': RecurrencePlot(dimension=6), 'flatten': True}
    arr_actual = MultivariateTransformer(**params).fit_transform(X)
    arr_desired = []
    for i in range(n_features):
        arr_desired.append(params['estimator'].transform(X[:, i]).reshape(
            (n_samples, -1)))
    arr_desired = np.concatenate(arr_desired, axis=1)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)
