"""Testing for Time Series Forest."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.classification import TimeSeriesForest
from pyts.classification.time_series_forest import WindowFeatureExtractor


n_samples, n_timestamps = 5, 20
X_arange = np.arange(n_samples * n_timestamps).reshape(n_samples, n_timestamps)
X_ones = np.ones((n_samples, n_timestamps))
y = [0, 0, 1, 1, 2]


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_windows': '4'}, TypeError,
      "'n_windows' must be an integer or a float."),

     ({'min_window_size': None}, TypeError,
      "'min_window_size' must be an integer or a float."),

     ({'n_windows': 0}, ValueError,
      "If 'n_windows' is an integer, it must be positive (got 0)."),

     ({'n_windows': -3}, ValueError,
      "If 'n_windows' is an integer, it must be positive (got -3)."),

     ({'n_windows': -0.5}, ValueError,
      "If 'n_windows' is a float, it must be greater than 0 (got -0.5)."),

     ({'n_windows': -2.}, ValueError,
      "If 'n_windows' is a float, it must be greater than 0 (got -2.0)."),

     ({'min_window_size': 0}, ValueError,
      "If 'min_window_size' is an integer, it must be greater than or equal "
      "to 1 and lower than or equal to n_timestamps (got 0)."),

     ({'min_window_size': -87}, ValueError,
      "If 'min_window_size' is an integer, it must be greater than or equal "
      "to 1 and lower than or equal to n_timestamps (got -87)."),

     ({'min_window_size': 32}, ValueError,
      "If 'min_window_size' is an integer, it must be greater than or equal "
      "to 1 and lower than or equal to n_timestamps (got 32)."),

     ({'min_window_size': 0.}, ValueError,
      "If 'min_window_size' is a float, it must be greater than 0 and "
      "lower than or equal to 1 (got 0.0)."),

     ({'min_window_size': 1.2}, ValueError,
      "If 'min_window_size' is a float, it must be greater than 0 and "
      "lower than or equal to 1 (got 1.2)."),

     ({'min_window_size': -0.3}, ValueError,
      "If 'min_window_size' is a float, it must be greater than 0 and "
      "lower than or equal to 1 (got -0.3).")]
)
def test_parameter_check(params, error, err_msg):
    """Test parameter validation."""
    fe = WindowFeatureExtractor(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        fe.fit(X_arange)


@pytest.mark.parametrize(
    'params',
    [{'min_window_size': 5},
     {'min_window_size': 15, 'n_windows': 80},
     {'min_window_size': 0.5},
     {'min_window_size': 0.5, 'n_windows': 20}]
)
def test_indices_window_feature_extractor(params):
    """Test that the indices are compatible with the input parameters."""
    fe = WindowFeatureExtractor(**params).fit(X_arange)
    indices = fe.indices_

    min_window_size = fe.get_params()['min_window_size']
    if isinstance(min_window_size, (float, np.floating)):
        min_window_size = np.ceil(min_window_size * n_timestamps)

    np.testing.assert_array_less(-1, indices[:, 0])
    np.testing.assert_array_less(indices[:, 1], n_timestamps + 1)
    np.testing.assert_array_less(min_window_size,
                                 indices[:, 1] - indices[:, 0] + 1)


@pytest.mark.parametrize(
    'params',
    [{'n_windows': 0.5},
     {'n_windows': 1.8},
     {'n_windows': 100, 'min_window_size': 0.2}]
)
def test_actual_results_random_indices(params):
    """Test the actual results with random indices."""
    fe = WindowFeatureExtractor(**params).fit(X_ones)

    n_windows = fe.get_params()['n_windows']
    if isinstance(n_windows, (float, np.floating)):
        n_windows = int(np.ceil(n_windows * n_timestamps))

    min_window_size = fe.get_params()['min_window_size']
    if isinstance(min_window_size, (float, np.floating)):
        min_window_size = int(np.ceil(min_window_size * n_timestamps))

    # With an array full of ones
    arr_actual = fe.transform(X_ones)
    arr_desired = np.tile(
        np.c_[np.ones(n_samples), np.zeros((n_samples, 2))], n_windows
    )
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)

    # With an array arange-like, test the slope
    arr_actual = fe.transform(X_arange)
    if min_window_size > 1:
        np.testing.assert_allclose(arr_actual[:, 2::3], 1, atol=1e-5, rtol=0)
    else:
        assert np.all(np.isin(arr_actual[:, 2::3], [0, 1]))


@pytest.mark.parametrize(
    'indices, arr_desired',
    [([[0, 12]], [[2.125, 1.29301005, 0.25], [2.5, 1.32287566, 0]]),

     ([[0, 1], [0, 2], [8, 12]],
      [[0, 0, 0, 0.5, 0.5, 1, 3.25, 1.08972474, 0.9],
       [4, 0, 0, 3, 1, -2, 2.25, 1.08972474, 0.5]])]
)
def test_actual_results_fixed_indices(indices, arr_desired):
    """Test the actual results with fixed indices."""
    X = [[0, 1, 3, 2, 1, 3, 1.5, 1, 2, 3, 3, 5],
         [4, 2, 0, 2, 3, 5, 3.0, 2, 1, 2, 4, 2]]
    fe = WindowFeatureExtractor()
    fe.indices_ = np.asarray(indices)
    arr_actual = fe.transform(X)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0)


@pytest.mark.parametrize('params', [{}, {'oob_score': True}, {'max_depth': 3}])
def test_attributes_time_series_forest(params):
    """Test the attributes of a fitted instance of TimeSeriesForest."""
    real_attributes = ['estimator_', 'classes_', 'estimators_',
                       'feature_importances_', 'indices_', 'n_features_in_',
                       'oob_decision_function_', 'oob_score_']
    fake_attributes = ['yolo', 'whoopsy', 'mistake_were_made_']

    clf = TimeSeriesForest(n_estimators=30, random_state=42, **params)
    clf.fit(X_arange, y)

    for attribute in real_attributes:
        assert hasattr(clf, attribute)

    for attribute in fake_attributes:
        assert not hasattr(clf, attribute)


def test_methods_time_series_forest():
    """Test the supported methods of TimeSeriesForest."""
    clf = TimeSeriesForest(n_estimators=7).fit(X_arange, y)
    assert clf.apply(X_arange).shape == (n_samples, 7)
    assert clf.decision_path(X_arange)[0].shape[0] == n_samples
    assert clf.decision_path(X_arange)[1].shape == (8,)
    assert clf.predict(X_arange).shape == (n_samples,)
    assert clf.predict_proba(X_arange).shape == (n_samples, 3)
    assert isinstance(clf.score(X_arange, y), (float, np.floating))
