"""Testing for Time Series Bag-of-Features."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re
from pyts.classification import TSBF
from pyts.classification.tsbf import IntervalFeatureExtractor
from pyts.classification.tsbf import extract_features, histogram
from sklearn.tree import DecisionTreeClassifier


n_samples, n_timestamps = 5, 20
X_arange = np.arange(n_samples * n_timestamps).reshape(n_samples, n_timestamps)
X_ones = np.ones((n_samples, n_timestamps))
X_oob_proba = np.array(
    [[[0.37, 0.95],
      [0.73, 0.60],
      [0.16, 0.16],
      [0.06, 0.87],
      [0.60, 0.71],
      [0.02, 0.97],
      [0.83, 0.21],
      [0.18, 0.18]],

     [[0.30, 0.52],
      [0.43, 0.29],
      [0.61, 0.14],
      [0.29, 0.37],
      [0.46, 0.79],
      [0.20, 0.51],
      [0.59, 0.05],
      [0.61, 0.17]]]
)
y = [0, 0, 1, 1, 2]


def var(n):
    return np.sqrt((n ** 2 - 1) / 12)


@pytest.mark.parametrize(
    'n, res_desired',
    [(3, 0.816496580927726),
     (6, 1.707825127659933),
     (18, 5.188127472091127)]
)
def test_var(n, res_desired):
    np.testing.assert_allclose(var(n), res_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': X_arange,
       'interval_indices': np.array(
           [[[1, 3, 5], [3, 4, 5], [2, 8, 14]],
            [[3, 5, 7], [4, 5, 6], [8, 14, 20]]]
        ),
       'n_samples': 5, 'n_subseries': 3, 'n_intervals': 3},
      [[1.5, 0.5, 1., 3.5, 0.5, 1., 5.5, 0.5, 1., 3.5, var(6), 1., 7.],
       [3., 0., 0., 4., 0., 0., 5., 0., 0., 4., var(3), 3., 6.],
       [4.5, var(6), 1., 10.5, var(6), 1., 16.5, var(6), 1., 10.5, var(18),
        2., 20.],
       [21.5, 0.5, 1., 23.5, 0.5, 1., 25.5, 0.5, 1., 23.5, var(6), 1., 7.],
       [23., 0., 0., 24., 0., 0., 25., 0., 0., 24., var(3), 3., 6.],
       [24.5, var(6), 1., 30.5, var(6), 1., 36.5, var(6), 1., 30.5, var(18),
        2., 20.],
       [41.5, 0.5, 1., 43.5, 0.5, 1., 45.5, 0.5, 1., 43.5, var(6), 1., 7.],
       [43., 0., 0., 44., 0., 0., 45., 0., 0., 44., var(3), 3., 6.],
       [44.5, var(6), 1., 50.5, var(6), 1., 56.5, var(6), 1., 50.5, var(18),
        2., 20.],
       [61.5, 0.5, 1., 63.5, 0.5, 1., 65.5, 0.5, 1., 63.5, var(6), 1., 7.],
       [63., 0., 0., 64., 0., 0., 65., 0., 0., 64., var(3), 3., 6.],
       [64.5, var(6), 1., 70.5, var(6), 1., 76.5, var(6), 1., 70.5, var(18),
        2., 20.],
       [81.5, 0.5, 1., 83.5, 0.5, 1., 85.5, 0.5, 1., 83.5, var(6), 1., 7.],
       [83., 0., 0., 84., 0., 0., 85., 0., 0., 84., var(3), 3., 6.],
       [84.5, var(6), 1., 90.5, var(6), 1., 96.5, var(6), 1., 90.5, var(18),
        2., 20.]]
      ),

     ({'X': X_ones,
       'interval_indices': np.array(
            [[[3, 6, 9, 12], [9, 10, 11, 12]],
             [[6, 9, 12, 15], [10, 11, 12, 13]]]
         ),
      'n_samples': 5, 'n_subseries': 2, 'n_intervals': 4},
      [[1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 3., 15.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 9., 13.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 3., 15.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 9., 13.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 3., 15.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 9., 13.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 3., 15.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 9., 13.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 3., 15.],
       [1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 9., 13.]])]
)
def test_extract_features(params, arr_desired):
    arr_actual = extract_features(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, arr_desired',
    [({'X': X_oob_proba, 'bins': 2, 'n_bins': 2,
       'n_samples': 2, 'n_classes': 2},
      [[5, 3, 3, 5, 0.36875, 0.58125], [5, 3, 5, 3, 0.43625, 0.355]]),

     ({'X': X_oob_proba, 'bins': np.array([0., 0.5, 1.]), 'n_bins': 2,
       'n_samples': 2, 'n_classes': 2},
      [[5, 3, 3, 5, 0.36875, 0.58125], [5, 3, 5, 3, 0.43625, 0.355]]),

     ({'X': X_oob_proba, 'bins': 3, 'n_bins': 3,
       'n_samples': 2, 'n_classes': 2},
      [[4, 2, 2, 3, 1, 4, 0.36875, 0.58125],
       [3, 5, 0, 4, 3, 1, 0.43625, 0.355]]),

     ({'X': X_oob_proba, 'bins': np.linspace(0., 1., 4), 'n_bins': 3,
       'n_samples': 2, 'n_classes': 2},
      [[4, 2, 2, 3, 1, 4, 0.36875, 0.58125],
       [3, 5, 0, 4, 3, 1, 0.43625, 0.355]]),

     ({'X': X_oob_proba, 'bins': np.array([0., 0.2, 0.8, 1.]), 'n_bins': 3,
       'n_samples': 2, 'n_classes': 2},
      [[4, 3, 1, 2, 3, 3, 0.36875, 0.58125],
       [0, 8, 0, 3, 5, 0, 0.43625, 0.355]]),

     ({'X': X_oob_proba[:1], 'bins': np.array([0., 0.2, 0.8, 1.]), 'n_bins': 3,
       'n_samples': 1, 'n_classes': 2},
      [[4, 3, 1, 2, 3, 3, 0.36875, 0.58125]]),

     ({'X': X_oob_proba[1:], 'bins': np.array([0., 0.2, 0.8, 1.]), 'n_bins': 3,
       'n_samples': 1, 'n_classes': 2},
      [[0, 8, 0, 3, 5, 0, 0.43625, 0.355]])]
)
def test_histogram(params, arr_desired):
    arr_actual = histogram(**params)
    np.testing.assert_allclose(arr_actual, arr_desired, atol=1e-5, rtol=0.)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'min_subsequence_size': '4'}, TypeError,
      "'min_subsequence_size' must be an integer or a float."),

     ({'min_subsequence_size': None}, TypeError,
      "'min_subsequence_size' must be an integer or a float."),

     ({'min_subsequence_size': 0}, ValueError,
      "If 'min_subsequence_size' is an integer, it must be greater than or "
      "equal to 1 and lower than or equal to n_timestamps (got 0)."),

     ({'min_subsequence_size': -3}, ValueError,
      "If 'min_subsequence_size' is an integer, it must be greater than or "
      "equal to 1 and lower than or equal to n_timestamps (got -3)."),

     ({'min_subsequence_size': -0.5}, ValueError,
      "If 'min_subsequence_size' is a float, it must be greater "
      "than 0 and lower than or equal to 1 (got -0.5)."),

     ({'min_subsequence_size': -2.}, ValueError,
      "If 'min_subsequence_size' is a float, it must be greater "
      "than 0 and lower than or equal to 1 (got -2.0)."),

     ({'min_interval_size': '4'}, TypeError,
      "'min_interval_size' must be an integer or a float."),

     ({'min_interval_size': []}, TypeError,
      "'min_interval_size' must be an integer or a float."),

     ({'min_interval_size': 0}, ValueError,
      "If 'min_interval_size' is an integer, it must be positive (got 0)."),

     ({'min_interval_size': -6}, ValueError,
      "If 'min_interval_size' is an integer, it must be positive (got -6)."),

     ({'min_interval_size': 0.}, ValueError,
      "If 'min_interval_size' is a float, it must be greater than 0 "
      "(got 0.0)."),

     ({'min_interval_size': -1.}, ValueError,
      "If 'min_interval_size' is a float, it must be greater than 0 "
      "(got -1.0)."),

     ({'min_subsequence_size': 2, 'min_interval_size': 3}, ValueError,
      "'min_interval_size' must be lower than or equal to "
      "'min_subsequence_size' (3 > 2)."),

     ({'min_subsequence_size': 0.1, 'min_interval_size': 3}, ValueError,
      "'min_interval_size' must be lower than or equal to "
      "'min_subsequence_size' (3 > 2)."),

     ({'min_subsequence_size': 0.2, 'min_interval_size': 10}, ValueError,
      "'min_interval_size' must be lower than or equal to "
      "'min_subsequence_size' (10 > 4)."),

     ({'min_subsequence_size': 0.2, 'min_interval_size': 0.5}, ValueError,
      "'min_interval_size' must be lower than or equal to "
      "'min_subsequence_size' (10 > 4)."),

     ({'n_subsequences': []}, TypeError,
      "'n_subsequences' must be 'auto', an integer or a float."),

     ({'n_subsequences': None}, TypeError,
      "'n_subsequences' must be 'auto', an integer or a float."),

     ({'n_subsequences': 0}, ValueError,
      "If 'n_subsequences' is an integer, it must be positive (got 0)."),

     ({'n_subsequences': -6}, ValueError,
      "If 'n_subsequences' is an integer, it must be positive (got -6)."),

     ({'n_subsequences': 0.}, ValueError,
      "If 'n_subsequences' is a float, it must be greater than 0 "
      "(got 0.0)."),

     ({'n_subsequences': -1.}, ValueError,
      "If 'n_subsequences' is a float, it must be greater than 0 "
      "(got -1.0).")]
)
def test_parameter_check_interval_feature_extractor(params, error, err_msg):
    """Test parameter validation."""
    fe = IntervalFeatureExtractor(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        fe.fit(X_arange)


@pytest.mark.parametrize(
    'params, interval_indices_shape, min_subsequence_size',
    [({'min_subsequence_size': 10, 'min_interval_size': 2,
       'n_subsequences': 9}, (9, 6), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 0.1,
       'n_subsequences': 9}, (9, 6), 10),
     ({'min_subsequence_size': 0.5, 'min_interval_size': 2,
       'n_subsequences': 9}, (9, 6), 10),
     ({'min_subsequence_size': 0.5, 'min_interval_size': 0.1,
       'n_subsequences': 9}, (9, 6), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 2,
       'n_subsequences': 4}, (4, 6), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 5,
       'n_subsequences': 6}, (6, 3), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 5,
       'n_subsequences': 0.3}, (6, 3), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 2,
       'n_subsequences': 'auto'}, (5, 6), 10),
     ({'min_subsequence_size': 10, 'min_interval_size': 5,
       'n_subsequences': 'auto'}, (2, 3), 10)]
)
def test_attributes_interval_feature_extractor(
    params, interval_indices_shape, min_subsequence_size
):
    """Test the attributes of a fitted instance of IntervalFeatureExtractor."""
    feature_extractor = IntervalFeatureExtractor(**params)
    feature_extractor.fit(X_arange, y)
    assert feature_extractor.interval_indices_.shape == interval_indices_shape
    assert feature_extractor.min_subsequence_size_ == min_subsequence_size


@pytest.mark.parametrize(
    'X, y, params',
    [(X_arange, y, {}),
     (X_arange, y, {'min_subsequence_size': 6, 'min_interval_size': 2}),
     (X_arange, y, {'min_subsequence_size': 0.8, 'min_interval_size': 2}),
     (X_arange, y, {'min_subsequence_size': 8, 'min_interval_size': 0.2}),
     (X_arange, y, {'min_subsequence_size': 0.8, 'min_interval_size': 0.1}),
     (X_arange, y, {'random_state': 42}),
     (X_arange[:3], y[:3], {}),
     (X_ones, y, {}),
     (X_ones, y, {'min_subsequence_size': 6, 'min_interval_size': 2}),
     (X_ones, y, {'min_subsequence_size': 0.8, 'min_interval_size': 2}),
     (X_ones, y, {'min_subsequence_size': 8, 'min_interval_size': 0.2}),
     (X_ones, y, {'min_subsequence_size': 0.8, 'min_interval_size': 0.1}),
     (X_ones, y, {'random_state': 42}),
     (X_ones[:3], y[:3], {})]
)
def test_methods_interval_feature_extractor(X, y, params):
    """Test the supported methods of IntervalFeatureExtractor."""
    n_samples = X.shape[0]
    feature_extractor = IntervalFeatureExtractor(**params).fit(X, y)
    n_subseries = feature_extractor.interval_indices_.shape[0]
    n_intervals = feature_extractor.interval_indices_.shape[1] - 1
    assert feature_extractor.transform(X).shape == (n_samples * n_subseries,
                                                    3 * n_intervals + 4)


@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'bins': '4'}, TypeError, "'bins' must be an integer or array-like."),

     ({'bins': None}, TypeError, "'bins' must be an integer or array-like."),

     ({'bins': [0, 1, -1]}, ValueError,
      "If 'bins' is array-like, the bin edges must increase monotonically."),

     ({'bins': (0, 1, -1, 0, 1)}, ValueError,
      "If 'bins' is array-like, the bin edges must increase monotonically."),

     ({'bins': np.array([0, -1, 2])}, ValueError,
      "If 'bins' is array-like, the bin edges must increase monotonically.")]
)
def test_parameter_check_tsbf(params, error, err_msg):
    """Test parameter validation."""
    clf = TSBF(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X_arange, y)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize(
    'params, error, err_msg',
    [({'n_estimators': 1}, ValueError,
      "At least one sample was never left out during the bootstrap. "
      "Increase the number of trees (n_estimators).")]
)
def test_no_oob_score(params, error, err_msg):
    clf = TSBF(**params)
    with pytest.raises(error, match=re.escape(err_msg)):
        clf.fit(X_arange, y)


@pytest.mark.parametrize(
    'X, y, params',
    [(X_arange, y, {}),
     (X_arange, y, {'oob_score': True}),
     (X_arange, y, {'n_estimators': 30, 'random_state': 42, 'bins': 5}),
     (X_arange[:3], y[:3], {}),
     (X_ones, y, {'bins': 3}),
     (X_ones, y, {'n_estimators': 15, 'random_state': 42, 'bins': 5}),
     (X_ones[:3], y[:3], {'bins': np.linspace(0, 1, 13)})]
)
def test_attributes_tsbf(X, y, params):
    """Test the attributes of a fitted instance of TSBF."""
    n_samples = X.shape[0]
    n_classes = np.unique(y).size
    n_estimators = params.get('n_estimators',
                              TSBF().get_params()['n_estimators'])
    bins = params.get('bins', TSBF().get_params()['bins'])
    n_bins = bins if isinstance(bins, (int, np.integer)) else len(bins) - 1
    n_features = (n_bins + 1) * n_classes
    clf = TSBF(**params)

    # Training phase
    clf.fit(X, y)
    assert isinstance(clf.estimator_, DecisionTreeClassifier)
    np.testing.assert_array_equal(clf.classes_, np.unique(y))
    assert len(clf.estimators_) == n_estimators
    assert clf.feature_importances_.shape == (n_features,)
    assert clf.interval_indices_.ndim == 2
    assert isinstance(clf.min_subsequence_size_, (int, np.integer))
    assert clf.n_features_in_ == n_features
    if params.get('oob_score', TSBF().get_params()['oob_score']):
        assert clf.oob_decision_function_.shape == (n_samples, n_classes)
        assert isinstance(clf.oob_score_, (float, np.floating))
    else:
        assert clf.oob_decision_function_ is None
        assert clf.oob_score_ is None


@pytest.mark.parametrize(
    'X, y, params',
    [(X_arange, y, {}),
     (X_arange, y, {'n_estimators': 30, 'random_state': 42}),
     (X_arange[:3], y[:3], {}),
     (X_ones, y, {}),
     (X_ones, y, {'bins': 3}),
     (X_ones, y, {'bins': np.linspace(0., 1., 4)}),
     (X_ones, y, {'n_estimators': 15, 'random_state': 42}),
     (X_ones[:3], y[:3], {})]
)
def test_methods_tsbf(X, y, params):
    """Test the supported methods of TSBF."""
    n_samples = X.shape[0]
    n_classes = np.unique(y).size
    n_estimators = params.get('n_estimators',
                              TSBF().get_params()['n_estimators'])
    clf = TSBF(**params).fit(X, y)
    assert clf.apply(X).shape == (n_samples, n_estimators)
    assert clf.decision_path(X)[0].shape[0] == n_samples
    assert clf.decision_path(X)[1].shape == (n_estimators + 1,)
    assert clf.predict(X).shape == (n_samples,)
    assert clf.predict_proba(X).shape == (n_samples, n_classes)
    assert isinstance(clf.score(X, y), (float, np.floating))
