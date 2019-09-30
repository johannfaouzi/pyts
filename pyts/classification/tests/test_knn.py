"""Testing for k-nearest-neighbors."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from pyts.classification import KNeighborsClassifier


X = np.arange(40).reshape(8, 5)
y = np.array([0, 0, 0, 1, 1, 0, 1, 1])
y_proba = np.vstack([1 - y, y]).T
X_test = X + 0.1


@pytest.mark.parametrize(
    'params',
    [({'metric': 'euclidean'}),
     ({'metric': 'manhattan'}),
     ({'metric': 'dtw'}),
     ({'metric': 'dtw_classic'}),
     ({'metric': 'dtw_sakoechiba'}),
     ({'metric': 'dtw_sakoechiba', 'metric_params': {}}),
     ({'metric': 'dtw_sakoechiba', 'metric_params': {'window_size': 0.5}}),
     ({'metric': 'dtw_itakura'}),
     ({'metric': 'dtw_itakura', 'metric_params': {}}),
     ({'metric': 'dtw_itakura', 'metric_params': {'max_slope': 3.}}),
     ({'metric': 'dtw_multiscale'}),
     ({'metric': 'dtw_fast'}),
     ({'metric': 'boss'})]
)
def test_actual_results(params):
    """Test that the actual results are the expected ones."""
    knn = KNeighborsClassifier(n_neighbors=1, **params)
    proba_actual = knn.fit(X, y).predict_proba(X_test)
    pred_actual = knn.predict(X_test)
    np.testing.assert_array_equal(proba_actual, y_proba)
    np.testing.assert_array_equal(pred_actual, y)
