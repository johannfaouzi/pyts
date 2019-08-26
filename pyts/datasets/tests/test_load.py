"""Testing for datatset loading."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
from ..load import (load_basic_motions, load_coffee, load_gunpoint,
                    load_pig_central_venous_pressure)


@pytest.mark.parametrize(
    'function, data_train_shape, data_test_shape, target_train_shape, '
    'target_test_shape, n_classes',
    [(load_basic_motions, (40, 6, 100), (40, 6, 100), (40,), (40,), 4),
     (load_coffee, (28, 286), (28, 286), (28,), (28,), 2),
     (load_gunpoint, (50, 150), (150, 150), (50,), (150,), 2),
     (load_pig_central_venous_pressure, (104, 2000), (208, 2000), (104,),
      (208,), 52)]
)
@pytest.mark.parametrize('return_X_y', [False, True])
def test_load_functions(function, data_train_shape, data_test_shape,
                        target_train_shape, target_test_shape, n_classes,
                        return_X_y):
    """Test the loading functions."""
    res = function(return_X_y=return_X_y)
    if return_X_y:
        data_train, data_test, target_train, target_test = res
    else:
        data_train = res.data_train
        data_test = res.data_test
        target_train = res.target_train
        target_test = res.target_test
        assert isinstance(res.url, str)
        assert isinstance(res.DESCR, str)
    for data in (data_train, data_test, target_train, target_test):
        assert isinstance(data, np.ndarray)

    assert data_train.shape == data_train_shape
    assert data_test.shape == data_test_shape
    assert target_train.shape == target_train_shape
    assert target_test.shape == target_test_shape
    assert np.unique(target_train).size == n_classes
    assert np.unique(target_test).size == n_classes
