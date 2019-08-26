"""Testing for utility tools."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import pytest
import re

from pyts.multivariate.utils import check_3d_array

n_samples, n_features, n_timestamps = 40, 3, 30
rng = np.random.RandomState(42)
X = rng.randn(n_samples, n_features, n_timestamps)


@pytest.mark.parametrize(
    'X, err_msg',
    [(X[0, 0], "X must be 3-dimensional (got 1)."),
     (X[0], "X must be 3-dimensional (got 2).")]
)
def test_3d_input(X, err_msg):
    """Test input data validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        check_3d_array(X)
