"""Code for BOSS metric."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import sqrt
from sklearn.utils import check_array


def boss(x, y):
    """Return the BOSS distance between two arrays.

    x : array-like, shape = (n_timestamps,)
        First array.

    y : array-like, shape = (n_timestamps,)
        Second array.

    Returns
    -------
    dist : float
        The BOSS distance between both arrays.

    References
    ----------
    .. [1] P. Schäfer, "The BOSS is concerned with time series classification
           in the presence of noise". Data Mining and Knowledge Discovery,
           29(6), 1505-1530 (2015).

    Examples
    --------
    >>> from pyts.metrics import boss
    >>> x = [0, 5, 5, 3, 4, 5]
    >>> y = [3, 0, 0, 0, 8, 0]
    >>> boss(x, y)
    10.0
    >>> boss(y, x)
    5.0

    """
    x = check_array(x, ensure_2d=False, dtype='float64')
    y = check_array(y, ensure_2d=False, dtype='float64')
    if x.ndim != 1:
        raise ValueError("'x' must a one-dimensional array.")
    if y.ndim != 1:
        raise ValueError("'y' must a one-dimensional array.")
    if x.shape != y.shape:
        raise ValueError("'x' and 'y' must have the same shape.")

    non_zero_idx = ~np.isclose(x, np.zeros_like(x), rtol=1e-5, atol=1e-8)
    return sqrt(np.sum((x[non_zero_idx] - y[non_zero_idx]) ** 2))
