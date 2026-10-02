"""The :mod:`pyts.multivariate.utils` module includes utility tools."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import numpy.typing as npt
from sklearn.utils import check_array


def check_3d_array(X: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Check that the input is a three-dimensional array.

    Parameters
    ----------
    X : array-like
        Input data.

    Returns
    -------
    X_new : array
        Input data as an array.

    """
    X = check_array(X, ensure_2d=False, allow_nd=True)
    if X.ndim != 3:
        raise ValueError(f"X must be 3-dimensional (got {X.ndim}).")
    return X
