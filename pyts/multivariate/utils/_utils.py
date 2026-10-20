"""The :mod:`pyts.multivariate.utils` module includes utility tools."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, cast

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
    X = cast(npt.NDArray[np.float64], check_array(X, ensure_2d=False, allow_nd=True))
    if X.ndim != 3:
        raise ValueError(f"X must be 3-dimensional (got {X.ndim}).")
    return X


def _check_n_features(X: npt.NDArray[Any], n_features_in: int) -> None:
    """Check that X has as many features as the data seen in ``fit``.

    Each feature is handled by its own fitted estimator, so a different
    number of features would otherwise either be silently ignored (extra
    features) or fail with an opaque ``IndexError`` (missing features).
    """
    n_features = X.shape[1]
    if n_features != n_features_in:
        raise ValueError(
            f"X has {n_features} features, but this estimator was fitted "
            f"with {n_features_in} features."
        )
