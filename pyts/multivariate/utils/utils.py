"""The :mod:`pyts.multivariate.utils` module includes utility tools."""

from sklearn.utils import check_array


def check_3d_array(X):
    """Check that the input is a three-dimensional arrayself.

    Parameters
    ----------
    X : array-like
        Input data

    """
    X = check_array(X, ensure_2d=False, allow_nd=True)
    if X.ndim != 3:
        raise ValueError("X must be 3-dimensional (got {0}).".format(X.ndim))
    return X
