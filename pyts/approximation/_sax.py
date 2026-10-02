"""Code for Symbolic Aggregate approXimation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from typing import Any, Self

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from pyts._base import UnivariateTransformerMixin
from pyts.preprocessing import KBinsDiscretizer


class SymbolicAggregateApproximation(BaseEstimator, UnivariateTransformerMixin):
    """Symbolic Aggregate approXimation.

    Parameters
    ----------
    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and
        ``min(n_timestamps, 26)``.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    raise_warning : bool (default = True)
        If True, a warning is raised when the number of bins is smaller for
        at least one sample. In this case, you should consider decreasing the
        number of bins or removing these samples.

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used. If 'ordinal', integers are used.

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
           novel symbolic representation of time series". Data Mining and
           Knowledge Discovery, 15(2), 107-144 (2007).

    Examples
    --------
    >>> from pyts.approximation import SymbolicAggregateApproximation
    >>> X = [[0, 4, 2, 1, 7, 6, 3, 5],
    ...      [2, 5, 4, 5, 3, 4, 2, 3]]
    >>> transformer = SymbolicAggregateApproximation()
    >>> print(transformer.transform(X))
    [['a' 'c' 'b' 'a' 'd' 'd' 'b' 'c']
     ['a' 'd' 'c' 'd' 'b' 'c' 'a' 'b']]

    """

    def __init__(
        self,
        n_bins: int = 4,
        strategy: str = 'quantile',
        raise_warning: bool = True,
        alphabet: str | npt.ArrayLike | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.strategy = strategy
        self.raise_warning = raise_warning
        self.alphabet = alphabet

    def fit(
        self, X: npt.ArrayLike | None = None, y: npt.ArrayLike | None = None
    ) -> Self:
        """Pass.

        Parameters
        ----------
        X
            Ignored
        y
            Ignored

        """
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Bin the data with the given alphabet.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array, shape = (n_samples, n_timestamps)
            Binned data.

        """
        X = check_array(X, dtype=np.float64)
        n_timestamps = X.shape[1]
        alphabet = self._check_params(n_timestamps)
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy=self.strategy,
            raise_warning=self.raise_warning,
        )
        indices = discretizer.fit_transform(X)
        if isinstance(alphabet, str):
            return indices
        else:
            return alphabet[indices]

    def _check_params(self, n_timestamps: int) -> npt.NDArray[Any] | str:
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins <= 26:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 and lower than "
                f"or equal to 26 (got {self.n_bins})."
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError(
                "'strategy' must be either 'uniform', 'quantile' "
                f"or 'normal' (got {self.strategy})"
            )
        if not (
            (self.alphabet is None)
            or (self.alphabet == 'ordinal')
            or (isinstance(self.alphabet, (list, tuple, np.ndarray)))
        ):
            raise TypeError(
                "'alphabet' must be None, 'ordinal' or array-like "
                f"with shape (n_bins,) (got {self.alphabet})"
            )
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])
        elif self.alphabet == 'ordinal':
            alphabet = 'ordinal'
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.n_bins,):
                raise ValueError(
                    "If 'alphabet' is array-like, its shape must be equal to (n_bins,)."
                )
        return alphabet
