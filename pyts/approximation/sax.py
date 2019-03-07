"""Code for Symbolic Aggregate approXimation."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
from ..preprocessing import KBinsDiscretizer


class SymbolicAggregateApproximation(BaseEstimator, TransformerMixin):
    """Symbolic Aggregate approXimation.

    Parameters
    ----------
    n_bins : int (default = 4)
        The number of bins to produce. The intervals for the bins are
        determined by the minimum and maximum of the input data. It must
        be greater than or equal to 2.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    alphabet : None or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used if `n_bins` is lower than 27, otherwise the alphabet
        will be defined to [chr(i) for i in range(n_bins)]. If 'ordinal',
        integers are used.

    References
    ----------
    .. [1] J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
           novel symbolic representation of time series". Data Mining and
           Knowledge Discovery, 15(2), 107-144 (2007).

    """

    def __init__(self, n_bins=4, strategy='quantile', alphabet=None):
        self.n_bins = n_bins
        self.strategy = strategy
        self.alphabet = alphabet

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored
        y
            Ignored

        """
        return self

    def transform(self, X):
        """Bin the data with the given alphabet.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        y
            Ignored

        Returns
        -------
        X_new : array, shape = (n_samples, n_timestamps)
            Binned data.

        """
        X = check_array(X, dtype='float64')
        n_timestamps = X.shape[1]
        alphabet = self._check_params(n_timestamps)
        discretizer = KBinsDiscretizer(
            n_bins=self.n_bins, strategy=self.strategy)
        indices = discretizer.fit_transform(X)
        if isinstance(alphabet, str):
            return indices
        else:
            return alphabet[indices]

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins <= n_timestamps:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 and lower than "
                "or equal to n_timestamps (got {0}).".format(self.n_bins)
            )
        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strategy))
        if not ((self.alphabet is None)
                or (self.alphabet == 'ordinal')
                or (isinstance(self.alphabet, (list, tuple, np.ndarray)))):
            raise TypeError("'alphabet' must be None, 'ordinal' or array-like "
                            "with shape (n_bins,) (got {0})"
                            .format(self.alphabet))
        if self.alphabet is None:
            if self.n_bins < 27:
                alphabet = np.array(
                    [chr(i) for i in range(97, 97 + self.n_bins)])
            else:
                try:
                    alphabet = np.asarray([chr(i) for i in range(self.n_bins)])
                except:
                    raise ValueError("'n_bins' is unexpectedly high. You "
                                     "should try with a smaller value.")
        elif self.alphabet == 'ordinal':
            alphabet = 'ordinal'
        else:
            alphabet = np.asarray(self.alphabet)
            if alphabet.shape != (self.n_bins, ):
                raise ValueError("If 'alphabet' is array-like, its shape "
                                 "must be equal to (n_bins,).")
        return alphabet
