"""Code for Symbolic Fourier Approximation."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from .dft import DiscreteFourierTransform
from .mcb import MultipleCoefficientBinning


class SymbolicFourierApproximation(BaseEstimator, TransformerMixin):
    """Symbolic Fourier Approximation.

    Parameters
    ----------
    n_coefs : None, int or float (default = None)
        The number of Fourier coefficients to keep. If None, all the Fourier
        coeeficients are kept. If an integer, the ``n_coefs`` most significant
        Fourier coefficients are returned if ``anova=True``, otherwise the
        first ``n_coefs`` Fourier coefficients are returned. If a float, it
        represents a percentage of the size of each time series and must be
        between 0 and 1. The number of coefficients will be computed as
        ``ceil(n_coefs * (n_timestamps - 1))`` if ``drop_sum=True`` and
        ``ceil(n_coefs * n_timestamps)`` if ``drop_sum=False``.

    n_bins : int (default = 4)
        The number of bins to produce. The intervals for the bins are
        determined by the minimum and maximum of the input data. It must
        be between 2 and 26.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the time
        series) is dropped. If False, the real part of the first Fourier
        coefficient is kept.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    norm_mean : bool (default = False)
        If True, center the data before scaling. If ``norm_mean=True`` and
        ``anova=False``, the first Fourier coefficient will be dropped.

    norm_std : bool (default = False)
        If True, scale the data to unit variance.

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used if `n_bins` is lower than 27, otherwise the alphabet
        will be defined to [chr(i) for i in range(n_bins)]. If 'ordinal',
        integers are used.

    Attributes
    ----------
    bin_edges_ : array, shape = (n_bins - 1,) or (n_timestamps, n_bins - 1)
        Bin edges with shape = (n_bins - 1,) if ``strategy='normal'`` or
        (n_timestamps, n_bins - 1) otherwise.

    support_ : array, shape = (n_coefs,)
        Indices of the kept Fourier coefficients.

    References
    ----------
    .. [1] P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
           and Index for Similarity Search in High Dimensional Datasets",
           International Conference on Extending Database Technology,
           15, 516-527 (2012).

    Examples
    --------
    >>> from pyts.approximation import SymbolicFourierApproximation
    >>> from pyts.datasets import load_gunpoint
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = SymbolicFourierApproximation(n_coefs=4)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (50, 4)

    """

    def __init__(self, n_coefs=None, n_bins=4, strategy='quantile',
                 drop_sum=False, anova=False, norm_mean=False, norm_std=False,
                 alphabet=None):
        self.n_coefs = n_coefs
        self.drop_sum = drop_sum
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.n_bins = n_bins
        self.strategy = strategy
        self.alphabet = alphabet

    def fit(self, X, y=None):
        """Select Fourier coefficients and compute bin edges for each feature.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels for each sample. Only used if ``anova=True`` or
            ``strategy='entropy'.``

        """
        dft = DiscreteFourierTransform(
            n_coefs=self.n_coefs, drop_sum=self.drop_sum, anova=self.anova,
            norm_mean=self.norm_mean, norm_std=self.norm_std
        )
        mcb = MultipleCoefficientBinning(
            n_bins=self.n_bins, strategy=self.strategy, alphabet=self.alphabet
        )
        self._pipeline = Pipeline([('dft', dft), ('mcb', mcb)])
        self._pipeline.fit(X, y)
        self.support_ = self._pipeline.named_steps['dft'].support_
        self.bin_edges_ = self._pipeline.named_steps['mcb'].bin_edges_
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array, shape = (n_samples, n_coefs)
            Transformed data.

        """
        check_is_fitted(self, ['support_', 'bin_edges_'])
        return self._pipeline.transform(X)

    def fit_transform(self, X, y=None):
        """Fit then transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        y : None or array-like, shape = (n_samples,)
            Class labels for each sample. Only used if ``anova=True`` or
            ``strategy='entropy'.``

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_coefs)
            Transformed data.

        """
        dft = DiscreteFourierTransform(
            n_coefs=self.n_coefs, drop_sum=self.drop_sum, anova=self.anova,
            norm_mean=self.norm_mean, norm_std=self.norm_std
        )
        mcb = MultipleCoefficientBinning(
            n_bins=self.n_bins, strategy=self.strategy, alphabet=self.alphabet
        )
        self._pipeline = Pipeline([('dft', dft), ('mcb', mcb)])
        X_sfa = self._pipeline.fit_transform(X, y)
        self.support_ = self._pipeline.named_steps['dft'].support_
        self.bin_edges_ = self._pipeline.named_steps['mcb'].bin_edges_
        return X_sfa
