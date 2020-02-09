"""WEASEL+MUSE algorithm."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np

from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from ...transformation import WEASEL
from ..utils import check_3d_array


class WEASELMUSE(BaseEstimator, TransformerMixin):
    r"""WEASEL+MUSE algorithm.

    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    window_sizes : array-like (default = [0.1, 0.3, 0.5, 0.7, 0.9])
        Size of the sliding windows. All the elements must be either integers
        or floats. In the latter case, each element represents the percentage
        of the size of each time series and must be between 0 and 1; the size
        of the sliding windows will be computed as
        ``np.ceil(window_sizes * n_timestamps)``.

    window_steps : None or array-like (default = None)
        Step of the sliding windows. If None, each ``window_step`` is equal to
        ``window_size`` so that the windows are non-overlapping. Otherwise, all
        the elements must be either integers or floats. In the latter case,
        each element represents the percentage of the size of each time series
        and must be between 0 and 1; the step of the sliding windows will be
        computed as ``np.ceil(window_steps * n_timestamps)``.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = True)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = True)
        If True, center each subseries before scaling.

    norm_std : bool (default = True)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    chi2_threshold : int or float (default = 2)
        The threshold used to perform feature selection. Only the words with
        a chi2 statistic above this threshold will be kept.

    sparse : bool (default = True)
        Return a sparse matrix if True, else return an array.

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of features indices to terms. Each value is a string with
        4 values separated by a whitespace:

            - 'o' or 'd': whether the word is extracted from the original \
                time series ('orig') or from the derivates ('diff')
            - int : feature index
            - int : window size
            - str : word


    References
    ----------
    .. [1] P. Schäfer, and U. Leser, "Multivariate Time Series Classification
           with WEASEL+MUSE". Proceedings of ACM Conference, (2017).

    Examples
    --------
    >>> from pyts.datasets import load_basic_motions
    >>> from pyts.multivariate.transformation import WEASELMUSE
    >>> X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)
    >>> transformer = WEASELMUSE()
    >>> X_new = transformer.fit_transform(X_train, y_train)
    >>> X_new.shape
    (40, 9086)

    """

    def __init__(self, word_size=4, n_bins=4,
                 window_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], window_steps=None,
                 anova=False, drop_sum=True, norm_mean=True, norm_std=True,
                 strategy='quantile', chi2_threshold=2, sparse=True,
                 alphabet=None):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_sizes = window_sizes
        self.window_steps = window_steps
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.chi2_threshold = chi2_threshold
        self.sparse = sparse
        self.alphabet = alphabet

    def fit(self, X, y):
        """Fit the model according to the given training data.

        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : array-like, shape = (n_samples,)
            Class labels.

        Returns
        -------
        self : object

        """
        X = check_3d_array(X)
        _, n_features, n_timestamps = X.shape

        X_diff = np.abs(np.diff(X))

        estimator = WEASEL(
            word_size=self.word_size, n_bins=self.n_bins,
            window_sizes=self.window_sizes, window_steps=self.window_steps,
            anova=self.anova, drop_sum=self.drop_sum, norm_mean=self.norm_mean,
            norm_std=self.norm_std, strategy=self.strategy,
            chi2_threshold=self.chi2_threshold, sparse=self.sparse,
            alphabet=self.alphabet
        )
        self._estimators = [clone(estimator) for _ in range(n_features)]
        self._estimators_diff = [clone(estimator) for _ in range(n_features)]

        self.vocabulary_ = {}

        for i, transformer in enumerate(self._estimators):
            transformer.fit(X[:, i, :], y)
            self._update_vocabulary(str(i + 1), transformer, original=True)

        for i, transformer in enumerate(self._estimators_diff):
            transformer.fit(X_diff[:, i, :], y)
            self._update_vocabulary(str(i + 1), transformer, original=False)

        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        Returns
        -------
        X_new : sparse matrix, shape = (n_samples, n_features_new)
            Document-term matrix with relevant learned features only.

        """
        check_is_fitted(self, 'vocabulary_')
        X = check_3d_array(X)
        n_samples, _, _ = X.shape
        X_diff = np.abs(np.diff(X))

        X_new = []
        for i, transformer in enumerate(self._estimators):
            X_new.append(transformer.transform(X[:, i, :]))
        for i, transformer in enumerate(self._estimators_diff):
            X_new.append(transformer.transform(X_diff[:, i, :]))

        if self.sparse:
            return csr_matrix(hstack(X_new))
        return np.hstack(X_new)

    def fit_transform(self, X, y):
        """Fit the data then transform it.

        X : array-like, shape = (n_samples, n_features, n_timestamps)
            Multivariate time series.

        y : array-like, shape = (n_samples,)
            Class labels.

        Returns
        -------
        X_new : array, shape = (n_samples, n_features_new)
            Document-term matrix with relevant features only.

        """
        X = check_3d_array(X)
        n_samples, n_features, n_timestamps = X.shape

        X_diff = np.abs(np.diff(X))

        estimator = WEASEL(
            word_size=self.word_size, n_bins=self.n_bins,
            window_sizes=self.window_sizes, window_steps=self.window_steps,
            anova=self.anova, drop_sum=self.drop_sum, norm_mean=self.norm_mean,
            norm_std=self.norm_std, strategy=self.strategy,
            chi2_threshold=self.chi2_threshold, sparse=self.sparse,
            alphabet=self.alphabet
        )
        self._estimators = [clone(estimator) for _ in range(n_features)]
        self._estimators_diff = [clone(estimator) for _ in range(n_features)]

        self.vocabulary_ = {}

        X_new = []
        for i, transformer in enumerate(self._estimators):
            X_new.append(transformer.fit_transform(X[:, i, :], y))
            self._update_vocabulary(str(i + 1), transformer, original=True)
        for i, transformer in enumerate(self._estimators_diff):
            X_new.append(transformer.fit_transform(X_diff[:, i, :], y))
            self._update_vocabulary(str(i + 1), transformer, original=False)

        if self.sparse:
            return csr_matrix(hstack(X_new))
        return np.hstack(X_new)

    def _update_vocabulary(self, feature_idx, estimator, original):
        """Update the vocabulary."""
        old_length = len(self.vocabulary_)
        if original:
            for (key, value) in estimator.vocabulary_.items():
                self.vocabulary_[old_length + key] = (
                    "o " + feature_idx + " " + value
                )
        else:
            for (key, value) in estimator.vocabulary_.items():
                self.vocabulary_[old_length + key] = (
                    "d " + feature_idx + " " + value
                )
