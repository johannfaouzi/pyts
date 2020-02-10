"""Code for Bag-of-patterns representation for time series."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.validation import check_array, check_is_fitted
from ..bag_of_words import BagOfWords


class BagOfPatterns(BaseEstimator, TransformerMixin):
    """Bag-of-patterns representation for time series.

    This algorithm uses a sliding window to extract subsequences from the
    time series and transforms each subsequence into a word using the
    Piecewise Aggregate Approximation and the Symbolic Aggregate approXimation
    algorithms. Thus it transforms each time series into a bag of words.
    Then it derives the frequencies of each word for each time series.

    Parameters
    ----------
    window_size : int or float (default = 0.5)
        Length of the sliding window. If float, it represents
        a percentage of the size of each time series and must be
        between 0 and 1.

    word_size : int or float (default = 0.5)
        Length of the words. If float, it represents
        a percentage of the length of the sliding window and must be
        between 0. and 1.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and
        ``min(window_size, 26)``.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'normal')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The step of
        sliding window will be computed as
        ``ceil(window_step * n_timestamps)``.

    norm_mean : bool (default = True)
        If True, center each subseries before scaling.

    norm_std : bool (default = True)
        If True, scale each subseries to unit variance.

    sparse : bool (default = True)
        Return a sparse matrix if True, else return an array.

    overlapping : bool (default = True)
        If True, time points may belong to two bins when decreasing the size
        of the subsequence with the Piecewise Aggregate Approximation
        algorithm. If False, each time point belong to one single bin, but
        the size of the bins may vary.

    alphabet : None or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] J. Lin, R. Khade and Y. Li, "Rotation-invariant similarity in time
           series using bag-of-patterns representation". Journal of Intelligent
           Information Systems, 39 (2), 287-315 (2012).

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.transformation import BagOfPatterns
    >>> X = np.arange(12).reshape(2, 6)
    >>> bop = BagOfPatterns(window_size=4, word_size=4, sparse=False)
    >>> bop.fit_transform(X)
    array(...)
    >>> bop.set_params(numerosity_reduction=False)
    BagOfPatterns(...)
    >>> bop.fit_transform(X)
    array(...)

    """

    def __init__(self, window_size=0.5, word_size=0.5, n_bins=4,
                 strategy='normal', numerosity_reduction=True, window_step=1,
                 norm_mean=True, norm_std=True, sparse=True, overlapping=True,
                 alphabet=None):
        self.window_size = window_size
        self.word_size = word_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.numerosity_reduction = numerosity_reduction
        self.window_step = window_step
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.sparse = sparse
        self.overlapping = overlapping
        self.alphabet = alphabet

    def fit(self, X, y=None):
        """Learn the dictionary.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Input data

        y
            Ignored

        Returns
        -------
        self : object

        """
        # Transform each time series into a bag of words
        bow = BagOfWords(
            window_size=self.window_size, word_size=self.word_size,
            n_bins=self.n_bins, strategy=self.strategy,
            numerosity_reduction=self.numerosity_reduction,
            window_step=self.window_step, norm_mean=self.norm_mean,
            norm_std=self.norm_std, overlapping=self.overlapping,
            alphabet=self.alphabet
        )
        X_bow = bow.transform(X)

        # Learn the vocabulary
        vectorizer = CountVectorizer()
        vectorizer.fit(X_bow)
        self.vocabulary_ = {value: key for key, value in
                            vectorizer.vocabulary_.items()}
        self._vectorizer = vectorizer
        return self

    def transform(self, X):
        """Derive word frequencies for each time series.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array, shape = (n_samples, n_words)
            Word frequencies.

        """
        X = check_array(X, dtype='float64')
        check_is_fitted(self, 'vocabulary_')

        # Transform each time series into a bag of words
        bow = BagOfWords(
            window_size=self.window_size, word_size=self.word_size,
            n_bins=self.n_bins, strategy=self.strategy,
            numerosity_reduction=self.numerosity_reduction,
            window_step=self.window_step, norm_mean=self.norm_mean,
            norm_std=self.norm_std, overlapping=self.overlapping,
            alphabet=self.alphabet
        )
        X_bow = bow.transform(X)

        # Derive frequencies for each word in the vocabulary
        X_bop = self._vectorizer.transform(X_bow)
        if not self.sparse:
            return X_bop.A
        return csr_matrix(X_bop)

    def fit_transform(self, X, y=None):
        """Derive word frequencies for each time series.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        y
            Ignored

        Returns
        -------
        X_new : array, shape = (n_samples, n_words)
            Word frequencies.

        """
        # Transform each time series into a bag of words
        bow = BagOfWords(
            window_size=self.window_size, word_size=self.word_size,
            n_bins=self.n_bins, strategy=self.strategy,
            numerosity_reduction=self.numerosity_reduction,
            window_step=self.window_step, norm_mean=self.norm_mean,
            norm_std=self.norm_std, overlapping=self.overlapping,
            alphabet=self.alphabet
        )
        X_bow = bow.transform(X)

        # Derive frequencies of each word
        vectorizer = CountVectorizer()
        X_bop = vectorizer.fit_transform(X_bow)
        self.vocabulary_ = {value: key for key, value in
                            vectorizer.vocabulary_.items()}
        self._vectorizer = vectorizer
        if not self.sparse:
            return X_bop.A
        return csr_matrix(X_bop)
