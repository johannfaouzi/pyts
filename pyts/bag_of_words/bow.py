"""Code for Bag-of-words representation for time series."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array
from ..approximation import (
    PiecewiseAggregateApproximation, SymbolicAggregateApproximation)
from ..base import UnivariateTransformerMixin
from ..preprocessing import KBinsDiscretizer, StandardScaler
from ..preprocessing.discretizer import _digitize
from ..utils.utils import _windowed_view


class WordExtractor(BaseEstimator, UnivariateTransformerMixin):
    r"""Transform discretized time series into sequences of words.

    Parameters
    ----------
    window_size : int or float (default = 0.1)
        Size of the sliding window (i.e. the size of each word). If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    Examples
    --------
    >>> from pyts.bag_of_words import WordExtractor
    >>> X = [['a', 'a', 'b', 'a', 'b', 'b', 'b', 'b', 'a'],
    ...      ['a', 'b', 'c', 'c', 'c', 'c', 'a', 'a', 'c']]
    >>> word = WordExtractor(window_size=2)
    >>> print(word.transform(X))
    ['aa ab ba ab bb ba' 'ab bc cc ca aa ac']
    >>> word = WordExtractor(window_size=2, numerosity_reduction=False)
    >>> print(word.transform(X))
    ['aa ab ba ab bb bb bb ba' 'ab bc cc cc cc ca aa ac']
    """

    def __init__(self, window_size=0.1, window_step=1,
                 numerosity_reduction=True):
        self.window_size = window_size
        self.window_step = window_step
        self.numerosity_reduction = numerosity_reduction

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform time series into sequences of words.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array, shape = (n_samples,)
            Transformed data. Each row is a string consisting of words
            separated by a whitespace.

        """
        X = check_array(X, dtype=None)
        n_samples, n_timestamps = X.shape
        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_window = _windowed_view(X, n_samples, n_timestamps,
                                  window_size, window_step)
        X_word = np.asarray([[''.join(X_window[i, j])
                              for j in range(n_windows)]
                             for i in range(n_samples)])

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
        return X_bow

    def _check_params(self, n_timestamps):
        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 1 <= self.window_size <= n_timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_size)
                )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.window_size)
                )
            window_size = ceil(self.window_size * n_timestamps)
        if not isinstance(self.window_step,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_step' must be an integer or a float.")
        if isinstance(self.window_step, (int, np.integer)):
            if not 1 <= self.window_step <= n_timestamps:
                raise ValueError(
                    "If 'window_step' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_step)
                )
            window_step = self.window_step
        else:
            if not 0 < self.window_step <= 1:
                raise ValueError(
                    "If 'window_step' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.window_step)
                )
            window_step = ceil(self.window_step * n_timestamps)
        return window_size, window_step


class BagOfWords(BaseEstimator, TransformerMixin):
    """Bag-of-words representation for time series.

    This algorithm uses a sliding window to extract subsequences from the
    time series and transforms each subsequence into a word using the
    Piecewise Aggregate Approximation and the Symbolic Aggregate approXimation
    algorithms. Thus it transforms each time series into a bag of words.

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
        ``min(n_timestamps, 26)``.

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
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    threshold_std : int, float or None (default = 0.01)
        Threshold used to determine whether a subsequence is standardized.
        Subsequences whose standard deviations are lower than this threshold
        are not standardized. If None, all the subsequences are standardized.

    norm_mean : bool (default = True)
        If True, center each subseries before scaling.

    norm_std : bool (default = True)
        If True, scale each subseries to unit variance.

    overlapping : bool (default = True)
        If True, time points may belong to two bins when decreasing the size
        of the subsequence with the Piecewise Aggregate Approximation
        algorithm. If False, each time point belong to one single bin, but
        the size of the bins may vary.

    raise_warning : bool (default = False)
        If True, a warning is raised when the number of bins is smaller for
        at least one subsequence. In this case, you should consider decreasing
        the number of bins, using another strategy to compute the bins or
        removing the corresponding time series.

    alphabet : None or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    References
    ----------
    .. [1] J. Lin, R. Khade and Y. Li, "Rotation-invariant similarity in time
           series using bag-of-patterns representation". Journal of Intelligent
           Information Systems, 39 (2), 287-315 (2012).

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.bag_of_words import BagOfWords
    >>> X = np.arange(12).reshape(2, 6)
    >>> bow = BagOfWords(window_size=4, word_size=4)
    >>> bow.transform(X)
    array(['abcd', 'abcd'], dtype='<U4')
    >>> bow.set_params(numerosity_reduction=False)
    BagOfWords(...)
    >>> bow.transform(X)
    array(['abcd abcd abcd', 'abcd abcd abcd'], dtype='<U14')

    """

    def __init__(self, window_size=0.5, word_size=0.5, n_bins=4,
                 strategy='normal', numerosity_reduction=True, window_step=1,
                 threshold_std=0.01, norm_mean=True, norm_std=True,
                 overlapping=True, raise_warning=False, alphabet=None):
        self.window_size = window_size
        self.word_size = word_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.numerosity_reduction = numerosity_reduction
        self.window_step = window_step
        self.threshold_std = threshold_std
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.overlapping = overlapping
        self.raise_warning = raise_warning
        self.alphabet = alphabet

    def fit(self, X, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform each time series into a bag of words.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array, shape = (n_samples,)
            Bags of words.

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape
        window_size, word_size, window_step, alphabet = self._check_params(
            n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        # Standardize time series if quantile from standard normal distribution
        if self.strategy == 'normal':
            X_scaled = StandardScaler().transform(X)
        else:
            X_scaled = X

        # Extract subsequences using a sliding window
        X_window = _windowed_view(
            X_scaled, n_samples, n_timestamps, window_size, window_step
        ).reshape(n_samples * n_windows, window_size)

        if self.threshold_std is not None:

            # Identify subsequences whose standard deviation below threshold
            idx = np.std(X_window, axis=1) < self.threshold_std

            if np.any(idx):
                # Subsequences with standard deviations below threshold
                X_paa = PiecewiseAggregateApproximation(
                    window_size=None, output_size=word_size,
                    overlapping=self.overlapping
                ).transform(X_window[idx])

                # Compute the bin edges
                discretizer = KBinsDiscretizer(
                    n_bins=self.n_bins, strategy=self.strategy,
                    raise_warning=self.raise_warning
                )
                bin_edges = discretizer._compute_bins(
                    X_scaled, n_samples, self.n_bins, self.strategy)

                # Tile the bin edges for subsequences from the same time series
                if self.strategy != 'normal':
                    count = np.bincount(
                        np.floor_divide(np.nonzero(idx)[0], n_windows)
                    )
                    bin_edges = np.vstack([
                        np.tile(bin_edges[i], (count[i], 1))
                        for i in range(count.size) if count[i] != 0
                    ])

                X_sax_below_thresh = alphabet[_digitize(X_paa, bin_edges)]

        # Subsequences with standard deviations above threshold
        if (self.threshold_std is None) or (not np.all(idx)):
            pipeline = make_pipeline(
                StandardScaler(
                    with_mean=self.norm_mean, with_std=self.norm_std
                ),
                PiecewiseAggregateApproximation(
                    window_size=None, output_size=word_size,
                    overlapping=self.overlapping
                ),
                SymbolicAggregateApproximation(
                    n_bins=self.n_bins, strategy=self.strategy,
                    alphabet=self.alphabet, raise_warning=self.raise_warning
                )
            )
            if self.threshold_std is None:
                X_sax_above_thresh = pipeline.fit_transform(X_window)
            else:
                X_sax_above_thresh = pipeline.fit_transform(X_window[~idx])

        # Concatenate SAX words
        if self.threshold_std is not None:
            if np.any(idx):
                if not np.all(idx):
                    X_sax = np.empty((n_samples * n_windows, word_size),
                                     dtype='<U1')
                    X_sax[idx] = X_sax_below_thresh
                    X_sax[~idx] = X_sax_above_thresh
                else:
                    X_sax = X_sax_below_thresh
            else:
                X_sax = X_sax_above_thresh
        else:
            X_sax = X_sax_above_thresh
        X_sax = X_sax.reshape(n_samples, n_windows, word_size)

        # Join letters to make words
        X_word = np.asarray([[''.join(X_sax[i, j])
                              for j in range(n_windows)]
                             for i in range(n_samples)])

        # Apply numerosity reduction
        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        return X_bow

    def _check_params(self, n_timestamps):
        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 1 <= self.window_size <= n_timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_size)
                )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.window_size)
                )
            window_size = ceil(self.window_size * n_timestamps)

        if not isinstance(self.word_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'word_size' must be an integer or a float.")
        if isinstance(self.word_size, (int, np.integer)):
            if not 1 <= self.word_size <= window_size:
                raise ValueError(
                    "If 'word_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "window_size (got {0}).".format(self.word_size)
                )
            word_size = self.word_size
        else:
            if not 0 < self.word_size <= 1:
                raise ValueError(
                    "If 'word_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    "(got {0}).".format(self.word_size)
                )
            word_size = ceil(self.word_size * window_size)

        if not isinstance(self.n_bins, (int, np.integer)):
            raise TypeError("'n_bins' must be an integer.")
        if not 2 <= self.n_bins <= 26:
            raise ValueError(
                "'n_bins' must be greater than or equal to 2 and lower than "
                "or equal to 26 (got {0})."
                .format(self.n_bins)
            )

        if self.strategy not in ['uniform', 'quantile', 'normal']:
            raise ValueError("'strategy' must be either 'uniform', 'quantile' "
                             "or 'normal' (got {0})".format(self.strategy))

        if not isinstance(self.window_step,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_step' must be an integer or a float.")
        if isinstance(self.window_step, (int, np.integer)):
            if not 1 <= self.window_step <= n_timestamps:
                raise ValueError(
                    "If 'window_step' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.window_step)
                )
            window_step = self.window_step
        else:
            if not 0 < self.window_step <= 1:
                raise ValueError(
                    "If 'window_step' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.window_step)
                )
            window_step = ceil(self.window_step * n_timestamps)

        threshold_std_none = self.threshold_std is None
        threshold_std_int_float = isinstance(
            self.threshold_std, (int, np.integer, float, np.floating))
        if not (threshold_std_none or threshold_std_int_float):
            raise TypeError(
                "'threshold_std' must be an integer, a float or None."
            )
        if threshold_std_int_float and (not self.threshold_std >= 0.):
            raise ValueError("If 'threshold_std' is an integer or a float, it "
                             "must be non-negative (got {0})."
                             .format(self.threshold_std))

        if not ((self.alphabet is None)
                or (isinstance(self.alphabet, (list, tuple, np.ndarray)))):
            raise TypeError("'alphabet' must be None or array-like "
                            "with shape (n_bins,) (got {0})."
                            .format(self.alphabet))
        if self.alphabet is None:
            alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])
        else:
            alphabet = check_array(self.alphabet, ensure_2d=False, dtype=None)
            if alphabet.shape != (self.n_bins,):
                raise ValueError("If 'alphabet' is array-like, its shape "
                                 "must be equal to (n_bins,).")

        return window_size, word_size, window_step, alphabet
