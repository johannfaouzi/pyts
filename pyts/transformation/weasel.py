"""Code for Word ExtrAction for time SEries cLassification."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, hstack
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from ..approximation import SymbolicFourierApproximation
from ..utils.utils import _windowed_view


class WEASEL(BaseEstimator, TransformerMixin):
    """Word ExtrAction for time SEries cLassification.

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

    anova : bool (default = True)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = True)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = True)
        If True, center each subseries before scaling.

    norm_std : bool (default = True)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'entropy')
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
        A mapping of features indices to terms.

    References
    ----------
    .. [1] P. Schäfer, and U. Leser, "Fast and Accurate Time Series
           Classification with WEASEL". Conference on Information and Knowledge
           Management, 637-646 (2017).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.transformation import WEASEL
    >>> X_train, _, y_train, _ = load_gunpoint(return_X_y=True)
    >>> weasel = WEASEL(sparse=False)
    >>> weasel.fit(X_train, y_train)
    WEASEL(...)
    >>> weasel.transform(X_train)
    array(...)

    """

    def __init__(self, word_size=4, n_bins=4,
                 window_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], window_steps=None,
                 anova=True, drop_sum=True, norm_mean=True, norm_std=True,
                 strategy='entropy', chi2_threshold=2, sparse=True,
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

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(X, y, dtype='float64')
        check_classification_targets(y)
        n_samples, n_timestamps = X.shape
        window_sizes, window_steps = self._check_params(n_timestamps)
        self._window_sizes = window_sizes
        self._window_steps = window_steps

        self._sfa_list = []
        self._vectorizer_list = []
        self._relevant_features_list = []
        self.vocabulary_ = {}

        for (window_size, window_step) in zip(window_sizes, window_steps):
            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = _windowed_view(
                X, n_samples, n_timestamps, window_size, window_step
            )
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

            sfa = SymbolicFourierApproximation(
                n_coefs=self.word_size, drop_sum=self.drop_sum,
                anova=self.anova, norm_mean=self.norm_mean,
                norm_std=self.norm_std, n_bins=self.n_bins,
                strategy=self.strategy, alphabet=self.alphabet
            )
            y_repeated = np.repeat(y, n_windows)
            X_sfa = sfa.fit_transform(X_windowed, y_repeated)

            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            X_word = X_word.reshape(n_samples, n_windows)

            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            X_counts = vectorizer.fit_transform(X_bow)
            chi2_statistics, _ = chi2(X_counts, y)
            relevant_features = np.where(
                chi2_statistics > self.chi2_threshold)[0]

            old_length_vocab = len(self.vocabulary_)
            vocabulary = {value: key
                          for (key, value) in vectorizer.vocabulary_.items()}
            for i, idx in enumerate(relevant_features):
                self.vocabulary_[i + old_length_vocab] = \
                    str(window_size) + " " + vocabulary[idx]

            self._relevant_features_list.append(relevant_features)
            self._sfa_list.append(sfa)
            self._vectorizer_list.append(vectorizer)

        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X_new : sparse matrix, shape = (n_samples, n_features)
            Document-term matrix with relevant features only.

        """
        check_is_fitted(self, ['_relevant_features_list', '_sfa_list',
                               '_vectorizer_list', 'vocabulary_'])

        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape

        X_features = coo_matrix((n_samples, 0), dtype=np.int64)

        for (window_size, window_step, sfa,
             vectorizer, relevant_features) in zip(
                 self._window_sizes, self._window_steps, self._sfa_list,
                 self._vectorizer_list, self._relevant_features_list):

            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = _windowed_view(
                X, n_samples, n_timestamps, window_size, window_step
            )
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)
            X_sfa = sfa.transform(X_windowed)

            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            X_word = X_word.reshape(n_samples, n_windows)
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
            X_counts = vectorizer.transform(X_bow)[:, relevant_features]
            X_features = hstack([X_features, X_counts])

        if not self.sparse:
            return X_features.A
        return csr_matrix(X_features)

    def fit_transform(self, X, y):
        """Fit the data then transform it.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Train samples.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        X_new : array, shape (n_samples, n_words)
            Document-term matrix.

        """
        X, y = check_X_y(X, y, dtype='float64')
        check_classification_targets(y)
        n_samples, n_timestamps = X.shape
        window_sizes, window_steps = self._check_params(n_timestamps)
        self._window_sizes = window_sizes
        self._window_steps = window_steps

        self._sfa_list = []
        self._vectorizer_list = []
        self._relevant_features_list = []
        self.vocabulary_ = {}

        X_features = coo_matrix((n_samples, 0), dtype=np.int64)

        for (window_size, window_step) in zip(window_sizes, window_steps):
            n_windows = ((n_timestamps - window_size + window_step)
                         // window_step)
            X_windowed = _windowed_view(
                X, n_samples, n_timestamps, window_size, window_step
            )
            X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

            sfa = SymbolicFourierApproximation(
                n_coefs=self.word_size, drop_sum=self.drop_sum,
                anova=self.anova, norm_mean=self.norm_mean,
                norm_std=self.norm_std, n_bins=self.n_bins,
                strategy=self.strategy, alphabet=self.alphabet
            )
            y_repeated = np.repeat(y, n_windows)
            X_sfa = sfa.fit_transform(X_windowed, y_repeated)

            X_word = np.asarray([''.join(X_sfa[i])
                                 for i in range(n_samples * n_windows)])
            X_word = X_word.reshape(n_samples, n_windows)

            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])
            vectorizer = CountVectorizer(ngram_range=(1, 2))
            X_counts = vectorizer.fit_transform(X_bow)
            chi2_statistics, _ = chi2(X_counts, y)
            relevant_features = np.where(
                chi2_statistics > self.chi2_threshold)[0]
            X_features = hstack([X_features, X_counts[:, relevant_features]])

            old_length_vocab = len(self.vocabulary_)
            vocabulary = {value: key
                          for (key, value) in vectorizer.vocabulary_.items()}
            for i, idx in enumerate(relevant_features):
                self.vocabulary_[i + old_length_vocab] = \
                    str(window_size) + " " + vocabulary[idx]

            self._relevant_features_list.append(relevant_features)
            self._sfa_list.append(sfa)
            self._vectorizer_list.append(vectorizer)

        if not self.sparse:
            return X_features.A
        return csr_matrix(X_features)

    def _check_params(self, n_timestamps):
        if not isinstance(self.word_size, (int, np.integer)):
            raise TypeError("'word_size' must be an integer.")
        if not self.word_size >= 1:
            raise ValueError("'word_size' must be a positive integer.")

        if not isinstance(self.window_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'window_sizes' must be array-like.")
        window_sizes = check_array(self.window_sizes, ensure_2d=False,
                                   dtype=None)
        if window_sizes.ndim != 1:
            raise ValueError("'window_sizes' must be one-dimensional.")
        if not issubclass(window_sizes.dtype.type, (np.integer, np.floating)):
            raise ValueError("The elements of 'window_sizes' must be integers "
                             "or floats.")
        if issubclass(window_sizes.dtype.type, np.floating):
            if not (np.min(window_sizes) > 0 and np.max(window_sizes) <= 1):
                raise ValueError(
                    "If the elements of 'window_sizes' are floats, they all "
                    "must be greater than 0 and lower than or equal to 1."
                )
            window_sizes = np.ceil(window_sizes * n_timestamps).astype('int64')
        if not np.max(window_sizes) <= n_timestamps:
            raise ValueError("All the elements in 'window_sizes' must be "
                             "lower than or equal to n_timestamps.")

        if self.drop_sum and not self.word_size < np.min(window_sizes):
            raise ValueError(
                "If 'drop_sum=True', 'word_size' must be lower than "
                "the minimum value in 'window_sizes'."
            )
        if not (self.drop_sum or self.word_size <= np.min(window_sizes)):
            raise ValueError(
                "If 'drop_sum=False', 'word_size' must be lower than or "
                "equal to the minimum value in 'window_sizes'."
            )

        if not ((self.window_steps is None)
                or isinstance(self.window_steps, (list, tuple, np.ndarray))):
            raise TypeError("'window_steps' must be None or array-like.")
        if self.window_steps is None:
            window_steps = window_sizes
        else:
            window_steps = check_array(self.window_steps, ensure_2d=False,
                                       dtype=None)
            if window_steps.ndim != 1:
                raise ValueError("'window_steps' must be one-dimensional.")
            if window_steps.size != window_sizes.size:
                raise ValueError("If 'window_steps' is not None, it must have "
                                 "the same size as 'window_sizes'.")
            if not issubclass(window_steps.dtype.type,
                              (np.integer, np.floating)):
                raise ValueError(
                    "If 'window_steps' is not None, the elements of "
                    "'window_steps' must be integers or floats."
                )
            if issubclass(window_steps.dtype.type, np.floating):
                if not ((np.min(window_steps) > 0
                         and np.max(window_steps) <= 1)):
                    raise ValueError(
                        "If the elements of 'window_steps' are floats, they "
                        "all must be greater than 0 and lower than or equal "
                        "to 1."
                    )
                window_steps = np.ceil(
                    window_steps * n_timestamps).astype('int64')
            if not ((np.min(window_steps) >= 1)
                    and (np.max(window_steps) <= n_timestamps)):
                raise ValueError("All the elements in 'window_steps' must be "
                                 "greater than or equal to 1 and lower than "
                                 "or equal to n_timestamps.")

        if not isinstance(self.chi2_threshold,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'chi2_threshold' must be a float or an "
                            "integer.")
        if not self.chi2_threshold > 0:
            raise ValueError("'chi2_threshold' must be positive.")
        return window_sizes, window_steps
