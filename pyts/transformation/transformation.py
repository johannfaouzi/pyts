"""The :mod:`pyts.transformation` module includes transformation algorithms.

Implemented algorithms are:
- Bag-of-SFA Symbols
- Word ExtrAction for time SEries cLassification
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ..quantization import SFA
from ..utils import numerosity_reduction


standard_library.install_aliases()


class BOSS(BaseEstimator, TransformerMixin):
    """Bag-of-SFA Symbols.

    Parameters
    ----------
    n_coefs : None or int (default = None)
        The number of Fourier coefficients to keep. If ``n_coefs=None``,
        all Fourier coefficients are returned. If ``n_coefs`` is an integer,
        the ``n_coefs`` most significant Fourier coefficients are returned if
        ``anova=True``, otherwise the first ``n_coefs`` Fourier coefficients
        are returned. A even number is required (for real and imaginary values)
        if ``anova=False``.

    window_size : int
        The size of the window.

    anova : bool (default = False)
        If True, the Fourier coefficients selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    norm_mean : bool (default = True)
        If True, center the data before scaling. If ``norm_mean=True`` and
        ``anova=False``, the first Fourier coefficient will be dropped.

    norm_std : bool (default = True)
        If True, scale the data to unit variance.

    n_bins : int (default = 4)
        The number of bins. Ignored if ``quantiles='entropy'``.

    quantiles : {'gaussian', 'empirical'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used.

    variance_selection : bool (default = False)
        If True, the Fourier coefficients with low variance are removed.

    variance_threshold : float (default = 0.)
        Fourier coefficients with a training-set variance lower than this
        threshold will be removed. Ignored if ``variance_selection=False``.

    numerosity_reduction : bool (default = True)
        If True, numerosity reduction is applied: When the same word
        occurs several times in a row, only one instance of this word is kept.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of features indices to terms.

    """

    def __init__(self, n_coefs, window_size, anova=False, norm_mean=True,
                 norm_std=True, n_bins=4, quantiles='empirical',
                 variance_selection=False, variance_threshold=0.,
                 numerosity_reduction=True):
        self.n_coefs = n_coefs
        self.window_size = window_size
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.variance_selection = variance_selection
        self.variance_threshold = variance_threshold
        self.numerosity_reduction = numerosity_reduction

    def fit(self, X, y=None, overlapping=True):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y :
            Ignored.

        overlapping : boolean (default = True)
            whether or not overlapping windows are used for the training
            phase.

        Returns
        -------
        self : object

        """
        # Check input data
        X = check_array(X)
        n_samples, n_features = X.shape

        # Check parameters
        if (not isinstance(self.n_coefs, int)) and (self.n_coefs is not None):
            raise TypeError("'n_coefs' must be None or an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if isinstance(self.n_coefs, int) and self.n_coefs % 2 != 0:
            raise ValueError("'n_coefs' must be an even integer.")
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower than or equal to "
                             "the size of each time series.")
        if isinstance(self.n_coefs, int) and self.n_coefs > self.window_size:
            raise ValueError("'n_coefs' must be lower than or equal to "
                             "'window_size'.")
        if not isinstance(self.norm_mean, (int, float)):
            raise TypeError("'norm_mean' must be a boolean.")
        if not isinstance(self.norm_std, (int, float)):
            raise TypeError("'norm_std' must be a boolean.")
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.quantiles not in ['empirical', 'gaussian']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")
        if not isinstance(self.variance_selection, (int, float)):
            raise TypeError("'variance_selection' must be a boolean.")
        if not isinstance(self.variance_threshold, (int, float)):
            raise TypeError("'variance_threshold' must be a float.")
        if not isinstance(self.numerosity_reduction, (int, float)):
            raise TypeError("'numerosity_reduction' must be a boolean.")
        if not isinstance(overlapping, (int, float)):
            raise TypeError("'overlapping' must be a boolean.")

        self.vocabulary_ = {}

        if overlapping:
            n_windows = n_features - self.window_size + 1
            X_window = np.asarray([X[:, i: i + self.window_size]
                                   for i in range(n_windows)])
            X_window = X_window.reshape(n_samples * n_windows, -1, order='F')
        else:
            n_windows = n_features // self.window_size
            remainder = n_features % self.window_size
            if remainder == 0:
                window_idx = np.array_split(np.arange(0, n_features),
                                            n_windows)
            else:
                split_idx = np.arange(self.window_size,
                                      n_windows * (self.window_size + 1),
                                      self.window_size)
                window_idx = np.split(np.arange(0, n_features), split_idx)[:-1]
            X_window = X[:, window_idx].reshape(n_samples * n_windows, -1)

        sfa = SFA(self.n_coefs, False, self.norm_mean,
                  self.norm_std, self.n_bins, self.quantiles,
                  self.variance_selection, self.variance_threshold)
        count = CountVectorizer(ngram_range=(1, 1))

        X_sfa = sfa.fit_transform(X_window)
        X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_sfa).reshape(n_samples, -1)
        word_size = len(X_sfa[0, 0])
        if word_size == 1:
            count.set_params(tokenizer=self._tok)
        if self.numerosity_reduction:
            X_sfa = np.apply_along_axis(numerosity_reduction, 1, X_sfa)
        else:
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)
        count.fit(X_sfa)

        for key, value in count.vocabulary_.items():
            self.vocabulary_[value] = key

        self._sfa = sfa
        self._count = count
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : sparse matrix, shape [n_samples, n_words]
            Document-term matrix.

        """
        # Check fitted
        check_is_fitted(self, ['_sfa', '_count', 'vocabulary_'])

        # Check X
        X = check_array(X)
        n_samples, n_features = X.shape

        n_windows = n_features - self.window_size + 1
        X_window = np.asarray([X[:, i: i + self.window_size]
                               for i in range(n_windows)])
        X_window = X_window.reshape(n_samples * n_windows, -1, order='F')

        X_sfa = self._sfa.transform(X_window)
        X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_sfa).reshape(n_samples, -1)
        if self.numerosity_reduction:
            X_sfa = np.apply_along_axis(numerosity_reduction, 1, X_sfa)
        else:
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)
        tf = self._count.transform(X_sfa)
        return tf

    def fit_transform(self, X, y=None, overlapping=True):
        """Fit the data then transform it.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : sparse matrix, shape [n_samples, n_words]
            Document-term matrix.

        """
        # Check input data
        X = check_array(X)
        n_samples, n_features = X.shape

        # Check parameters
        if (not isinstance(self.n_coefs, int)) and (self.n_coefs is not None):
            raise TypeError("'n_coefs' must be None or an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if isinstance(self.n_coefs, int) and self.n_coefs % 2 != 0:
            raise ValueError("'n_coefs' must be an even integer.")
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower than or equal to "
                             "the size of each time series.")
        if isinstance(self.n_coefs, int) and self.n_coefs > self.window_size:
            raise ValueError("'n_coefs' must be lower than or equal to "
                             "'window_size'.")
        if not isinstance(self.norm_mean, (int, float)):
            raise TypeError("'norm_mean' must be a boolean.")
        if not isinstance(self.norm_std, (int, float)):
            raise TypeError("'norm_std' must be a boolean.")
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.quantiles not in ['empirical', 'gaussian']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")
        if not isinstance(self.variance_selection, (int, float)):
            raise TypeError("'variance_selection' must be a boolean.")
        if not isinstance(self.variance_threshold, (int, float)):
            raise TypeError("'variance_threshold' must be a float.")
        if not isinstance(self.numerosity_reduction, (int, float)):
            raise TypeError("'numerosity_reduction' must be a boolean.")
        if not isinstance(overlapping, (int, float)):
            raise TypeError("'overlapping' must be a boolean.")

        self.vocabulary_ = {}

        if overlapping:
            n_windows = n_features - self.window_size + 1
            X_window = np.asarray([X[:, i: i + self.window_size]
                                   for i in range(n_windows)])
            X_window = X_window.reshape(n_samples * n_windows, -1, order='F')
        else:
            n_windows = n_features // self.window_size
            remainder = n_features % self.window_size
            if remainder == 0:
                window_idx = np.array_split(np.arange(0, n_features),
                                            n_windows)
            else:
                split_idx = np.arange(self.window_size,
                                      n_windows * (self.window_size + 1),
                                      self.window_size)
                window_idx = np.split(np.arange(0, n_features), split_idx)[:-1]
            X_window = X[:, window_idx].reshape(n_samples * n_windows, -1)

        sfa = SFA(self.n_coefs, False, self.norm_mean,
                  self.norm_std, self.n_bins, self.quantiles,
                  self.variance_selection, self.variance_threshold)
        count = CountVectorizer(ngram_range=(1, 1))

        X_sfa = sfa.fit_transform(X_window)
        X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_sfa).reshape(n_samples, -1)
        word_size = len(X_sfa[0, 0])
        if word_size == 1:
            count.set_params(tokenizer=self._tok)
        if self.numerosity_reduction:
            X_sfa = np.apply_along_axis(numerosity_reduction, 1, X_sfa)
        else:
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)
        tf = count.fit_transform(X_sfa)

        for key, value in count.vocabulary_.items():
            self.vocabulary_[value] = key
        return tf

    def _tok(self, x):
        return x.split(' ')


class WEASEL(BaseEstimator, TransformerMixin):
    """Word ExtrAction for time SEries cLassification.

    Parameters
    ----------
    n_coefs : int
        The number of Fourier coefficients to keep. The `n_coefs` most
        significant Fourier coefficients are returned.

    window_sizes : array-like
        The size of the windows.

    anova : bool (default = False)
        If True, the Fourier coefficients selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    norm_mean : bool (default = True)
        If True, center the data before scaling. If ``norm_mean=True`` and
        ``anova=False``, the first Fourier coefficient will be dropped.

    norm_std : bool (default = True)
        If True, scale the data to unit variance.

    n_bins : int (default = 4)
        The number of bins (also known as the size of the alphabet).

    variance_selection : bool (default = False)
        If True, the Fourier coefficients with low variance are removed.

    variance_threshold : float (default = 0.)
        Fourier coefficients with a training-set variance lower than this
        threshold will be removed. Ignored if ``variance_selection=False``.

    pvalue_threshold : float (default = 0.9)
        threshold for the feature selection. Features with p-values above
        'pvalue_threshold' for the Chi-2 test are kept.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of features indices to terms.

    """

    def __init__(self, n_coefs, window_sizes, norm_mean=True,
                 norm_std=True, n_bins=4, variance_selection=False,
                 variance_threshold=0., pvalue_threshold=0.9):
        self.n_coefs = n_coefs
        self.window_sizes = window_sizes
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.n_bins = n_bins
        self.variance_selection = variance_selection
        self.variance_threshold = variance_threshold
        self.pvalue_threshold = pvalue_threshold

    def fit(self, X, y, overlapping=False):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Class labels for each data sample.

        overlapping : boolean (default = False)
            If True, overlapping windows are used.

        Returns
        -------
        self : object

        """
        # Check parameters
        if not isinstance(self.n_coefs, int):
            raise TypeError("'n_coefs' must be an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if not isinstance(self.window_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'window_sizes' must be array-like.")
        if (isinstance(self.n_coefs, int) and
                self.n_coefs > np.min(self.window_sizes)):
            raise ValueError("'n_coefs' must be lower than or equal to the "
                             "minimum value in 'window_sizes'.")
        if not isinstance(self.norm_mean, (int, float)):
            raise TypeError("'norm_mean' must be a boolean.")
        if not isinstance(self.norm_std, (int, float)):
            raise TypeError("'norm_std' must be a boolean.")
        if not isinstance(self.n_bins, int):
            raise ValueError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if not isinstance(self.variance_selection, (int, float)):
            raise TypeError("'variance_selection' must be a boolean.")
        if not isinstance(self.variance_threshold, (int, float)):
            raise TypeError("'variance_threshold' must be a float.")
        if not isinstance(self.pvalue_threshold, (int, float)):
            raise TypeError("'pvalue_threshold' must be a float or an "
                            "integer.")
        if (self.pvalue_threshold < 0) or (self.pvalue_threshold > 1):
            raise ValueError("'pvalue_threshold' must be between 0 and 1.")
        if not isinstance(overlapping, (int, float)):
            raise TypeError("'overlapping' must be a boolean.")

        # Check X and y
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        n_samples, n_features = X.shape

        # Save parameters
        self._sfa_list = []
        self._count_list = []
        self._relevant_features_list = []
        self.vocabulary_ = {}

        for window_size in self.window_sizes:
            if overlapping:
                n_windows = n_features - window_size + 1
                X_window = np.asarray([X[:, i: i + window_size]
                                       for i in range(n_windows)])
                X_window = X_window.reshape(n_samples * n_windows,
                                            -1,
                                            order='F')
            else:
                n_windows = n_features // window_size
                remainder = n_features % window_size
                if remainder == 0:
                    window_idx = np.array_split(np.arange(0, n_features),
                                                n_windows)
                else:
                    split_idx = np.arange(window_size,
                                          (n_windows + 1) * window_size,
                                          window_size)
                    window_idx = np.split(np.arange(0, n_features),
                                          split_idx)[:-1]
                X_window = X[:, window_idx].reshape(n_samples * n_windows, -1)

            sfa = SFA(self.n_coefs, True, self.norm_mean,
                      self.norm_std, self.n_bins, 'entropy',
                      self.variance_selection, self.variance_threshold)
            count = CountVectorizer(ngram_range=(1, 2))

            y_window = np.repeat(y_ind, n_windows)
            X_sfa = sfa.fit_transform(X_window, y_window)
            X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                        1,
                                        X_sfa).reshape(n_samples, -1)
            word_size = len(X_sfa[0, 0])
            if word_size == 1:
                count.set_params(tokenizer=self._tok)
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)

            tf = count.fit_transform(X_sfa)
            _, pval = chi2(tf, y_ind)
            relevant_features = np.where(pval > self.pvalue_threshold)[0]

            old_size = len(self.vocabulary_)
            for key, value in count.vocabulary_.items():
                idx = np.where(relevant_features == value)[0]
                if idx.size == 1:
                    word = str(window_size) + " " + key
                    self.vocabulary_[idx[0] + old_size] = word

            self._relevant_features_list.append(relevant_features)
            self._sfa_list.append(sfa)
            self._count_list.append(count)

        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        Returns
        -------
        X_new : sparse matrix, shape [n_samples, n_relevant_features]
            Document-term matrix with relevant features only.

        """
        # Check fitted
        check_is_fitted(self, ['_relevant_features_list', '_sfa_list',
                               '_count_list', 'vocabulary_'])

        # Check X
        X = check_array(X)
        n_samples, n_features = X.shape

        X_features = scipy.sparse.csr_matrix((n_samples, 0), dtype=np.int64)

        for (window_size, sfa, count,
             relevant_features) in zip(self.window_sizes,
                                       self._sfa_list,
                                       self._count_list,
                                       self._relevant_features_list):

            n_windows = n_features - window_size + 1
            X_window = np.asarray([X[:, i: i + window_size]
                                   for i in range(n_windows)])
            X_window = X_window.reshape(n_samples * n_windows, -1, order='F')

            X_sfa = sfa.transform(X_window)
            X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                        1,
                                        X_sfa).reshape(n_samples, -1)
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)
            tf = count.transform(X_sfa)

            X_features = scipy.sparse.hstack([X_features,
                                              tf[:, relevant_features]])

        return X_features

    def _tok(self, x):
        return x.split(' ')
