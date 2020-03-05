"""Code for Bag-of-SFA Symbols in Vector Space."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from math import ceil
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from ..approximation import SymbolicFourierApproximation
from ..utils.utils import _windowed_view


class BOSSVS(BaseEstimator, ClassifierMixin):
    """Bag-of-SFA Symbols in Vector Space.

    Each time series is transformed into an histogram using the
    Bag-of-SFA Symbols (BOSS) algorithm. Then, for each class, the histograms
    are added up and a tf-idf vector is computed. The predicted class for
    a new sample is the class giving the highest cosine similarity between
    its tf vector and the tf-idf vectors of each class.

    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    window_size : int or float (default = 10)
        Size of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_size * n_timestamps)``.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The window
        size will be computed as ``ceil(window_step * n_timestamps)``.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    norm_mean : bool (default = False)
        If True, center each subseries before scaling.

    norm_std : bool (default = False)
        If True, scale each subseries to unit variance.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = False)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = True)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    idf_ : array, shape = (n_features,) , or None
        The learned idf vector (global term weights) when ``use_idf=True``,
        None otherwise.

    tfidf_ : array, shape = (n_classes, n_words)
        Term-document matrix.

    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] P. Schäfer, "Scalable Time Series Classification". Data Mining and
           Knowledge Discovery, 30(5), 1273-1298 (2016).

    Examples
    --------
    >>> from pyts.classification import BOSSVS
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = BOSSVS(window_size=28)
    >>> clf.fit(X_train, y_train)
    BOSSVS(...)
    >>> clf.score(X_test, y_test)
    0.98

    """

    def __init__(self, word_size=4, n_bins=4, window_size=10, window_step=1,
                 anova=False, drop_sum=False, norm_mean=False, norm_std=False,
                 strategy='quantile', alphabet=None, numerosity_reduction=True,
                 use_idf=True, smooth_idf=False, sublinear_tf=True):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_size = window_size
        self.window_step = window_step
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.strategy = strategy
        self.alphabet = alphabet
        self.numerosity_reduction = numerosity_reduction
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):
        """Compute the document-term matrix.

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
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = self.classes_.size

        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, window_size, window_step
        )
        X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

        sfa = SymbolicFourierApproximation(
            n_coefs=self.word_size, drop_sum=self.drop_sum, anova=self.anova,
            norm_mean=self.norm_mean, norm_std=self.norm_std,
            n_bins=self.n_bins, strategy=self.strategy, alphabet=self.alphabet
        )
        y_repeated = np.repeat(y, n_windows)
        X_sfa = sfa.fit_transform(X_windowed, y_repeated)

        X_word = np.asarray([''.join(X_sfa[i])
                             for i in range(n_samples * n_windows)])
        X_word = X_word.reshape(n_samples, n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        X_class = np.array([' '.join(X_bow[y_ind == i])
                            for i in range(n_classes)])

        tfidf = TfidfVectorizer(
            norm=None, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf
        )
        self.tfidf_ = tfidf.fit_transform(X_class).toarray()
        self.vocabulary_ = {value: key for key, value in
                            tfidf.vocabulary_.items()}
        if self.use_idf:
            self.idf_ = tfidf.idf_
        else:
            self.idf_ = None

        self._window_size = window_size
        self._window_step = window_step
        self._n_windows = n_windows
        self._tfidf = tfidf
        self._sfa = sfa
        return self

    def decision_function(self, X):
        """Evaluate the cosine similarity between document-term matrix and X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X : array, shape (n_samples, n_classes)
            Cosine similarity between the document-term matrix and X.

        """
        check_is_fitted(self, ['vocabulary_', 'tfidf_', 'idf_', '_tfidf'])
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, self._window_size, self._window_step
        )
        X_windowed = X_windowed.reshape(-1, self._window_size)

        X_sfa = self._sfa.transform(X_windowed)
        X_word = np.asarray([''.join(X_sfa[i]) for i in range(X_sfa.shape[0])])
        X_word = X_word.reshape(n_samples, self._n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1],
                              np.full(n_samples, True)]
            X_bow = np.asarray([' '.join(X_word[i, not_equal[i]])
                                for i in range(n_samples)])
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        X_tf = self._tfidf.transform(X_bow).toarray()
        if self.idf_ is not None:
            X_tf /= self.idf_
        return cosine_similarity(X_tf, self.tfidf_)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        y_pred : array, shape = (n_samples,)
            Class labels for each data sample.

        """
        return self.classes_[self.decision_function(X).argmax(axis=1)]

    def _check_params(self, n_timestamps):
        if not isinstance(self.word_size, (int, np.integer)):
            raise TypeError("'word_size' must be an integer.")
        if not self.word_size >= 1:
            raise ValueError("'word_size' must be a positive integer.")

        if not isinstance(self.window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if self.drop_sum:
                if not 1 <= self.window_size <= (n_timestamps - 1):
                    raise ValueError(
                        "If 'window_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
                        "(n_timestamps - 1) if 'drop_sum=True'."
                    )
            else:
                if not 1 <= self.window_size <= n_timestamps:
                    raise ValueError(
                        "If 'window_size' is an integer, it must be greater "
                        "than or equal to 1 and lower than or equal to "
                        "n_timestamps if 'drop_sum=False'."
                    )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
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
                    "n_timestamps."
                )
            window_step = self.window_step
        else:
            if not 0 < self.window_step <= 1:
                raise ValueError(
                    "If 'window_step' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
                )
            window_step = ceil(self.window_step * n_timestamps)

        if self.drop_sum:
            if not self.word_size <= (window_size - 1):
                raise ValueError(
                    "'word_size' must be lower than or equal to "
                    "(window_size - 1) if 'drop_sum=True'."
                )
        else:
            if not self.word_size <= window_size:
                raise ValueError(
                    "'word_size' must be lower than or equal to "
                    "window_size if 'drop_sum=False'."
                )
        return window_size, window_step
