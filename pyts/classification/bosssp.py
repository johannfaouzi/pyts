"""Code for Bag-of-SFA Symbols using Spatial Pyramids."""

# Author: Sven Barray
# License: BSD-3-Clause

import numpy as np
from collections import Counter
from math import ceil
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator
from ..approximation import SymbolicFourierApproximation
from ..base import UnivariateClassifierMixin
from ..utils.utils import _windowed_view


class BOSSSP(BaseEstimator, UnivariateClassifierMixin):
    """Bag-of-SFA Symbols using Spatial Pyramids.

    The time series is transformed into an histogram using the
    Bag-of-SFA Symbols (BOSS) algorithm. Then, the time series is divided
    into smaller series on which the BOSS algorithm is applied again.
    A final histogram is produced, combining the histograms of all the
    series used.

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

    level : integer (default = 3)
         Number of times the series is being divided. Maximum of 3.

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

    References
    ----------
    .. [1] James Large et al., “On time series classification with
           dictionary-based classifiers”. Intelligent Data Analysis 23.5 (2019)

    """

    def __init__(self, word_size=4, n_bins=4, window_size=10, level=3,
                 window_step=1, anova=False, drop_sum=False, norm_mean=False,
                 norm_std=False, strategy='quantile', alphabet=None,
                 numerosity_reduction=True, use_idf=True, smooth_idf=False,
                 sublinear_tf=True):
        self.word_size = word_size
        self.n_bins = n_bins
        self.window_size = window_size
        self.level = level
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
        length = len(X[0])
        words = self._boss_word_extractor(X, y, 1)

        if self.level >= 2:
            X_2 = [ts[:length//2] for ts in X]
            X_3 = [ts[length//2:] for ts in X]
            for ts in range(len(X)):
                words_2 = self._boss_word_extractor(X_2, y, 2)[ts]
                words_3 = self._boss_word_extractor(X_3, y, 3)[ts]
                words[ts] = words[ts] + words_2 + words_3

        if self.level == 3:
            X_4 = [ts[:length//4] for ts in X]
            X_5 = [ts[length//4:length//2] for ts in X]
            X_6 = [ts[length//2:length*3//4] for ts in X]
            X_7 = [ts[length*3//4:] for ts in X]
            for ts in range(len(X)):
                words_4 = self._boss_word_extractor(X_4, y, 4)[ts]
                words_5 = self._boss_word_extractor(X_5, y, 5)[ts]
                words_6 = self._boss_word_extractor(X_6, y, 6)[ts]
                words_7 = self._boss_word_extractor(X_7, y, 7)[ts]
                words[ts] = words[ts] + words_4 + words_5 + words_6 + words_7

        full_sorted_wordcount = []
        for ts in words:
            wordcount_current_ts = dict(Counter(ts))
            sorted_wordcount_current_ts = {key: value for key, value in sorted(
                                            wordcount_current_ts.items())}
            full_sorted_wordcount.append(sorted_wordcount_current_ts)

        self._word_count = full_sorted_wordcount
        return self

    def _boss_word_extractor(self, X, y, distinguishing_stamp):
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        check_classification_targets(y)

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

        full_words_list = []
        for time_series in X_bow:
            wl_current_ts = np.asarray(time_series.split(' ')).tolist()
            wl_current_ts_with_stamp = [str(distinguishing_stamp) + '' + word
                                        for word in wl_current_ts]
            full_words_list.append(wl_current_ts_with_stamp)
        return full_words_list

    def decision_function(self, X):
        pass

    def predict(self, X):
        pass

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
                    "If 'drop_sum=True', 'word size' must be lower than or"
                    "equal to (window_size - 1) if 'level=1', lower than or"
                    "equal to (window_size//2 - 1) if 'level = 2' and lower"
                    "or equal to (window_size//4 - 1) if 'level = 3'"
                )
        else:
            if not self.word_size <= window_size:
                raise ValueError(
                    "If 'drop_sum=False', 'word size' must be lower than or"
                    "equal to window_size if 'level=1', lower than or"
                    "equal to (window_size//2) if 'level = 2' and lower"
                    "or equal to (window_size//4) if 'level = 3'"
                )

        if not isinstance(self.level, (int, np.integer)):
            raise TypeError("'level' must be an integer.")
        if not self.level >= 1:
            raise ValueError("'level' must be a positive integer.")
        if self.level >= 4:
            raise ValueError("'level' must not exceed 3.")

        return window_size, window_step
