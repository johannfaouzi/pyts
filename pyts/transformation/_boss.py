"""Code for Bag-of-SFA Symbols."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import ceil
from typing import Self

import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted

from pyts._base import UnivariateTransformerMixin
from pyts.approximation import SymbolicFourierApproximation
from pyts.utils._utils import _windowed_view


class BOSS(BaseEstimator, UnivariateTransformerMixin):
    """Bag of Symbolic Fourier Approximation Symbols.

    For each time series, subseries are extracted using a sliding window.
    Then the subseries are transformed into a word using the Symbolic
    Fourier Approximation (SFA) algorithm. For each time series, the words
    are grouped together and a histogram counting the occurences of each
    word is created.

    Parameters
    ----------
    word_size : int (default = 4)
        Size of each word.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and 26.

    strategy : str (default = 'quantile')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution
        - 'entropy': Bin edges are computed using information gain

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

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    sparse : bool (default = True)
        Return a sparse matrix if True, else return an array.

    alphabet : None, 'ordinal' or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] P. Schäfer, "The BOSS is concerned with time series classification
           in the presence of noise". Data Mining and Knowledge Discovery,
           29(6), 1505-1530 (2015).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.transformation import BOSS
    >>> X_train, X_test, _, _ = load_gunpoint(return_X_y=True)
    >>> boss = BOSS(word_size=2, n_bins=2, sparse=False)
    >>> boss.fit(X_train)
    BOSS(...)
    >>> sorted(boss.vocabulary_.values())
    ['aa', 'ab', 'ba', 'bb']
    >>> boss.transform(X_test)
    array(...)

    """

    def __init__(
        self,
        word_size: int = 4,
        n_bins: int = 4,
        strategy: str = 'quantile',
        window_size: int | float = 10,
        window_step: int | float = 1,
        anova: bool = False,
        drop_sum: bool = False,
        norm_mean: bool = False,
        norm_std: bool = False,
        numerosity_reduction: bool = True,
        sparse: bool = True,
        alphabet: npt.ArrayLike | None = None,
    ) -> None:
        self.word_size = word_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.window_size = window_size
        self.window_step = window_step
        self.anova = anova
        self.drop_sum = drop_sum
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.numerosity_reduction = numerosity_reduction
        self.sparse = sparse
        self.alphabet = alphabet

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> Self:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : None or array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        if y is not None:
            check_classification_targets(y)

        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, window_size, window_step
        )
        X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

        sfa = SymbolicFourierApproximation(
            n_coefs=self.word_size,
            drop_sum=self.drop_sum,
            anova=self.anova,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            n_bins=self.n_bins,
            strategy=self.strategy,
            alphabet=self.alphabet,
        )
        if y is None:
            y_repeated = None
        else:
            y_repeated = np.repeat(y, n_windows)
        X_sfa = sfa.fit_transform(X_windowed, y_repeated)

        X_word = np.asarray([''.join(X_sfa[i]) for i in range(n_samples * n_windows)])
        X_word = X_word.reshape(n_samples, n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1], np.full(n_samples, True)]
            X_bow = np.asarray(
                [' '.join(X_word[i, not_equal[i]]) for i in range(n_samples)]
            )
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        vectorizer = CountVectorizer()
        vectorizer.fit(X_bow)
        self.vocabulary_ = {value: key for key, value in vectorizer.vocabulary_.items()}
        self._window_size = window_size
        self._window_step = window_step
        self._n_windows = n_windows
        self._sfa = sfa
        self._vectorizer = vectorizer
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.int64] | csr_matrix:
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X_new : sparse matrix, shape = (n_samples, n_words)
            Document-term matrix.

        """
        check_is_fitted(self, ['_sfa', '_vectorizer', 'vocabulary_'])
        X = check_array(X)
        n_samples, n_timestamps = X.shape

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, self._window_size, self._window_step
        )
        X_windowed = X_windowed.reshape(-1, self._window_size)

        X_sfa = self._sfa.transform(X_windowed)
        X_word = np.asarray([''.join(X_sfa[i]) for i in range(X_sfa.shape[0])])
        X_word = X_word.reshape(n_samples, self._n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1], np.full(n_samples, True)]
            X_bow = np.asarray(
                [' '.join(X_word[i, not_equal[i]]) for i in range(n_samples)]
            )
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        X_boss = self._vectorizer.transform(X_bow)
        if not self.sparse:
            # sklearn has no py.typed stubs, so CountVectorizer.transform's
            # return type is inferred from source as the generic `spmatrix`
            # base, which lacks `toarray` statically even though every
            # concrete sparse matrix (what is actually returned) has it.
            return X_boss.toarray()  # pyrefly: ignore[missing-attribute]
        return csr_matrix(X_boss)

    # This overrides UnivariateTransformerMixin.fit_transform with a fully
    # custom implementation (rather than delegating to fit().transform())
    # and, by pre-existing design, does not accept arbitrary **fit_params.
    def fit_transform(  # pyrefly: ignore[bad-override]
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.int64] | csr_matrix:
        """Fit the data then transform it.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : None or array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        X_new : sparse matrix, shape = (n_samples, n_words)
            Document-term matrix.

        """
        X = check_array(X)
        n_samples, n_timestamps = X.shape
        if y is not None:
            check_classification_targets(y)

        window_size, window_step = self._check_params(n_timestamps)
        n_windows = (n_timestamps - window_size + window_step) // window_step

        X_windowed = _windowed_view(
            X, n_samples, n_timestamps, window_size, window_step
        )
        X_windowed = X_windowed.reshape(n_samples * n_windows, window_size)

        sfa = SymbolicFourierApproximation(
            n_coefs=self.word_size,
            drop_sum=self.drop_sum,
            anova=self.anova,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
            n_bins=self.n_bins,
            strategy=self.strategy,
            alphabet=self.alphabet,
        )
        if y is None:
            y_repeated = None
        else:
            y_repeated = np.repeat(y, n_windows)
        X_sfa = sfa.fit_transform(X_windowed, y_repeated)

        X_word = np.asarray([''.join(X_sfa[i]) for i in range(n_samples * n_windows)])
        X_word = X_word.reshape(n_samples, n_windows)

        if self.numerosity_reduction:
            not_equal = np.c_[X_word[:, 1:] != X_word[:, :-1], np.full(n_samples, True)]
            X_bow = np.asarray(
                [' '.join(X_word[i, not_equal[i]]) for i in range(n_samples)]
            )
        else:
            X_bow = np.asarray([' '.join(X_word[i]) for i in range(n_samples)])

        vectorizer = CountVectorizer()
        X_boss = vectorizer.fit_transform(X_bow)
        self.vocabulary_ = {value: key for key, value in vectorizer.vocabulary_.items()}
        self._window_size = window_size
        self._window_step = window_step
        self._n_windows = n_windows
        self._sfa = sfa
        self._vectorizer = vectorizer
        if not self.sparse:
            # Same generic-`spmatrix`-inference issue as in transform()
            # above: CountVectorizer.fit_transform really returns a
            # concrete sparse matrix, which does have `toarray`.
            return X_boss.toarray()  # pyrefly: ignore[missing-attribute]
        return csr_matrix(X_boss)

    def _check_params(self, n_timestamps: int) -> tuple[int, int]:
        if not isinstance(self.word_size, (int, np.integer)):
            raise TypeError("'word_size' must be an integer.")
        if not self.word_size >= 1:
            raise ValueError("'word_size' must be a positive integer.")

        if not isinstance(self.window_size, (int, np.integer, float, np.floating)):
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

        if not isinstance(self.window_step, (int, np.integer, float, np.floating)):
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
