"""The :mod:`pyts.approximation` module includes approximation algorithms.

Implemented algorithms are:
- Piecewise Aggregate Approximation
- Discrete Fourier Transform
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ..utils import segmentation
from ..preprocessing import StandardScaler


standard_library.install_aliases()


class PAA(BaseEstimator, TransformerMixin):
    """Piecewise Aggregate Approximation.

    Parameters
    ----------
    window_size : int or None (default = None)
        Length of the sliding window.

    output_size : int or None (default = None)
        Size of the returned time series.

    overlapping : bool (default = True)
        When ``output_size`` is specified, the window size is fixed
        if ``overlapping=True`` and may vary if ``overlapping=False``.
        Ignored if ``window_size`` is specified.

    """

    def __init__(self, window_size=None, output_size=None, overlapping=True):
        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Reduce the dimensionality of each time series.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : array-like, shape = [n_samples, n_features]
            Transformed data.

        """
        # Check input data
        X = check_array(X)

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters and compute window_size if output_size is given
        window_size = self._check_params(n_samples, n_features)

        if window_size == 1:
            return X
        else:
            start, end, size = segmentation(n_features, window_size,
                                            self.overlapping, self.output_size)
            return np.apply_along_axis(self._paa, 1, X, start, end, size)

    def _check_params(self, n_samples, n_features):
        if (self.window_size is None and self.output_size is None):
            raise TypeError("'window_size' xor 'output_size' must be "
                            "specified.")
        elif (self.window_size is not None and self.output_size is not None):
            raise TypeError("'window_size' xor 'output_size' must be "
                            "specified.")
        elif (self.window_size is not None and self.output_size is None):
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.window_size, int):
                raise TypeError("'window_size' must be an integer.")
            if self.window_size < 1:
                raise ValueError("'window_size' must be greater than or equal "
                                 "to 1.")
            if self.window_size > n_features:
                raise ValueError(
                    "'window_size' must be lower or equal than the size of "
                    "each time series.")
            window_size = self.window_size
        else:
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.output_size, int):
                raise TypeError("'output_size' must be an integer.")
            if self.output_size < 1:
                raise ValueError("'output_size' must be greater or equal than "
                                 " 1.")
            if self.output_size > n_features:
                raise ValueError(
                    "'output_size' must be lower or equal than the size of "
                    "each time series.")
            window_size = n_features // self.output_size
            window_size += 0 if n_features % self.output_size == 0 else 1
        return window_size

    def _paa(self, ts, start, end, size):
        return np.array([ts[start[i]:end[i]].mean() for i in range(size)])


class DFT(BaseEstimator, TransformerMixin):
    """Discrete Fourier Transform.

    Parameters
    ----------
    n_coefs : None or int (default = None)
        The number of Fourier coefficients to keep. If ``n_coefs=None``,
        all Fourier coefficients are returned. If ``n_coefs`` is an integer,
        the ``n_coefs`` most significant Fourier coefficients are returned if
        ``anova=True``, otherwise the first ``n_coefs`` Fourier coefficients
        are returned. A even number is required (for real and imaginary values)
        if ``anova=False``.

    anova : bool (default = False)
        If True, the Fourier coefficients selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    norm_mean : bool (default = True)
        If True, center the data before scaling. If ``norm_mean=True`` and
        ``anova=False``, the first Fourier coefficient will be dropped.

    norm_std : bool (default = True)
        If True, scale the data to unit variance.

    Attributes
    ----------
    coefs_ : array-like, shape [n_coefs]
        Indices of the Fourier coefficients that are kept.

    """

    def __init__(self, n_coefs=None, anova=False,
                 norm_mean=True, norm_std=True):
        self.n_coefs = n_coefs
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : None or array-like, shape = [n_samples] (default = None)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        # Check parameters
        if (not isinstance(self.n_coefs, int)) and (self.n_coefs is not None):
            raise TypeError("'n_coefs' must be None or an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if not isinstance(self.anova, (int, float)):
            raise TypeError("'anova' must be a boolean.")
        if (not self.anova) and isinstance(self.n_coefs, int):
            if self.n_coefs % 2 != 0:
                raise ValueError("If 'anova' = False, 'n_coefs' must be an "
                                 "even integer.")

        if not self.anova:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)

        n_samples, n_features = X.shape
        self._n_features = n_features

        # Normalization
        ss = StandardScaler(self.norm_mean, self.norm_std)
        X = ss.fit_transform(X)

        if not self.anova:
            if self.n_coefs is None:
                self.coefs_ = np.arange(2 * self._n_features)
            else:
                if self.norm_mean:
                    self.coefs_ = np.arange(2, self.n_coefs + 2)
                else:
                    self.coefs_ = np.arange(self.n_coefs)
        else:
            X_fft = np.fft.fft(X, axis=0)
            X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
            X_fft = X_fft.reshape(n_samples, -1, order='F')
            self.coefs_ = self._anova_selection(X_fft, y)

        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        Returns
        -------
        X_new : array-like, shape [n_samples, n_coefs]
            The selected Fourier coefficients for each sample.

        """
        # Check fitted
        check_is_fitted(self, ['coefs_', '_n_features'])

        # Check X
        X = check_array(X)
        if X.shape[1] != self._n_features:
            raise ValueError("The length of each time series (X.shape[1]) "
                             "does not match the length of each time series "
                             "when the estimator was fitted.")

        n_samples = X.shape[0]

        # Normalization
        ss = StandardScaler(self.norm_mean, self.norm_std)
        X = ss.fit_transform(X)

        # Fast Fourier Transform
        X_fft = np.fft.fft(X)
        X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
        X_fft = X_fft.reshape(n_samples, -1, order='F')

        return X_fft[:, self.coefs_]

    def fit_transform(self, X, y=None):
        """Fit the model than transform the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : None or array-like, shape = [n_samples] (default = None)
            Class labels for each data sample.

        Returns
        -------
        X_new : array-like, shape [n_samples, n_coefs]
            The selected Fourier coefficients for each sample.

        """
        # Check parameters
        if (not isinstance(self.n_coefs, int)) and (self.n_coefs is not None):
            raise TypeError("'n_coefs' must be None or an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if not isinstance(self.anova, (int, float)):
            raise TypeError("'anova' must be a boolean.")
        if (not self.anova) and isinstance(self.n_coefs, int):
            if self.n_coefs % 2 != 0:
                raise ValueError("'n_coefs' must be an even integer.")

        if not self.anova:
            X = check_array(X)
        else:
            X, y = check_X_y(X, y)

        n_samples, n_features = X.shape
        self._n_features = n_features

        # Normalization
        ss = StandardScaler(self.norm_mean, self.norm_std)
        X = ss.fit_transform(X)

        X_fft = np.fft.fft(X)
        X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
        X_fft = X_fft.reshape(n_samples, -1, order='F')

        if not self.anova:
            if self.n_coefs is None:
                self.coefs_ = np.arange(2 * self._n_features)
            else:
                if self.norm_mean:
                    self.coefs_ = np.arange(2, self.n_coefs + 2)
                else:
                    self.coefs_ = np.arange(self.n_coefs)
        else:
            self.coefs_ = self._anova_selection(X_fft, y)

        return X_fft[:, self.coefs_]

    def _anova_selection(self, X, y):

        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size

        masks = []
        for cur_class in range(n_classes):
            masks.append(y_ind == cur_class)

        within_std = np.sum([X[class_mask].std(axis=0)
                             for class_mask in masks], axis=0)
        feature_mask = np.where(within_std > 0)[0]
        if feature_mask.size == 0:
            raise ZeroDivisionError("The within-group mean square value is "
                                    "equal to zero for every feature.")
        F, _ = f_classif(X[:, feature_mask], y)
        if isinstance(self.n_coefs, int):
            return feature_mask[np.argsort(F)[-self.n_coefs:][::-1]]
        else:
            return feature_mask[np.argsort(F)[::-1]]
