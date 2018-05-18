"""The :mod:`pyts.quantization` module includes quantization algorithms.

Implemented algorithms are:
- Symbolic Aggregate approXimation
- Multiple Coefficient Binning
- Symbolic Fourier Approximation
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
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from ..approximation import DFT


standard_library.install_aliases()


class SAX(BaseEstimator, TransformerMixin):
    """Symbolic Aggregate approXimation.

    Parameters
    ----------
    n_bins : int (default = 4)
        Number of bins (also known as the size of the alphabet).

    quantiles : {'gaussian', 'empirical'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used.

    """

    def __init__(self, n_bins=4, quantiles='gaussian'):
        self.n_bins = n_bins
        self.quantiles = quantiles

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        """
        return self

    def transform(self, X):
        """Quantize the time series.

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

        # Check parameters
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.n_bins > 26:
            raise ValueError("'n_bins' must be lower than or equal to 26.")
        if self.quantiles not in ['gaussian', 'empirical']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")

        # Compute alphabet
        alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])

        if self.quantiles == 'gaussian':
            bins = scipy.stats.norm.ppf(np.linspace(0, 1,
                                                    self.n_bins + 1
                                                    )[1:-1])
            indices = np.apply_along_axis(np.digitize, 1, X, bins)
        else:
            bins = np.percentile(X,
                                 np.linspace(0, 100, self.n_bins + 1)[1:-1],
                                 axis=1)
            indices = np.array([np.digitize(X[i], bins[:, i])
                                for i in range(n_samples)])
        return alphabet[indices]


class MCB(BaseEstimator, TransformerMixin):
    """Multiple Coefficient Binning.

    Parameters
    ----------
    n_bins : int (default = 4)
        The number of bins. Ignored if ``quantiles='entropy'``.

    quantiles : {'gaussian', 'empirical', 'entropy'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used. If 'entropy', quantiles are computed using
        the breakpoints leading to the maximum information gain.

    """

    def __init__(self, n_bins=4, quantiles='gaussian'):
        self.n_bins = n_bins
        self.quantiles = quantiles

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
        if not isinstance(self.n_bins, int):
            raise ValueError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.n_bins > 26:
            raise ValueError("'n_bins' must be lower than or equal to 26.")
        if self.quantiles not in ['empirical', 'gaussian', 'entropy']:
            raise ValueError("'quantiles' must be either 'gaussian', "
                             "'empirical' or 'entropy'.")
        if (self.quantiles == 'entropy') and (y is None):
            raise ValueError("The combination of quantiles = 'entropy' and "
                             "y = None is not possible.")

        # Check X
        if self.quantiles == 'entropy':
            X, y = check_X_y(X, y)
            check_classification_targets(y)
            le = LabelEncoder()
            y_ind = le.fit_transform(y)
        else:
            X = check_array(X)
        self._n_features = X.shape[1]

        if self.quantiles == 'empirical':
            self._bins = np.percentile(X, np.linspace(0, 100,
                                                      self.n_bins + 1
                                                      )[1: -1], axis=0)
        elif self.quantiles == 'gaussian':
            self._bins = scipy.stats.norm.ppf(np.linspace(0, 1,
                                                          self.n_bins + 1
                                                          )[1:-1])
        else:
            self._bins = np.zeros((self.n_bins - 1, self._n_features))
            for feature in range(self._n_features):
                self._bins[:, feature] = self._fit_bins(X[:, feature], y_ind)
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        Returns
        -------
        X_new : array-like, shape [n_samples, n_features]

        """
        # Check fitted
        check_is_fitted(self, ['_bins', '_n_features'])

        # Check X
        X = check_array(X)
        if X.shape[1] != self._n_features:
            raise ValueError("The length of each time series (X.shape[1]) "
                             "does not match the length of each time series "
                             "when the estimator was fitted.")

        # Compute alphabet
        alphabet = np.array([chr(i) for i in range(97, 97 + self.n_bins)])

        if self.quantiles == 'gaussian':
            indices = np.apply_along_axis(np.digitize, 0, X, self._bins)
        else:
            indices = np.array([np.digitize(X[:, i], self._bins[:, i])
                                for i in range(self._n_features)]).T
        return alphabet[indices]

    def _entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        return scipy.stats.entropy(counts)

    def _entropy_split(self, x, y, split_point):
        left_idx = x <= split_point
        entropy_left = self._entropy(y[left_idx])
        entropy_right = self._entropy(y[~left_idx])
        left_mean = left_idx.mean()
        return left_mean * entropy_left + (1 - left_mean) * entropy_right

    def _information_gain(self, x, y, split_point):
        return self._entropy(y) - self._entropy_split(x, y, split_point)

    def _best_information_gain(self, x, y):
        unique = np.unique(x)
        if unique.size == 1:
            return unique[0], 0
        else:
            unique_mid = (unique[:-1] + unique[1:]) / 2
            information_gain = []
            for split_point in unique_mid:
                information_gain.append(self._information_gain(x, y,
                                                               split_point))
            idxmax = np.argmax(information_gain)
            return unique_mid[idxmax], information_gain[idxmax]

    def _fit_bins(self, x, y):
        bins = []
        for _ in range(self.n_bins - 1):
            if len(bins) == 0:
                sp, ig = self._best_information_gain(x, y)
                bins.append(sp)
            else:
                bin_idx = np.digitize(x, bins)
                sp_list, ig_list = [], []
                for unique in np.unique(bin_idx):
                    idx = bin_idx == unique
                    sp, ig = self._best_information_gain(x[idx], y[idx])
                    sp_list.append(sp)
                    ig_list.append(ig)
                idxmax = np.argmax(ig_list)
                bins.append(sp_list[idxmax])
                bins.sort()
        return bins


class SFA(BaseEstimator, TransformerMixin):
    """Symbolic Fourier Approximation.

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

    n_bins : int (default = 4)
        The number of bins. Ignored if ``quantiles='entropy'``.

    quantiles : {'gaussian', 'empirical', 'entropy'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used. If 'entropy', quantiles are computed using
        the breakpoints leading to the maximum information gain.

    variance_selection : bool (default = False)
        If True, the Fourier coefficients with low variance are removed.

    variance_threshold : float (default = 0.)
        Fourier coefficients with a training-set variance lower than this
        threshold will be removed. Ignored if ``variance_selection=False``.

    """

    def __init__(self, n_coefs=None, anova=True, norm_mean=True,
                 norm_std=True, n_bins=4, quantiles='entropy',
                 variance_selection=False, variance_threshold=0.):
        self.n_coefs = n_coefs
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.variance_selection = variance_selection
        self.variance_threshold = variance_threshold

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
        if not isinstance(self.variance_selection, (int, float)):
            raise ValueError("'variance_selection' must be a boolean.")
        if not isinstance(self.anova, (int, float)):
            raise ValueError("'variance_threshold' must be a float.")

        if self.variance_selection:
            self._pipeline = Pipeline([("dft", DFT(self.n_coefs,
                                                   self.anova,
                                                   self.norm_mean,
                                                   self.norm_std)),
                                       ("slc", VarianceThreshold(
                                        self.variance_threshold)),
                                       ("mcb", MCB(self.n_bins,
                                                   self.quantiles))])
        else:
            self._pipeline = Pipeline([("dft", DFT(self.n_coefs,
                                                   self.anova,
                                                   self.norm_mean,
                                                   self.norm_std)),
                                       ("mcb", MCB(self.n_bins,
                                                   self.quantiles))])
        self._pipeline.fit(X, y)
        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data used to scale along the features axis.

        Returns
        -------
        X_new : array-like, shape [n_samples]

        """
        return self._pipeline.transform(X)
