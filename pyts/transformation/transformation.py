from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
import numpy as np
import scipy.stats
from sklearn.base import BaseEstimator
from pyts.utils import paa, sax, vsm, gaf, mtf


class StandardScaler(BaseEstimator):
    """
    Row-wise Standard Scaler (zero empirical mean, unit empirical variance).

    Parameters
    ----------
    epsilon : float (default = 1e-3)
        value added to empirical variance before dividing.

    """

    def __init__(self, epsilon=1e-3):
        self.epsilon = epsilon

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Check parameters
        if not isinstance(self.epsilon, (int, float)):
            raise TypeError("'epsilon' must be a float or an int.")
        if self.epsilon < 0:
            raise ValueError("'epsilon' must be greater or equal than 0.")

        return ((X.T - X.mean(axis=1)) / (X.std(axis=1) + self.epsilon)).T


class PAA(BaseEstimator):
    """
    Piecewise Aggregate Approximation.

    Parameters
    ----------
    window_size : int or None (default = None)
        size of the sliding window

    output_size : int or None (default = None)
        size of the returned time series

    overlapping : bool (default = True)
        when output_size is specified, the window_size is fixed
        if overlapping is True and may vary if overlapping is False.
        Ignored if window_size is specified.

    """

    def __init__(self, window_size=None, output_size=None, overlapping=True):

        self.window_size = window_size
        self.output_size = output_size
        self.overlapping = overlapping

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, new_n_features]
            Transformed data. If output_size is not None,
            new_n_features is equal to output_size. Otherwise,
            new_n_features is equal to ceil(n_features // window_size).
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters and compute window_size if output_size is given
        if (self.window_size is None and self.output_size is None):
            raise ValueError("'window_size' xor 'output_size' must be specified.")
        elif (self.window_size is not None and self.output_size is not None):
            raise ValueError("'window_size' xor 'output_size' must be specified.")
        elif (self.window_size is not None and self.output_size is None):
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.window_size, int):
                raise TypeError("'window_size' must be an integer.")
            if self.window_size < 1:
                raise ValueError("'window_size' must be greater or equal than 1.")
            if self.window_size > n_features:
                raise ValueError("'window_size' must be lower or equal than the size of each time series.")
            window_size = self.window_size
        else:
            if not isinstance(self.overlapping, (float, int)):
                raise TypeError("'overlapping' must be a boolean.")
            if not isinstance(self.output_size, int):
                raise TypeError("'output_size' must be an integer.")
            if self.output_size < 1:
                raise ValueError("'output_size' must be greater or equal than 1.")
            if self.output_size > n_features:
                raise ValueError("'output_size' must be lower or equal than the size of each time series.")
            window_size = n_features // self.output_size
            window_size += 0 if n_features % self.output_size == 0 else 1

        return np.apply_along_axis(paa, 1, X, n_features, window_size, self.overlapping, self.output_size)


class SAX(BaseEstimator):
    """
    Symbolic Aggregate approXimation.

    Parameters
    ----------
    n_bins : int (default = 8)
        number of bins (also known as the size of the alphabet)

    quantiles : str (default = 'gaussian')
        the way to compute quantiles. Possible values:

            - 'gaussian' : quantiles from a gaussian distribution N(0,1)
            - 'empirical' : empirical quantiles
    """

    def __init__(self, n_bins=8, quantiles='gaussian'):

        self.n_bins = n_bins
        self.quantiles = quantiles

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, n_features]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater or equal than 2.")
        if self.n_bins > 52:
            raise ValueError("'n_bins' must be lower or equal than 52.")
        if self.quantiles not in ['gaussian', 'empirical']:
            raise ValueError("'quantiles' must be either 'gaussian' or 'empirical'.")

        # Alphabet
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"

        # Compute gaussian quantiles if quantiles == 'gaussian'
        if self.quantiles == 'gaussian':
            quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, num=self.n_bins + 1)[1:])
        else:
            quantiles = self.quantiles

        return np.apply_along_axis(sax, 1, X, self.n_bins, quantiles, alphabet)


class VSM(BaseEstimator):
    """
    Vector Space Model.

    Parameters
    ----------
    window_size : int (default = 4)
        size of the window (size of each word)

    numerosity_reduction : bool (default = True)
        if True, deletes all but one occurence of back to back
        identical occurences of the same word
    """

    def __init__(self, window_size=4, numerosity_reduction=True):

        self.window_size = window_size
        self.numerosity_reduction = numerosity_reduction

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples]

        Returns
        -------
        X_new : list, size = [n_samples]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 1):
            raise ValueError("'X' must be a 1-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.size, len(X[0])

        # Check parameters
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size < 1:
            raise ValueError("'window_size' must be greater or equal than 1.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower or equal than the size of each time series.")
        if not isinstance(self.numerosity_reduction, (float, int)):
            raise TypeError("'numerosity_reduction' must be a boolean.")

        return np.array([vsm(X[i], n_features, self.window_size, self.numerosity_reduction) for i in range(n_samples)])


class GASF(BaseEstimator):
    """
    Gramian Angular Summation Field.

    Parameters
    ----------
    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    overlapping : bool (default = False)
        if True, reducing the size of the time series with PAA is
        done with possible overlapping windows.

    scale : str (default = '-1')
        the lower bound of the scaled time series. Possible values:

            - '-1' : the time series are scaled in [-1,1]
            - '0' : the time series are scaled in [0,1]
    """

    def __init__(self, image_size=32, overlapping=False, scale='-1'):

        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'image_size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater or equal than 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower or equal than the size of each time series.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if self.scale not in ['0', '-1']:
            raise ValueError("'scale' must be either '0' or '-1'.")

        return np.apply_along_axis(gaf, 1, X, n_features, self.image_size,
                                   self.overlapping, 's', self.scale)


class GADF(BaseEstimator):
    """
    Gramian Angular Difference Field.

    Parameters
    ----------
    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    overlapping : bool (default = False)
        if True, reducing the size of the time series with PAA is
        done with possible overlapping windows.

    scale : str (default = '-1')
        the lower bound of the scaled time series. Possible values:

            - '-1' : the time series are scaled in [-1,1]
            - '0' : the time series are scaled in [0,1]
    """

    def __init__(self, image_size=32, overlapping=False, scale='-1'):

        self.image_size = image_size
        self.overlapping = overlapping
        self.scale = scale

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'image_size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater or equal than 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower or equal than the size of each time series.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")
        if self.scale not in ['0', '-1']:
            raise ValueError("'scale' must be either '0' or '-1'.")

        return np.apply_along_axis(gaf, 1, X, n_features, self.image_size,
                                   self.overlapping, 'd', self.scale)


class MTF(BaseEstimator):
    """
    Markov Transition Field.

    Parameters
    ----------
    image_size : int (default = 32)
        determines the shapes of the output image :
        image_size x image_size

    n_bins : int (default = 8)
        number of bins (also known as the size of the alphabet)

    quantiles : str (default = 'gaussian')
        the way to compute quantiles. Possible values:

            - 'gaussian' : quantiles from a gaussian distribution N(0,1)
            - 'empirical' : empirical quantiles

    overlapping : bool (default = False)
        if False, reducing the image with the blurring kernel
        will be applied on non-overlapping rectangles; if True,
        it will be applied on possible overlapping squares.
    """

    def __init__(self, image_size=32, n_bins=8, quantiles='empirical', overlapping=False):

        self.image_size = image_size
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.overlapping = overlapping

    def fit(self, X, y=None):
        pass

    def fit_transform(self, X, y=None):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        return self.transform(X)

    def transform(self, X):
        """Trasnform the provided data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        X_new : np.ndarray, shape = [n_samples, image_size, image_size]
            Transformed data.
        """

        # Check input data
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("'X' must be a 2-dimensional np.ndarray.")

        # Shape parameters
        n_samples, n_features = X.shape

        # Check parameters
        if not isinstance(self.image_size, int):
            raise TypeError("'size' must be an integer.")
        if self.image_size < 2:
            raise ValueError("'image_size' must be greater or equal than 2.")
        if self.image_size > n_features:
            raise ValueError("'image_size' must be lower or equal than the size of each time series.")
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater or equal than 2.")
        if self.quantiles not in ['gaussian', 'empirical']:
            raise ValueError("'quantiles' must be either 'gaussian' or 'empirical'.")
        if not isinstance(self.overlapping, (float, int)):
            raise TypeError("'overlapping' must be a boolean.")

        # Compute gaussian quantiles if quantiles == 'gaussian'
        if self.quantiles == 'gaussian':
            quantiles = scipy.stats.norm.ppf(np.linspace(0, 1, num=self.n_bins)[1:])
        else:
            quantiles = self.quantiles

        return np.apply_along_axis(mtf, 1, X, n_features, self.image_size,
                                   self.n_bins, quantiles, self.overlapping)
