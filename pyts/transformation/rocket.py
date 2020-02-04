from numba import njit, prange
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state)


@njit()
def generate_kernels(n_kernels, n_timestamps, kernel_sizes, seed):
    """Generate the kernels.

    Parameters
    ----------
    n_kernels : int
        Number of kernels

    n_timestamps : int
        Number of timestamps

    kernel_sizes : array
        Possible sizes for the kernels.

    seed : int
        Seed for the random number generator.

    Returns
    -------
    weights : array, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    lengths : array, shape = (n_kernels,)
        Length of each kernel.

    biases : array, shape = (n_kernels,)
        Bias of each kernel.

    dilations : array, shape = (n_kernels,)
        Dilation of each kernel.

    paddings : array, shape = (n_kernels,)
        Padding of each kernel.

    """
    # Fix the random see
    np.random.seed(seed)

    # Lengths of the kernels
    lengths = np.random.choice(kernel_sizes, size=n_kernels)

    # Weights of the kernels
    cumsum_lengths = np.concatenate((np.array([0]), np.cumsum(lengths)))
    weights_all = np.random.randn(cumsum_lengths[-1])
    weights = np.zeros((n_kernels, np.int64(np.max(kernel_sizes))))
    for i in prange(n_kernels):
        weights[i, :lengths[i]] = (
            weights_all[cumsum_lengths[i]: cumsum_lengths[i+1]] -
            np.mean(weights_all[cumsum_lengths[i]: cumsum_lengths[i+1]])
        )

    # Biases
    biases = np.random.uniform(-1, 1, size=n_kernels)

    # Dilations
    upper_bounds = np.log2(np.floor_divide(n_timestamps - 1, lengths - 1))
    powers = np.empty(n_kernels)
    for i in prange(n_kernels):
        powers[i] = np.random.uniform(0, upper_bounds[i])
    dilations = np.floor(np.power(2, powers))

    # Paddings
    paddings = np.zeros(n_kernels)
    padding_cond = np.random.randint(0, 2, n_kernels).astype(np.bool_)
    paddings[padding_cond] = np.floor_divide(
        (lengths - 1) * dilations, 2)[padding_cond]

    return weights, lengths, biases, dilations, paddings


@njit(fastmath=True)
def apply_one_kernel_one_sample(
    x, n_timestamps, weight, length, bias, dilation, padding
):
    """Apply one kernel to one time series.

    Parameters
    ----------
    x : array, shape = (n_timestamps,)
        One time series.

    n_timestamps : int
        Number of timestamps.

    weights : array, shape = (length,)
        Weights of the kernel. Zero padding values are added.

    length : int
        Length of the kernel.

    bias : int
        Bias of the kernel.

    dilation : int
        Dilation of the kernel.

    padding : int
        Padding of the kernel.

    Returns
    -------
    x_new : array, shape = (2,)
        Extracted features using the kernel.

    """
    # Compute padded x
    n_conv = n_timestamps - ((length - 1) * dilation) + (2 * padding)
    if padding > 0:
        x_pad = np.zeros(n_timestamps + 2 * padding)
        x_pad[padding:-padding] = x
    else:
        x_pad = x

    # Compute the convolutions
    x_conv = np.zeros(n_conv)
    for i in prange(n_conv):
        for j in prange(length):
            x_conv[i] += weight[j] * x_pad[i + (j * dilation)]
    x_conv += bias

    # Return the features: maximum and proportion of positive values
    return np.max(x_conv), np.mean(x_conv > 0)


@njit(parallel=True)
def apply_all_kernels(X, weights, lengths, biases, dilations, paddings):
    """Apply one kernel to a data set of time series.

    Parameters
    ----------
    X : array, shape = (n_samples, n_timestamps)
        Input data.

    weights : array, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    lengths : array, shape = (n_kernels,)
        Length of each kernel.

    biases : array, shape = (n_kernels,)
        Bias of each kernel.

    dilations : array, shape = (n_kernels,)
        Dilation of each kernel.

    paddings : array, shape = (n_kernels,)
        Padding of each kernel.

    Returns
    -------
    X_new : array, shape = (n_samples, 2 * n_kernels)
        Extracted features using all the kernels.

    """
    n_samples, n_timestamps = X.shape
    n_kernels = lengths.size
    X_new = np.empty((n_samples, 2 * n_kernels))
    for i in prange(n_samples):
        for j in prange(n_kernels):
            X_new[i, (2 * j):(2 * j + 2)] = apply_one_kernel_one_sample(
                X[i], n_timestamps, weights[j], lengths[j],
                biases[j], dilations[j], paddings[j]
            )
    return X_new


class ROCKET(BaseEstimator, TransformerMixin):
    """RandOm Convolutional KErnel Transformation.

    This algorithm randomly generates a great variety of convolutional kernels
    and extracts two features for each convolution: the maximum and the
    proportion of positive values.

    Parameters
    ----------
    n_kernels : int (default = 10000)
        Number of kernels.

    kernel_sizes : array-like (default = (7, 9, 11))
        The possible sizes of the kernels.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    weights_ : array, shape = (n_kernels, max(kernel_sizes))
        Weights of the kernels. Zero padding values are added.

    length_ : array, shape = (n_kernels,)
        Length of each kernel.

    bias_ : array, shape = (n_kernels,)
        Bias of each kernel.

    dilation_ : array, shape = (n_kernels,)
        Dilation of each kernel.

    padding_ : array, shape = (n_kernels,)
        Padding of each kernel.

    References
    ----------
    .. [1] A. Dempster, F. Petitjean and G. I. Webb, "ROCKET: Exceptionally
           fast and accurate time series classification using random
           convolutional kernels". https://arxiv.org/abs/1910.13051.

    Examples
    --------
    >>> from pyts.transformation import ROCKET
    >>> X = np.arange(100).reshape(5, 20)
    >>> rocket = ROCKET(n_kernels=10)
    >>> rocket.fit_transform(X).shape
    (5, 20)

    """
    def __init__(self, n_kernels=10000, kernel_sizes=(7, 9, 11),
                 random_state=None):
        self.n_kernels = n_kernels
        self.kernel_sizes = kernel_sizes
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : None or array-like, shape = (n_samples,)
            Class labels for each data sample. Ignored.

        Returns
        -------
        self : object

        """
        X = check_array(X, dtype='float64')
        n_samples, n_timestamps = X.shape

        kernel_sizes, seed = self._check_params(n_timestamps)

        # Generate the kernels
        weights, lengths, biases, dilations, paddings = generate_kernels(
            self.n_kernels, n_timestamps, kernel_sizes, seed)

        self.weights_ = weights
        self.length_ = lengths
        self.bias_ = biases
        self.dilation_ = dilations.astype('int64')
        self.padding_ = paddings.astype('int64')

        return self

    def transform(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X_new : array, shape = (n_samples, 2 * n_kernels)
            Extracted features from the kernels.

        """
        X = check_array(X, dtype='float64')
        check_is_fitted(self, ['weights_', 'length_', 'bias_',
                               'dilation_', 'padding_'])
        X_new = apply_all_kernels(
            X, self.weights_, self.length_, self.bias_,
            self.dilation_, self.padding_
        )
        return X_new

    def _check_params(self, n_timestamps):
        if not isinstance(self.n_kernels, (int, np.integer)):
            raise TypeError("'n_kernels' must be an integer (got {})."
                            .format(self.n_kernels))

        if not isinstance(self.kernel_sizes, (list, tuple, np.ndarray)):
            raise TypeError("'kernel_sizes' must be a list, a tuple or "
                            "an array (got {}).".format(self.kernel_sizes))
        kernel_sizes = check_array(self.kernel_sizes, ensure_2d=False,
                                   dtype='int64', accept_large_sparse=False)
        if not np.all(1 <= kernel_sizes):
            raise ValueError("All the values in 'kernel_sizes' must be "
                             "greater than or equal to 1 ({} < 1)."
                             .format(kernel_sizes.min()))
        if not np.all(kernel_sizes <= n_timestamps):
            raise ValueError("All the values in 'kernel_sizes' must be lower "
                             "than or equal to 'n_timestamps' ({} > {})."
                             .format(kernel_sizes.max(), n_timestamps))

        rng = check_random_state(self.random_state)
        seed = rng.randint(np.iinfo(np.uint32).max, dtype='u8')

        return kernel_sizes, seed
