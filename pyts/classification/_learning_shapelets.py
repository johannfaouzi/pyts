"""Code for Learning Time-Series Shapelets algorithm."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import warnings
from itertools import chain
from math import ceil
from typing import Any, Literal, Self, cast

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import check_array, compute_class_weight
from sklearn.utils.multiclass import (
    # sklearn ships no py.typed marker; pyrefly's project-mode module
    # indexing does not surface this underscore-prefixed helper even though
    # it is a real attribute of this module at runtime (used internally by
    # OneVsOneClassifier/OneVsRestClassifier).
    _ovr_decision_function,  # pyrefly: ignore[missing-module-attribute]
    check_classification_targets,
)
from sklearn.utils.validation import (
    # Same untyped-sklearn module-indexing gap as above.
    _check_sample_weight,  # pyrefly: ignore[missing-module-attribute]
    check_is_fitted,
    check_random_state,
    check_X_y,
)

from pyts._base import UnivariateClassifierMixin
from pyts.utils._utils import _windowed_view

# A ragged collection of shapelets, grouped by length: one 2D array of
# shape (n_shapelets_per_size, length) per shapelet length/scale.
_Shapelets = tuple[npt.NDArray[np.float64], ...]
# One 1D array of the (repeated) shapelet length per shapelet length/scale,
# aligned with ``_Shapelets``.
_Lengths = tuple[npt.NDArray[np.int64], ...]


@njit(
    [
        "float64(float64)",
        "float64[:](float64[:])",
        "float64[:](int64[:])",
        "float64[:,:](float64[:,:])",
    ],
    fastmath=True,
)
def _expit(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Compute the expit (logistic) function."""
    return 1 / (1 + np.exp(-x))


@njit(
    [
        "float64(float64, float64)",
        "float64[:](int64[:], int64[:])",
        "float64[:](int64[:], float64[:])",
    ],
    fastmath=True,
)
def _xlogy(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the x * log(y) function."""
    return x * np.log(y)


@njit(
    [
        "float64(float64[:], float64)",
        "float64(int64[:], float64)",
    ],
    fastmath=True,
)
def _softmin(arr: npt.NDArray[np.float64], alpha: float) -> np.float64:
    """Derive the soft-minimum of an array."""
    maximum = np.max(alpha * arr)
    exp = np.exp(alpha * arr - maximum)
    num = np.sum(arr * exp)
    den = np.sum(exp)
    return num / den


@njit(
    [
        "float64[:](float64[:], float64)",
        "float64[:](int64[:], float64)",
    ],
    fastmath=True,
)
def _softmin_grad(
    arr: npt.NDArray[np.float64], alpha: float
) -> npt.NDArray[np.float64]:
    """Derive the gradient of the softmin function."""
    minimum = _softmin(arr, alpha)
    maximum = np.max(alpha * arr)
    exp = np.exp(alpha * arr - maximum)
    num = exp * (1 + alpha * (arr - minimum))
    den = np.sum(exp)
    return num / den


@njit("float64[:,:](float64[:,:], int64, int64)", fastmath=True)
def _softmax(
    X: npt.NDArray[np.float64], n_samples: int, n_classes: int
) -> npt.NDArray[np.float64]:
    """Derive the softmax of a 2D-array."""
    maximum = np.empty((n_samples, 1))
    for i in prange(n_samples):
        maximum[i, 0] = np.max(X[i])
    exp = np.exp(X - maximum)
    sum_ = np.empty((n_samples, 1))
    for i in prange(n_samples):
        sum_[i, 0] = np.sum(exp[i])
    return exp / sum_


@njit(
    [
        "float64[:](float64[:,:,:], float64[:], float64)",
        "float64[:](int64[:,:,:], int64[:], float64)",
        "float64[:](int64[:,:,:], float64[:], float64)",
        "float64[:](float64[:,:,:], int64[:], float64)",
    ],
    fastmath=True,
)
def _derive_shapelet_distances(
    X: npt.NDArray[np.float64],
    shapelet: npt.NDArray[np.float64],
    alpha: float,
) -> npt.NDArray[np.float64]:
    """Derive the distance between a shapelet and all the time series."""
    n_samples, n_windows, _ = X.shape

    # Derive all squared distances
    mean = np.empty((n_samples, n_windows))
    for i in prange(n_samples):
        for j in prange(n_windows):
            mean[i, j] = np.mean((X[i, j] - shapelet) ** 2)

    # Derive the soft minimum of all the distances
    dist = np.empty(n_samples)
    for i in prange(n_samples):
        dist[i] = _softmin(mean[i], alpha)

    return dist


# Deliberately left without an explicit signature: ``shapelets`` and
# ``lengths`` are Python tuples with one entry per shapelet length/scale,
# so their arity equals ``shapelet_scale``, an unbounded caller-controlled
# hyperparameter (tests alone exercise arities 1 and 2, and the estimator's
# own test suite exercises 1, 3 and 5). Each arity is a distinct numba
# tuple type, so eagerly enumerating every arity used across production
# and tests is impractical here; lazy compilation is the correct choice.
@njit()
def _derive_all_squared_distances(
    X: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
    shapelets: _Shapelets,
    lengths: _Lengths,
    alpha: float,
) -> list[npt.NDArray[np.float64]]:
    """Derive the squared distances between all shapelets and time series."""
    distances = []  # save the distances in a list

    for i in prange(len(lengths)):
        window_size = lengths[i][0]
        X_window = _windowed_view(
            X, n_samples, n_timestamps, window_size, window_step=1
        )
        for j in prange(shapelets[i].shape[0]):
            dist = _derive_shapelet_distances(X_window, shapelets[i][j], alpha)
            distances.append(dist)

    return distances


# Deliberately left without an explicit signature: ``lengths`` is a
# Python tuple with one entry per shapelet length/scale, so its arity
# equals ``shapelet_scale``, an unbounded caller-controlled hyperparameter
# (tests exercise arities 1 and 2). Each arity is a distinct numba tuple
# type, so eagerly enumerating every arity used across production and
# tests is impractical here; lazy compilation is the correct choice.
@njit()
def _reshape_list_shapelets(
    shapelets: npt.NDArray[np.float64], lengths: _Lengths
) -> list[npt.NDArray[np.float64]]:
    """Reshape shapelets from a 1D-array to a list of 2D-arrays."""
    shapelets_reshaped = []
    start = 0
    for length in lengths:
        n_shapelets = length.size
        length_ = length[0]
        end = start + n_shapelets * length_
        shapelets_reshaped.append(shapelets[start:end].reshape(-1, length_))
        start = end
    return shapelets_reshaped


# Deliberately left without an explicit signature: ``shapelets`` and
# ``lengths`` are Python tuples with one entry per shapelet length/scale,
# so their arity equals ``shapelet_scale``, an unbounded caller-controlled
# hyperparameter (tests exercise arities 1 and 2). Each arity is a
# distinct numba tuple type, so eagerly enumerating every arity used
# across production and tests is impractical here; lazy compilation is
# the correct choice.
@njit()
def _reshape_array_shapelets(
    shapelets: _Shapelets, lengths: _Lengths
) -> npt.NDArray[np.float64]:
    """Reshape shapelets from a tuple of 2D-arrays to a 1D-array."""
    lengths_concatenated = np.concatenate(lengths)
    size = np.sum(lengths_concatenated)
    shapelets_reshaped = np.empty(size)
    start = 0
    for i in range(len(shapelets)):
        end = start + np.sum(lengths[i])
        shapelets_reshaped[start:end] = np.ravel(shapelets[i])
        start = end
    return shapelets_reshaped


def _loss(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_classes: int,
    weights: npt.NDArray[np.float64],
    shapelets: _Shapelets,
    lengths: _Lengths,
    alpha: float,
    penalty: Literal['l1', 'l2'],
    C: float,
    fit_intercept: bool,
    intercept_scaling: float,
    sample_weight: npt.NDArray[np.float64],
) -> np.float64:
    """Compute the objective function."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha
    )
    distances = np.asarray(distances).T

    # Add intercept
    if fit_intercept:
        distances = np.c_[np.ones(n_samples) * intercept_scaling, distances]

    # Derive probabilities and cross-entropy loss
    if weights.ndim == 1:
        proba = _expit(distances @ weights)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        # ``sample_weight`` must be 1D here (n_samples,), matching the
        # (n_samples,) per-sample log-loss vector it is multiplied
        # against elementwise: passing a (n_samples, 1)-shaped
        # ``sample_weight`` instead would make NumPy broadcast the two
        # into an (n_samples, n_samples) outer product rather than
        # pairing each sample with its own weight (see the ``fit``
        # methods below, which squeeze ``sample_weight`` to 1D before
        # calling ``_loss``).
        loss_value = -np.mean(
            sample_weight * (_xlogy(y, proba) + _xlogy(1 - y, 1 - proba))
        )
    else:
        proba = _softmax(distances @ weights, n_samples, n_classes)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        loss_value = -np.mean(sample_weight * np.sum(y * np.log(proba), axis=1))

    # Add regularization
    if penalty == 'l2':
        loss_value += (1 / C) * np.square(weights).sum()
    elif penalty == 'l1':
        loss_value += (1 / C) * np.abs(weights).sum()

    return loss_value


def _grad_weights(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_classes: int,
    weights: npt.NDArray[np.float64],
    shapelets: _Shapelets,
    lengths: _Lengths,
    alpha: float,
    penalty: Literal['l1', 'l2'],
    C: float,
    fit_intercept: bool,
    intercept_scaling: float,
    sample_weight: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the gradient of the loss with regards to the weights."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha
    )
    distances = np.asarray(distances).T

    # Add intercept
    if fit_intercept:
        distances = np.c_[np.ones(n_samples) * intercept_scaling, distances]

    # Derive probabilities and binary cross-entropy loss
    if weights.ndim == 1:
        proba = _expit(distances @ weights)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        gradients = ((proba - y)[:, None] * distances * sample_weight).mean(axis=0)
    else:
        proba = _softmax(distances @ weights, n_samples, n_classes)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        # Mathematically equivalent to
        # ``((proba - y)[:, None, :] * (distances * sample_weight)[:, :,
        # None]).mean(axis=0)``, but as a matrix product instead of a
        # broadcasted elementwise product: the original materialized a
        # full (n_samples, n_features, n_classes) intermediate just to
        # immediately reduce it away to (n_features, n_classes). This form
        # lets BLAS compute the same reduction directly, which is both
        # faster and avoids the oversized intermediate.
        gradients = (distances * sample_weight).T @ (proba - y) / n_samples

    if penalty == 'l2':
        gradients += (2 / C) * weights
    elif penalty == 'l1':
        gradients += (1 / C) * np.sign(weights)

    return gradients


# Deliberately left without an explicit signature: ``shapelets`` and
# ``lengths`` are Python tuples with one entry per shapelet length/scale,
# so their arity equals ``shapelet_scale``, an unbounded caller-controlled
# hyperparameter (the estimator's own test suite exercises 1, 3 and 5).
# Each arity is a distinct numba tuple type, so eagerly enumerating every
# arity used across production and tests is impractical here; lazy
# compilation is the correct choice.
@njit()
def _compute_shapelet_grad(
    X: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
    weights: npt.NDArray[np.float64],
    shapelets: _Shapelets,
    lengths: _Lengths,
    alpha: float,
    proba_minus_y: npt.NDArray[np.float64],
    weight_idx: int,
    sample_weight: npt.NDArray[np.float64],
) -> list[npt.NDArray[np.float64]]:

    gradients = []

    for i in range(len(lengths)):
        X_window = _windowed_view(
            X,
            n_samples,
            n_timestamps,
            window_size=lengths[i][0],
            window_step=1,
        )
        n_windows = X_window.shape[1]
        size = shapelets[i][0].size
        for j in range(shapelets[i].shape[0]):
            # Get the current shapelet
            shapelet = shapelets[i][j]

            # Derive the distance without materializing the
            # (n_samples, n_windows, size) ``shapelet - X_window``
            # difference array: each squared difference is accumulated
            # into ``dist`` directly, one timestamp at a time.
            dist = np.empty((n_samples, n_windows))
            for k in prange(n_samples):
                for m in prange(n_windows):
                    total = 0.0
                    for d in range(size):
                        diff_value = shapelet[d] - X_window[k, m, d]
                        total += diff_value * diff_value
                    dist[k, m] = total / size

            # Derive the softmin gradient
            softmin_gradient = np.empty((n_samples, n_windows))
            for k in prange(n_samples):
                softmin_gradient[k] = _softmin_grad(dist[k], alpha)

            # Compute the gradient. Every per-sample factor (the distance
            # gradient's ``row_sum``, ``proba_minus_y``/``class_combined``,
            # and ``sample_weight``) is multiplied together for a given
            # sample ``k`` *before* summing over samples, exactly mirroring
            # the weighted mean in ``_grad_weights``
            # (``((proba - y)[:, None] * distances * sample_weight)
            # .mean(axis=0)``). An earlier version of this function instead
            # factored ``proba_minus_y`` (or ``class_combined``) and
            # ``sample_weight`` out as a single scalar applied to the
            # samples' summed distance-gradient term -- mathematically
            # ``mean(a) * mean(b)`` instead of the correct ``mean(a * b)``
            # -- which silently decoupled each sample's weight from its own
            # gradient contribution whenever ``sample_weight`` was
            # non-uniform (verified against finite differences).
            grad = np.empty(size)
            scale = 2.0 / size
            if weights.ndim == 1:
                for d in prange(size):
                    total = 0.0
                    for k in range(n_samples):
                        row_sum = 0.0
                        for m in range(n_windows):
                            diff_value = (shapelet[d] - X_window[k, m, d]) * scale
                            row_sum += diff_value * softmin_gradient[k, m]
                        total += row_sum * proba_minus_y[k, 0] * sample_weight[k, 0]
                    grad[d] = (total / n_samples) * weights[weight_idx]
            else:
                class_combined = np.empty(n_samples)
                for k in range(n_samples):
                    class_total = 0.0
                    for c in range(weights.shape[1]):
                        class_total += weights[weight_idx, c] * proba_minus_y[k, c]
                    class_combined[k] = class_total
                for d in prange(size):
                    total = 0.0
                    for k in range(n_samples):
                        row_sum = 0.0
                        for m in range(n_windows):
                            diff_value = (shapelet[d] - X_window[k, m, d]) * scale
                            row_sum += diff_value * softmin_gradient[k, m]
                        total += row_sum * class_combined[k] * sample_weight[k, 0]
                    grad[d] = total / n_samples
            gradients.append(grad)

            # Update the weight index
            weight_idx += 1

    return gradients


def _grad_shapelets(
    X: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    n_classes: int,
    weights: npt.NDArray[np.float64],
    shapelets: _Shapelets,
    lengths: _Lengths,
    alpha: float,
    penalty: Literal['l1', 'l2'],
    C: float,
    fit_intercept: bool,
    intercept_scaling: float,
    sample_weight: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute the gradient of the loss with regards to the shapelets."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha
    )
    distances = np.asarray(distances).T

    # Add intercept
    if fit_intercept:
        distances = np.c_[np.ones(n_samples) * intercept_scaling, distances]
        weight_idx = 1
    else:
        weight_idx = 0

    # Derive probabilities and cross-entropy loss
    if weights.ndim == 1:
        proba = _expit(distances @ weights)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
    else:
        proba = _softmax(distances @ weights, n_samples, n_classes)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)

    # Reshape some arrays
    if weights.ndim == 1:
        proba_minus_y = (proba - y)[:, None]
    else:
        proba_minus_y = proba - y

    # Compute the gradients
    gradients = _compute_shapelet_grad(
        X,
        n_samples,
        n_timestamps,
        weights,
        shapelets,
        lengths,
        alpha,
        proba_minus_y,
        weight_idx,
        sample_weight,
    )
    gradients = np.concatenate(gradients)

    return gradients


class CrossEntropyLearningShapelets(BaseEstimator, UnivariateClassifierMixin):
    """Learning Shapelets algorithm with cross-entropy loss.

    Parameters
    ----------
    n_shapelets_per_size : int or float (default = 0.2)
        Number of shapelets per size. If float, it represents
        a fraction of the number of timestamps and the number
        of shapelets per size is equal to
        ``ceil(n_shapelets_per_size * n_timestamps)``.

    min_shapelet_length : int or float (default = 0.1)
        Minimum length of the shapelets. If float, it represents
        a fraction of the number of timestamps and the minimum
        length of the shapelets per size is equal to
        ``ceil(min_shapelet_length * n_timestamps)``.

    shapelet_scale : int (default = 3)
        The different scales for the lengths of the shapelets.
        The lengths of the shapelets are equal to
        ``min_shapelet_length * np.arange(1, shapelet_scale + 1)``.
        The total number of shapelets (and features)
        is equal to ``n_shapelets_per_size * shapelet_scale``.

    penalty : 'l1' or 'l2' (default = 'l2')
        Used to specify the norm used in the penalization.

    tol : float (default = 1e-3)
        Relative tolerance for stopping criterion.

    C : float (default = 1000)
        Inverse of regularization strength. It must be a positive float.
        Smaller values specify stronger regularization.

    learning_rate : float (default = 1.)
        Learning rate for gradient descent optimization. It must be a positive
        float. Note that the learning rate will be automatically decreased
        if the loss function is not decreasing.

    max_iter : int (default = 1000)
        Maximum number of iterations for gradient descent algorithm.

    alpha : float (default = -100)
        Scaling term in the softmin function. The lower, the more precised
        the soft minimum will be. Default value should be good for
        standardized time series.

    fit_intercept : bool (default = True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float (default = 1.)
        Scaling of the intercept. Only used if ``fit_intercept=True``.

    class_weight : dict, None or 'balanced' (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have unit weight.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : int (default = 0)
        Controls the verbosity. It must be a non-negative integer.
        If positive, loss at each iteration is printed.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    shapelets_ : array, shape = (n_shapelets,)
        Learned shapelets.

    coef_ : array, shape = (1, n_shapelets) or (n_classes, n_shapelets)
        Coefficients for each shapelet in the decision function.

    intercept_ : array, shape = (1,) or (n_classes,)
        Intercepts (a.k.a. biases) added to the decision function.
        If ``fit_intercept=False``, the intercepts are set to zero.

    n_iter_ : int
        Actual number of iterations.

    References
    ----------
    .. [1] J. Grabocka, N. Schilling, M. Wistuba and L. Schmidt-Thieme,
           "Learning Time-Series Shapelets". International Conference on Data
           Mining, 14, 392-401 (2014).

    """

    def __init__(
        self,
        n_shapelets_per_size: int | float = 0.2,
        min_shapelet_length: int | float = 0.1,
        shapelet_scale: int = 3,
        penalty: Literal['l1', 'l2'] = 'l2',
        tol: float = 0.001,
        C: float = 1000,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        alpha: float = -100,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: dict[Any, float] | Literal['balanced'] | None = None,
        verbose: int = 0,
        random_state: int | np.random.RandomState | None = None,
    ) -> None:
        self.n_shapelets_per_size = n_shapelets_per_size
        self.min_shapelet_length = min_shapelet_length
        self.shapelet_scale = shapelet_scale
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> Self:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        sample_weight : None or array-like, shape = (n_samples,) (default = None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        check_classification_targets(y)
        le = LabelEncoder().fit(y)
        y_ind = le.transform(y)
        self.classes_ = le.classes_
        n_classes = len(le.classes_)

        (n_shapelets_per_size, min_shapelet_length, sample_weight, rng) = (
            self._check_params(X, y, y_ind, le.classes_, sample_weight)
        )

        if n_classes > 2:
            # LabelBinarizer's return type is inferred from its untyped
            # sklearn source as a broader `ndarray | spmatrix` union (it can
            # return a sparse matrix if ``sparse_output=True``), even though
            # the default ``sparse_output=False`` used here always yields a
            # dense array; np.asarray narrows that back to a plain ndarray.
            y_ind = np.asarray(LabelBinarizer().fit_transform(y))

        # Shapelet initialization
        window_sizes = np.arange(
            min_shapelet_length,
            min_shapelet_length * (self.shapelet_scale + 1),
            min_shapelet_length,
        )

        n_shapelets_per_cluster = n_timestamps - window_sizes + 1
        if np.any(n_shapelets_per_size > n_shapelets_per_cluster):
            raise ValueError(
                "'n_shapelets_per_size' is too high given "
                "'min_shapelet_length' and 'shapelet_scale'."
            )

        shapelets = []
        lengths = []
        for window_size in window_sizes:
            X_window = _windowed_view(
                X, n_samples, n_timestamps, window_size, window_step=1
            )
            X_window = X_window.reshape(-1, window_size)
            kmeans = KMeans(
                n_init=10, n_clusters=n_shapelets_per_size, random_state=rng
            )
            kmeans.fit(X_window)
            shapelets.append(kmeans.cluster_centers_)
            lengths.append(np.full(n_shapelets_per_size, window_size))
        shapelets = tuple(shapelets)
        lengths = tuple(lengths)

        # Weight initialization
        n_shapelets = n_shapelets_per_size * self.shapelet_scale
        if n_classes == 2:
            if self.fit_intercept:
                weights = rng.randn(n_shapelets + 1) / 100
            else:
                weights = rng.randn(n_shapelets) / 100
        else:
            if self.fit_intercept:
                weights = rng.randn(n_shapelets + 1, n_classes) / 100
            else:
                weights = rng.randn(n_shapelets, n_classes) / 100

        # Gradient descent
        learning_rate = self.learning_rate
        losses = []
        iteration = 0
        loss_iteration = _loss(
            X,
            y_ind,
            n_classes,
            weights,
            shapelets,
            lengths,
            self.alpha,
            self.penalty,
            self.C,
            self.fit_intercept,
            self.intercept_scaling,
            # ``_loss`` expects a 1D ``sample_weight`` (unlike
            # ``_grad_weights``/``_grad_shapelets``, which need the
            # (n_samples, 1) shape to broadcast correctly against 2D
            # arrays); see the comment in ``_loss``.
            sample_weight[:, 0],
        )
        if self.verbose:
            print(f'Iteration {0}: loss = {loss_iteration:0.6f}')
        losses.append(loss_iteration)
        for iteration in range(1, self.max_iter + 1):
            # Update weights
            gradient_weights = _grad_weights(
                X,
                y_ind,
                n_classes,
                weights,
                shapelets,
                lengths,
                self.alpha,
                self.penalty,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                sample_weight,
            )
            weights -= learning_rate * gradient_weights

            # Update shapelets
            gradient_shapelets = _grad_shapelets(
                X,
                y_ind,
                n_classes,
                weights,
                shapelets,
                lengths,
                self.alpha,
                self.penalty,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                sample_weight,
            )
            shapelets_array = _reshape_array_shapelets(shapelets, lengths)
            shapelets_array -= learning_rate * gradient_shapelets
            shapelets = tuple(_reshape_list_shapelets(shapelets_array, lengths))

            # Compute current loss
            loss_iteration = _loss(
                X,
                y_ind,
                n_classes,
                weights,
                shapelets,
                lengths,
                self.alpha,
                self.penalty,
                self.C,
                self.fit_intercept,
                self.intercept_scaling,
                sample_weight[:, 0],
            )

            # If loss is increasing, decrease the learning rate
            if losses[-1] < loss_iteration:
                while losses[-1] < loss_iteration:
                    # Go back to previous state
                    weights += learning_rate * gradient_weights
                    shapelets_array = _reshape_array_shapelets(shapelets, lengths)
                    shapelets_array += learning_rate * gradient_shapelets
                    shapelets = tuple(_reshape_list_shapelets(shapelets_array, lengths))

                    # Update learning  rate
                    learning_rate /= 5

                    # Recompute shapelet gradient
                    weights -= learning_rate * gradient_weights
                    gradient_shapelets = _grad_shapelets(
                        X,
                        y_ind,
                        n_classes,
                        weights,
                        shapelets,
                        lengths,
                        self.alpha,
                        self.penalty,
                        self.C,
                        self.fit_intercept,
                        self.intercept_scaling,
                        sample_weight,
                    )
                    shapelets_array = _reshape_array_shapelets(shapelets, lengths)
                    shapelets_array -= learning_rate * gradient_shapelets
                    shapelets = tuple(_reshape_list_shapelets(shapelets_array, lengths))

                    loss_iteration = _loss(
                        X,
                        y_ind,
                        n_classes,
                        weights,
                        shapelets,
                        lengths,
                        self.alpha,
                        self.penalty,
                        self.C,
                        self.fit_intercept,
                        self.intercept_scaling,
                        sample_weight[:, 0],
                    )
            if self.verbose:
                print(f'Iteration {iteration}: loss = {loss_iteration:0.6f}')
            losses.append(loss_iteration)

            # Stopping criterion
            if abs(losses[-2] - losses[-1]) < self.tol * losses[-1]:
                break

        if iteration == self.max_iter:
            warnings.warn(
                'Maximum number of iterations reached without '
                'converging. Increase the maximum number of '
                'iterations.',
                ConvergenceWarning,
                stacklevel=2,
            )

        # Save results in attributes
        self._shapelets = shapelets
        self._lengths = lengths
        self.shapelets_ = [list(shapelet) for shapelet in shapelets]
        self.shapelets_ = np.asarray(
            list(chain.from_iterable(self.shapelets_)), dtype='object'
        )
        if n_classes == 2:
            if self.fit_intercept:
                self.intercept_ = np.array([weights[0]])
                self.coef_ = weights[1:].reshape(1, -1)
            else:
                self.intercept_ = np.array([0])
                self.coef_ = weights.reshape(1, -1)
        else:
            if self.fit_intercept:
                self.intercept_ = weights[0]
                self.coef_ = weights[1:].T
            else:
                self.intercept_ = np.zeros(weights.shape[1])
                self.coef_ = weights.T
        self.n_iter_ = iteration

        return self

    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Decision function scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        T : array-like of shape (n_samples,) or (n_samples, n_classes)
            Decision function scores for each sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.

        """
        check_is_fitted(self, ['shapelets_', 'coef_', 'intercept_', 'n_iter_'])

        X = check_array(X)
        n_samples, n_timestamps = X.shape

        # Derive distances between shapelets and time series
        distances = _derive_all_squared_distances(
            X,
            n_samples,
            n_timestamps,
            self._shapelets,
            self._lengths,
            self.alpha,
        )
        distances = np.asarray(distances).T

        # Add intercept
        if self.fit_intercept:
            distances = np.c_[np.ones(n_samples) * self.intercept_scaling, distances]

        # Derive decision function
        if self.fit_intercept:
            if len(self.classes_) == 2:
                weights = np.r_[self.intercept_, np.squeeze(self.coef_)]
            else:
                weights = np.r_[self.intercept_.reshape(1, -1), self.coef_.T]
        else:
            weights = self.coef_.T
        X_new = np.squeeze(distances @ weights)

        return X_new

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        T : array-like of shape (n_samples, n_classes)
            Probability of the samples for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        """
        X_new = self.decision_function(X)
        n_samples = X_new.shape[0]
        if len(self.classes_) == 2:
            proba = _expit(X_new)
            X_proba = np.c_[1 - proba, proba]
        else:
            X_proba = _softmax(X_new, n_samples, len(self.classes_))
        return X_proba

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Class labels for each data sample.

        """
        if len(self.classes_) == 2:
            y_pred = (self.decision_function(X) > 0.0).astype('int64')
        else:
            y_pred = self.decision_function(X).argmax(axis=1)
        return self.classes_[y_pred]

    def _check_params(
        self,
        X: npt.NDArray[np.float64],
        y: npt.NDArray[Any],
        y_ind: npt.NDArray[np.int64],
        classes: npt.NDArray[Any],
        sample_weight: npt.ArrayLike | None,
    ) -> tuple[int, int, npt.NDArray[np.float64], np.random.RandomState]:
        """Parameter check"""
        _n_samples, n_timestamps = X.shape

        if not isinstance(
            self.n_shapelets_per_size, (int, np.integer, float, np.floating)
        ):
            raise TypeError(
                "'n_shapelets_per_size' must be an integer or a "
                f"float (got {self.n_shapelets_per_size})."
            )
        if isinstance(self.n_shapelets_per_size, (int, np.integer)):
            if not 1 <= self.n_shapelets_per_size <= n_timestamps:
                raise ValueError(
                    "If 'n_shapelets_per_size' is an integer, it must be "
                    "greater than or equal to 1 and lower than or equal to "
                    f"n_timestamps (got {self.n_shapelets_per_size})."
                )
            n_shapelets_per_size = self.n_shapelets_per_size
        else:
            if not (0 < self.n_shapelets_per_size <= 1.0):
                raise ValueError(
                    "If 'n_shapelets_per_size' is a float, it must be "
                    "greater than 0 and lower than or equal to 1 "
                    f"(got {self.n_shapelets_per_size})."
                )
            n_shapelets_per_size = ceil(self.n_shapelets_per_size * n_timestamps)

        if not isinstance(
            self.min_shapelet_length, (int, np.integer, float, np.floating)
        ):
            raise TypeError(
                "'min_shapelet_length' must be an integer or a "
                f"float (got {self.min_shapelet_length})."
            )
        if isinstance(self.min_shapelet_length, (int, np.integer)):
            if not 1 <= self.min_shapelet_length <= n_timestamps:
                raise ValueError(
                    "If 'min_shapelet_length' is an integer, it must be "
                    "greater than or equal to 1 and lower than or equal to "
                    f"n_timestamps (got {self.min_shapelet_length})."
                )
            min_shapelet_length = self.min_shapelet_length
        else:
            if not (0 < self.min_shapelet_length <= 1.0):
                raise ValueError(
                    "If 'min_shapelet_length' is a float, it must be "
                    "greater than 0 and lower than or equal to 1 "
                    f"(got {self.min_shapelet_length})."
                )
            min_shapelet_length = ceil(self.min_shapelet_length * n_timestamps)

        if not (
            isinstance(self.shapelet_scale, (int, np.integer))
            and self.shapelet_scale > 0
        ):
            raise ValueError(
                "'shapelet_scale' must be a positive integer "
                f"(got {self.shapelet_scale})."
            )

        if self.shapelet_scale * min_shapelet_length > n_timestamps:
            raise ValueError(
                "'shapelet_scale' and 'min_shapelet_length' must be "
                "such that shapelet_scale * min_shapelet_length is "
                "smaller than or equal to n_timestamps."
            )

        if self.penalty not in ('l1', 'l2'):
            raise ValueError(
                f"'penalty' must be either 'l2' or 'l1' (got {self.penalty})."
            )

        if not (
            isinstance(self.C, (int, np.integer, float, np.floating)) and self.C > 0
        ):
            raise ValueError(f"'C' must be a positive float (got {self.C}).")

        if not (
            isinstance(self.tol, (int, np.integer, float, np.floating)) and self.tol > 0
        ):
            raise ValueError(f"'tol' must be a positive float (got {self.tol}).")

        if not (
            isinstance(self.learning_rate, (int, np.integer, float, np.floating))
            and self.learning_rate > 0
        ):
            raise ValueError(
                f"'learning_rate' must be a positive float (got {self.learning_rate})."
            )

        if not (isinstance(self.max_iter, (int, np.integer)) and self.max_iter >= 0):
            raise ValueError(
                f"'max_iter' must be a non-negative integer (got {self.max_iter})."
            )

        if not (
            isinstance(self.alpha, (int, np.integer, float, np.floating))
            and self.alpha < 0
        ):
            raise ValueError(f"'alpha' must be a negative float (got {self.alpha}).")

        if not isinstance(
            self.intercept_scaling, (int, np.integer, float, np.floating)
        ):
            raise ValueError(
                f"'intercept_scaling' must be a float (got {self.intercept_scaling})."
            )

        class_weight_balanced = (
            isinstance(self.class_weight, str) and self.class_weight == 'balanced'
        )
        if not (
            self.class_weight is None
            or class_weight_balanced
            or isinstance(self.class_weight, dict)
        ):
            raise ValueError(
                "'class_weight' must be None, a dictionary "
                f" or 'balanced' (got {self.class_weight})."
            )
        class_weight = compute_class_weight(self.class_weight, classes=classes, y=y)

        sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float64)
        sample_weight *= class_weight[y_ind]
        sample_weight = sample_weight.reshape(-1, 1)

        rng = check_random_state(self.random_state)

        if not (isinstance(self.verbose, (int, np.integer)) and self.verbose >= 0):
            raise ValueError(
                f"'verbose' must be a non-negative integer (got {self.verbose})."
            )

        return n_shapelets_per_size, min_shapelet_length, sample_weight, rng


class LearningShapelets(BaseEstimator, UnivariateClassifierMixin):
    """Learning Shapelets algorithm.

    This estimator consists of two steps: computing the distances between the
    shapelets and the time series, then computing a logistic regression using
    these distances as features. This algorithm learns the shapelets as well as
    the coefficients of the logistic regression.

    Parameters
    ----------
    n_shapelets_per_size : int or float (default = 0.2)
        Number of shapelets per size. If float, it represents
        a fraction of the number of timestamps and the number
        of shapelets per size is equal to
        ``ceil(n_shapelets_per_size * n_timestamps)``.

    min_shapelet_length : int or float (default = 0.1)
        Minimum length of the shapelets. If float, it represents
        a fraction of the number of timestamps and the minimum
        length of the shapelets per size is equal to
        ``ceil(min_shapelet_length * n_timestamps)``.

    shapelet_scale : int (default = 3)
        The different scales for the lengths of the shapelets.
        The lengths of the shapelets are equal to
        ``min_shapelet_length * np.arange(1, shapelet_scale + 1)``.
        The total number of shapelets (and features)
        is equal to ``n_shapelets_per_size * shapelet_scale``.

    penalty : 'l1' or 'l2' (default = 'l2')
        Used to specify the norm used in the penalization.

    tol : float (default = 1e-3)
        Tolerance for stopping criterion.

    C : float (default = 1000)
        Inverse of regularization strength. It must be a positive float.
        Smaller values specify stronger regularization.

    learning_rate : float (default = 1.)
        Learning rate for gradient descent optimization. It must be a positive
        float. Note that the learning rate will be automatically decreased
        if the loss function is not decreasing.

    max_iter : int (default = 1000)
        Maximum number of iterations for gradient descent algorithm.

    multi_class : {'multinomial', 'ovr', 'ovo'} (default = 'multinomial')
        Strategy for multiclass classification.
        'multinomial' stands for multinomial cross-entropy loss.
        'ovr' stands for one-vs-rest strategy.
        'ovo' stands for one-vs-one strategy.
        Ignored if the classification task is binary.

    alpha : float (default = -100)
        Scaling term in the softmin function. The lower, the more precised
        the soft minimum will be. Default value should be good for
        standardized time series.

    fit_intercept : bool (default = True)
        Specifies if a constant (a.k.a. bias or intercept) should be
        added to the decision function.

    intercept_scaling : float (default = 1.)
        Scaling of the intercept. Only used if ``fit_intercept=True``.

    class_weight : dict, None or 'balanced' (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have unit weight.
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``.

    verbose : int (default = 0)
        Controls the verbosity. It must be a non-negative integer.
        If positive, loss at each iteration is printed.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    n_jobs : None or int (default = None)
        The number of jobs to use for the computation. Only used if
        ``multi_class`` is 'ovr' or 'ovo'.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    shapelets_ : array shape = (n_tasks, n_shapelets)
        Learned shapelets. Each element of this array is a learned
        shapelet.

    coef_ : array, shape = (n_tasks, n_shapelets) or (n_classes, n_shapelets)
        Coefficients for each shapelet in the decision function.

    intercept_ : array, shape = (n_tasks,) or (n_classes,)
        Intercepts (a.k.a. biases) added to the decision function.
        If ``fit_intercept=False``, the intercepts are set to zero.

    n_iter_ : array, shape = (n_tasks,)
        Actual number of iterations.

    Notes
    -----
    The number of tasks (n_tasks) depends on the value of ``multi_class``
    and the number of classes. If there are two classes, the number of
    tasks is equal to 1. If there are more than two classes, the number
    of tasks is equal to:

        - 1 if ``multi_class='multinomial'``
        - n_classes if ``multi_class='ovr'``
        - n_classes * (n_classes - 1) / 2 if ``multi_class='ovo'``

    References
    ----------
    .. [1] J. Grabocka, N. Schilling, M. Wistuba and L. Schmidt-Thieme,
           "Learning Time-Series Shapelets". International Conference on Data
           Mining, 14, 392-401 (2014).

    Examples
    --------
    >>> from pyts.classification import LearningShapelets
    >>> X = [[1, 2, 2, 1, 2, 3, 2],
    ...      [0, 2, 0, 2, 0, 2, 3],
    ...      [0, 1, 2, 2, 1, 2, 2]]
    >>> y = [0, 1, 0]
    >>> clf = LearningShapelets(random_state=42, tol=0.01)
    >>> clf.fit(X, y)
    LearningShapelets(...)
    >>> clf.coef_.shape
    (1, 6)

    """

    def __init__(
        self,
        n_shapelets_per_size: int | float = 0.2,
        min_shapelet_length: int | float = 0.1,
        shapelet_scale: int = 3,
        penalty: Literal['l1', 'l2'] = 'l2',
        tol: float = 0.001,
        C: float = 1000,
        learning_rate: float = 1.0,
        max_iter: int = 1000,
        multi_class: Literal['multinomial', 'ovr', 'ovo'] = 'multinomial',
        alpha: float = -100,
        fit_intercept: bool = True,
        intercept_scaling: float = 1.0,
        class_weight: dict[Any, float] | Literal['balanced'] | None = None,
        verbose: int = 0,
        random_state: int | np.random.RandomState | None = None,
        n_jobs: int | None = None,
    ) -> None:
        self.n_shapelets_per_size = n_shapelets_per_size
        self.min_shapelet_length = min_shapelet_length
        self.shapelet_scale = shapelet_scale
        self.penalty = penalty
        self.tol = tol
        self.C = C
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.multi_class = multi_class
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.intercept_scaling = intercept_scaling
        self.class_weight = class_weight
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: npt.ArrayLike | None = None,
    ) -> Self:
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        sample_weight : None or array-like, shape = (n_samples,) (default = None)
            Array of weights that are assigned to individual samples.
            If not provided, then each sample is given unit weight.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(X, y)
        n_classes = len(LabelEncoder().fit(y).classes_)
        multi_class = self._check_params(n_classes)
        params = self.get_params()
        params.pop('n_jobs', None)
        params.pop('multi_class')

        self._multi_class = multi_class

        if multi_class in ('ovr', 'ovo'):
            base_clf = CrossEntropyLearningShapelets(**params)
            multi_clf: OneVsRestClassifier | OneVsOneClassifier
            if multi_class == 'ovr':
                multi_clf = OneVsRestClassifier(base_clf, n_jobs=self.n_jobs)
            else:
                multi_clf = OneVsOneClassifier(base_clf, n_jobs=self.n_jobs)
            multi_clf.fit(X, y)

            self.classes_ = multi_clf.classes_
            # ``estimators_`` is typed by pyrefly's source inference as a
            # list of the generic sklearn estimator base class, but at
            # runtime each element is a fitted clone of the
            # ``CrossEntropyLearningShapelets`` instance passed as
            # ``estimator`` above, which does define these shapelet-specific
            # attributes.
            estimators = cast(
                list[CrossEntropyLearningShapelets], multi_clf.estimators_
            )
            self._estimators = estimators
            self.shapelets_ = np.array([est.shapelets_ for est in estimators])
            self.coef_ = np.squeeze(np.asarray([est.coef_ for est in estimators]))
            self.intercept_ = np.squeeze(
                np.array([est.intercept_ for est in estimators])
            )
            self.n_iter_ = np.array([est.n_iter_ for est in estimators])
        else:
            clf = CrossEntropyLearningShapelets(**params)
            clf.fit(X, y)

            self.classes_ = clf.classes_
            self._clf = clf
            self.shapelets_ = np.array([clf.shapelets_])
            self.coef_ = clf.coef_
            self.intercept_ = clf.intercept_
            self.n_iter_ = np.array([clf.n_iter_])
        return self

    def decision_function(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Decision function scores.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        T : array, shape = (n_samples,) or (n_samples, n_classes)
            Decision function scores for each sample for each class in the
            model, where classes are ordered as they are in ``self.classes_``.

        """
        X = check_array(X)

        if self._multi_class == 'ovr':
            X_new = np.empty((X.shape[0], self.classes_.size))
            for i, estimator in enumerate(self._estimators):
                X_new[:, i] = estimator.decision_function(X)
        elif self._multi_class == 'ovo':
            predictions = np.vstack([est.predict(X) for est in self._estimators]).T
            confidences = np.vstack(
                [est.decision_function(X) for est in self._estimators]
            ).T
            X_new = _ovr_decision_function(predictions, confidences, len(self.classes_))
        else:
            X_new = self._clf.decision_function(X)

        return X_new

    def predict_proba(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Probability estimates.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        T : array, shape = (n_samples, n_classes)
            Probability of the samples for each class in the model,
            where classes are ordered as they are in ``self.classes_``.

        """
        X_new = self.decision_function(X)
        n_samples = X_new.shape[0]
        if self._multi_class == 'binary':
            proba = _expit(X_new)
            return np.c_[1 - proba, proba]
        elif self._multi_class == 'multinomial':
            proba = _softmax(X_new, n_samples, len(self.classes_))
        else:
            # OvR normalization, like LibLinear's predict_probability
            proba = _expit(X_new)
            proba /= proba.sum(axis=1).reshape((proba.shape[0], -1))
        return proba

    def predict(self, X: npt.ArrayLike) -> npt.NDArray[Any]:
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Class labels for each data sample.

        """
        X_new = self.decision_function(X)
        if X_new.ndim == 2:
            y_pred = X_new.argmax(axis=1)
        else:
            y_pred = (X_new > 0).astype('int64')
        return self.classes_[y_pred]

    def _check_params(
        self, n_classes: int
    ) -> Literal['binary', 'multinomial', 'ovr', 'ovo']:
        if self.multi_class not in ('multinomial', 'ovr', 'ovo'):
            raise ValueError(
                "'multi_class' must be either 'multinomial', "
                f"'ovr' or 'ovo' (got {self.multi_class})."
            )
        multi_class = 'binary' if n_classes == 2 else self.multi_class

        class_weight_dict = isinstance(self.class_weight, dict)
        if multi_class in ('ovr', 'ovo') and class_weight_dict:
            raise ValueError(
                "'class_weight' must be None or 'balanced' if "
                "'multi_class' is either 'ovr' or 'ovo'."
            )

        n_jobs_int = isinstance(self.n_jobs, (int, np.integer)) and self.n_jobs != 0
        if not (self.n_jobs is None or n_jobs_int):
            raise ValueError(
                "'n_jobs' must be None or an integer not equal "
                f"to zero (got {self.n_jobs})."
            )

        return multi_class
