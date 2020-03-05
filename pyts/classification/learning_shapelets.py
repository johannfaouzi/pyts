"""Code for Learning Time-Series Shapelets algorithm."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from itertools import chain
from math import ceil

from numba import njit, prange
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.multiclass import _ovr_decision_function
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.utils import compute_class_weight, check_array
from sklearn.utils.validation import (
    check_is_fitted, check_random_state, check_X_y, _check_sample_weight)
from sklearn.utils.multiclass import check_classification_targets

import warnings

from ..utils.utils import _windowed_view


@njit(fastmath=True)
def _expit(x):
    """Compute the expit (logistic) function."""
    return 1 / (1 + np.exp(-x))


@njit(fastmath=True)
def _xlogy(x, y):
    """Compute the x * log(y) function."""
    return x * np.log(y)


@njit(fastmath=True)
def _softmin(arr, alpha):
    """Derive the soft-minimum of an array."""
    maximum = np.max(alpha * arr)
    exp = np.exp(alpha * arr - maximum)
    num = np.sum(arr * exp)
    den = np.sum(exp)
    return num / den


@njit(fastmath=True)
def _softmin_grad(arr, alpha):
    """Derive the gradient of the softmin function."""
    minimum = _softmin(arr, alpha)
    maximum = np.max(alpha * arr)
    exp = np.exp(alpha * arr - maximum)
    num = exp * (1 + alpha * (arr - minimum))
    den = np.sum(exp)
    return num / den


@njit(fastmath=True)
def _softmax(X, n_samples, n_classes):
    """Derive the softmax of a 2D-array."""
    maximum = np.empty((n_samples, 1))
    for i in prange(n_samples):
        maximum[i, 0] = np.max(X[i])
    exp = np.exp(X - maximum)
    sum_ = np.empty((n_samples, 1))
    for i in prange(n_samples):
        sum_[i, 0] = np.sum(exp[i])
    return exp / sum_


@njit(fastmath=True)
def _derive_shapelet_distances(X, shapelet, alpha):
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


@njit()
def _derive_all_squared_distances(
    X, n_samples, n_timestamps, shapelets, lengths, alpha
):
    """Derive the squared distances between all shapelets and time series."""
    distances = []  # save the distances in a list

    for i in prange(len(lengths)):
        window_size = lengths[i][0]
        X_window = _windowed_view(X, n_samples, n_timestamps,
                                  window_size, window_step=1)
        for j in prange(shapelets[i].shape[0]):
            dist = _derive_shapelet_distances(X_window, shapelets[i][j], alpha)
            distances.append(dist)

    return distances


@njit()
def _reshape_list_shapelets(shapelets, lengths):
    """Reshape shapelets from a 1D-array to a list of 2D-arrays."""
    shapelets_reshaped = []
    start = 0
    for length in lengths:
        n_shapelets = length.size
        length_ = length[0]
        end = start + n_shapelets * length_
        shapelets_reshaped.append(shapelets[start: end].reshape(-1, length_))
        start = end
    return shapelets_reshaped


@njit()
def _reshape_array_shapelets(shapelets, lengths):
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


def _loss(X, y, n_classes, weights, shapelets, lengths, alpha, penalty, C,
          fit_intercept, intercept_scaling, sample_weight):
    """Compute the objective function."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha)
    distances = np.asarray(distances).T

    # Add intercept
    if fit_intercept:
        distances = np.c_[np.ones(n_samples) * intercept_scaling, distances]

    # Derive probabilities and cross-entropy loss
    if weights.ndim == 1:
        proba = _expit(distances @ weights)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        loss_value = - np.mean(
            sample_weight * (_xlogy(y, proba) + _xlogy(1 - y, 1 - proba)))
    else:
        proba = _softmax(distances @ weights, n_samples, n_classes)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        loss_value = - np.mean(
            sample_weight * np.sum(y * np.log(proba), axis=1))

    # Add regularization
    if penalty == 'l2':
        loss_value += (1 / C) * np.square(weights).sum()
    elif penalty == 'l1':
        loss_value += (1 / C) * np.abs(weights).sum()

    return loss_value


def _grad_weights(X, y, n_classes, weights, shapelets, lengths, alpha, penalty,
                  C, fit_intercept, intercept_scaling, sample_weight):
    """Compute the gradient of the loss with regards to the weights."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha)
    distances = np.asarray(distances).T

    # Add intercept
    if fit_intercept:
        distances = np.c_[np.ones(n_samples) * intercept_scaling, distances]

    # Derive probabilities and binary cross-entropy loss
    if weights.ndim == 1:
        proba = _expit(distances @ weights)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        gradients = ((proba - y)[:, None] *
                     distances *
                     sample_weight).mean(axis=0)
    else:
        proba = _softmax(distances @ weights, n_samples, n_classes)
        proba = np.clip(proba, 1e-8, 1 - 1e-8)
        gradients = ((proba - y)[:, None, :] *
                     (distances * sample_weight)[:, :, None]).mean(axis=0)

    if penalty == 'l2':
        gradients += (2 / C) * weights
    elif penalty == 'l1':
        gradients += (1 / C) * np.sign(weights)

    return gradients


@njit()
def _compute_shapelet_grad(
    X, n_samples, n_timestamps, weights, shapelets, lengths, alpha,
    proba_minus_y, weight_idx, sample_weight,
):

    gradients = []

    for i in range(len(lengths)):
        X_window = _windowed_view(X, n_samples, n_timestamps,
                                  window_size=lengths[i][0], window_step=1)
        n_windows = X_window.shape[1]
        size = shapelets[i][0].size
        for j in range(shapelets[i].shape[0]):
            # Get the current shapelet
            shapelet = shapelets[i][j]

            # Derive the difference and the distance
            diff = shapelet - X_window
            dist = np.empty((n_samples, n_windows))
            for k in prange(n_samples):
                for l in prange(n_windows):
                    dist[k, l] = np.mean(diff[k, l] ** 2)

            # Derive the softmin gradient
            softmin_gradient = np.empty((n_samples, n_windows))
            for k in prange(n_samples):
                softmin_gradient[k] = _softmin_grad(dist[k], alpha)

            # Normalize the difference for actual gradient
            diff *= (2 / size)

            # Compute the gradient
            grad = np.empty(size)
            if weights.ndim == 1:
                for k in prange(size):
                    grad[k] = np.mean(
                        np.sum(diff[:, :, k] * softmin_gradient, axis=1) *
                        weights[weight_idx] *
                        proba_minus_y *
                        sample_weight
                    )
            else:
                for k in prange(size):
                    grad[k] = np.mean(
                        np.sum(diff[:, :, k] * softmin_gradient, axis=1) *
                        np.sum(weights[weight_idx] * proba_minus_y, axis=1) *
                        sample_weight
                    )
            gradients.append(grad)

            # Update the weight index
            weight_idx += 1

    return gradients


def _grad_shapelets(X, y, n_classes, weights, shapelets, lengths, alpha,
                    penalty, C, fit_intercept, intercept_scaling,
                    sample_weight):
    """Compute the gradient of the loss with regards to the shapelets."""
    n_samples, n_timestamps = X.shape

    # Derive distances between shapelets and time series
    distances = _derive_all_squared_distances(
        X, n_samples, n_timestamps, shapelets, lengths, alpha)
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
        X, n_samples, n_timestamps, weights, shapelets, lengths,
        alpha, proba_minus_y, weight_idx, sample_weight
    )
    gradients = np.concatenate(gradients)

    return gradients


class CrossEntropyLearningShapelets(BaseEstimator, ClassifierMixin):
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

    def __init__(self, n_shapelets_per_size=0.2, min_shapelet_length=0.1,
                 shapelet_scale=3, penalty='l2', tol=0.001, C=1000,
                 learning_rate=1., max_iter=1000, alpha=-100,
                 fit_intercept=True, intercept_scaling=1.,
                 class_weight=None, verbose=0, random_state=None):
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

    def fit(self, X, y, sample_weight=None):
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

        """  # noqa: E501
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        check_classification_targets(y)
        le = LabelEncoder().fit(y)
        y_ind = le.transform(y)
        self.classes_ = le.classes_
        n_classes = len(le.classes_)

        (n_shapelets_per_size, min_shapelet_length, sample_weight,
         rng) = self._check_params(X, y, y_ind, le.classes_, sample_weight)

        if n_classes > 2:
            y_ind = LabelBinarizer().fit_transform(y)

        # Shapelet initialization
        window_sizes = np.arange(
            min_shapelet_length,
            min_shapelet_length * (self.shapelet_scale + 1),
            min_shapelet_length
        )

        n_shapelets_per_cluster = n_timestamps - window_sizes + 1
        if np.any(n_shapelets_per_size > n_shapelets_per_cluster):
            raise ValueError("'n_shapelets_per_size' is too high given "
                             "'min_shapelet_length' and 'shapelet_scale'.")

        shapelets = []
        lengths = []
        for window_size in window_sizes:
            X_window = _windowed_view(
                X, n_samples, n_timestamps, window_size, window_step=1)
            X_window = X_window.reshape(-1, window_size)
            kmeans = KMeans(n_clusters=n_shapelets_per_size, random_state=rng)
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
            X, y_ind, n_classes, weights, shapelets, lengths, self.alpha,
            self.penalty, self.C, self.fit_intercept, self.intercept_scaling,
            sample_weight
        )
        if self.verbose:
            print('Iteration {0}: loss = {1:0.6f}'.format(0, loss_iteration))
        losses.append(loss_iteration)
        for iteration in range(1, self.max_iter + 1):

            # Update weights
            gradient_weights = _grad_weights(
                X, y_ind, n_classes, weights, shapelets, lengths, self.alpha,
                self.penalty, self.C, self.fit_intercept,
                self.intercept_scaling, sample_weight
            )
            weights -= learning_rate * gradient_weights

            # Update shapelets
            gradient_shapelets = _grad_shapelets(
                X, y_ind, n_classes, weights, shapelets, lengths, self.alpha,
                self.penalty, self.C, self.fit_intercept,
                self.intercept_scaling, sample_weight
            )
            shapelets_array = _reshape_array_shapelets(shapelets, lengths)
            shapelets_array -= learning_rate * gradient_shapelets
            shapelets = tuple(
                _reshape_list_shapelets(shapelets_array, lengths))

            # Compute current loss
            loss_iteration = _loss(
                X, y_ind, n_classes, weights, shapelets, lengths, self.alpha,
                self.penalty, self.C, self.fit_intercept,
                self.intercept_scaling, sample_weight
            )

            # If loss is increasing, decrease the learning rate
            if losses[-1] < loss_iteration:
                while losses[-1] < loss_iteration:
                    # Go back to previous state
                    weights += learning_rate * gradient_weights
                    shapelets_array = _reshape_array_shapelets(
                        shapelets, lengths)
                    shapelets_array += learning_rate * gradient_shapelets
                    shapelets = tuple(
                        _reshape_list_shapelets(shapelets_array, lengths))

                    # Update learning  rate
                    learning_rate /= 5

                    # Recompute shapelet gradient
                    weights -= learning_rate * gradient_weights
                    gradient_shapelets = _grad_shapelets(
                        X, y_ind, n_classes, weights, shapelets, lengths,
                        self.alpha, self.penalty, self.C, self.fit_intercept,
                        self.intercept_scaling, sample_weight
                    )
                    shapelets_array = _reshape_array_shapelets(
                        shapelets, lengths)
                    shapelets_array -= learning_rate * gradient_shapelets
                    shapelets = tuple(
                        _reshape_list_shapelets(shapelets_array, lengths))

                    loss_iteration = _loss(
                        X, y_ind, n_classes, weights, shapelets, lengths,
                        self.alpha, self.penalty, self.C, self.fit_intercept,
                        self.intercept_scaling, sample_weight
                    )
            if self.verbose:
                print('Iteration {0}: loss = {1:0.6f}'
                      .format(iteration, loss_iteration))
            losses.append(loss_iteration)

            # Stopping criterion
            if abs(losses[-2] - losses[-1]) < self.tol * losses[-1]:
                break

        if iteration == self.max_iter:
            warnings.warn('Maximum number of iterations reached without '
                          'converging. Increase the maximum number of '
                          'iterations.', ConvergenceWarning)

        # Save results in attributes
        self._shapelets = shapelets
        self._lengths = lengths
        self.shapelets_ = [list(shapelet) for shapelet in shapelets]
        self.shapelets_ = np.asarray(
            list(chain.from_iterable(self.shapelets_)))
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

    def decision_function(self, X):
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
            X, n_samples, n_timestamps, self._shapelets,
            self._lengths, self.alpha
        )
        distances = np.asarray(distances).T

        # Add intercept
        if self.fit_intercept:
            distances = np.c_[np.ones(n_samples) * self.intercept_scaling,
                              distances]

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

    def predict_proba(self, X):
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

    def predict(self, X):
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
            y_pred = (self.decision_function(X) > 0.).astype('int64')
        else:
            y_pred = self.decision_function(X).argmax(axis=1)
        return self.classes_[y_pred]

    def _check_params(self, X, y, y_ind, classes, sample_weight):
        """Parameter check"""
        n_samples, n_timestamps = X.shape

        if not isinstance(self.n_shapelets_per_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'n_shapelets_per_size' must be an integer or a "
                            "float (got {})."
                            .format(self.n_shapelets_per_size))
        if isinstance(self.n_shapelets_per_size, (int, np.integer)):
            if not 1 <= self.n_shapelets_per_size <= n_timestamps:
                raise ValueError(
                    "If 'n_shapelets_per_size' is an integer, it must be "
                    "greater than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.n_shapelets_per_size)
                )
            n_shapelets_per_size = self.n_shapelets_per_size
        else:
            if not (0 < self.n_shapelets_per_size <= 1.):
                raise ValueError(
                    "If 'n_shapelets_per_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.n_shapelets_per_size)
                )
            n_shapelets_per_size = ceil(
                self.n_shapelets_per_size * n_timestamps)

        if not isinstance(self.min_shapelet_length,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'min_shapelet_length' must be an integer or a "
                            "float (got {}).".format(self.min_shapelet_length))
        if isinstance(self.min_shapelet_length, (int, np.integer)):
            if not 1 <= self.min_shapelet_length <= n_timestamps:
                raise ValueError(
                    "If 'min_shapelet_length' is an integer, it must be "
                    "greater than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.min_shapelet_length)
                )
            min_shapelet_length = self.min_shapelet_length
        else:
            if not (0 < self.min_shapelet_length <= 1.):
                raise ValueError(
                    "If 'min_shapelet_length' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.min_shapelet_length)
                )
            min_shapelet_length = ceil(self.min_shapelet_length * n_timestamps)

        if not (isinstance(self.shapelet_scale, (int, np.integer)) and
                self.shapelet_scale > 0):
            raise ValueError("'shapelet_scale' must be a positive integer "
                             "(got {}).".format(self.shapelet_scale))

        if self.shapelet_scale * min_shapelet_length > n_timestamps:
            raise ValueError(
                "'shapelet_scale' and 'min_shapelet_length' must be "
                "such that shapelet_scale * min_shapelet_length is "
                "smaller than or equal to n_timestamps."
            )

        if self.penalty not in ('l1', 'l2'):
            raise ValueError("'penalty' must be either 'l2' or 'l1' "
                             "(got {}).".format(self.penalty))

        if not (isinstance(self.C, (int, np.integer, float, np.floating)) and
                self.C > 0):
            raise ValueError("'C' must be a positive float (got {})."
                             .format(self.C))

        if not (isinstance(self.tol, (int, np.integer, float, np.floating)) and
                self.tol > 0):
            raise ValueError("'tol' must be a positive float (got {})."
                             .format(self.tol))

        if not (isinstance(self.learning_rate,
                           (int, np.integer, float, np.floating)) and
                self.learning_rate > 0):
            raise ValueError("'learning_rate' must be a positive float "
                             "(got {}).".format(self.learning_rate))

        if not (isinstance(self.max_iter, (int, np.integer)) and
                self.max_iter >= 0):
            raise ValueError("'max_iter' must be a non-negative integer "
                             "(got {}).".format(self.max_iter))

        if not (isinstance(self.alpha, (int, np.integer, float, np.floating))
                and self.alpha < 0):
            raise ValueError("'alpha' must be a negative float (got {})."
                             .format(self.alpha))

        if not isinstance(self.intercept_scaling,
                          (int, np.integer, float, np.floating)):
            raise ValueError("'intercept_scaling' must be a float (got {})."
                             .format(self.intercept_scaling))

        class_weight_balanced = (isinstance(self.class_weight, str) and
                                 self.class_weight == 'balanced')
        if not (self.class_weight is None or
                class_weight_balanced or
                isinstance(self.class_weight, dict)):
            raise ValueError("'class_weight' must be None, a dictionary "
                             " or 'balanced' (got {})."
                             .format(self.class_weight))
        class_weight = compute_class_weight(self.class_weight, classes, y)

        sample_weight = _check_sample_weight(sample_weight, X, dtype='float64')
        sample_weight *= class_weight[y_ind]
        sample_weight = sample_weight.reshape(-1, 1)

        rng = check_random_state(self.random_state)

        if not (isinstance(self.verbose, (int, np.integer)) and
                self.verbose >= 0):
            raise ValueError("'verbose' must be a non-negative integer "
                             "(got {}).".format(self.verbose))

        return n_shapelets_per_size, min_shapelet_length, sample_weight, rng


class LearningShapelets(BaseEstimator, ClassifierMixin):
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

    def __init__(self, n_shapelets_per_size=0.2, min_shapelet_length=0.1,
                 shapelet_scale=3, penalty='l2', tol=0.001, C=1000,
                 learning_rate=1., max_iter=1000, multi_class='multinomial',
                 alpha=-100, fit_intercept=True, intercept_scaling=1.,
                 class_weight=None, verbose=0, random_state=None, n_jobs=None):
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

    def fit(self, X, y, sample_weight=None):
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

        """  # noqa: E501
        X, y = check_X_y(X, y)
        n_classes = len(LabelEncoder().fit(y).classes_)
        multi_class = self._check_params(n_classes)
        params = self.get_params()
        params.pop('n_jobs', None)
        params.pop('multi_class')
        clf = CrossEntropyLearningShapelets(**params)
        if multi_class == 'ovr':
            clf = OneVsRestClassifier(clf, self.n_jobs)
        elif multi_class == 'ovo':
            clf = OneVsOneClassifier(clf, self.n_jobs)
        clf.fit(X, y)

        self.classes_ = clf.classes_
        self._multi_class = multi_class

        if multi_class in ('ovr', 'ovo'):
            self._estimators = clf.estimators_
            self.shapelets_ = np.array(
                [est.shapelets_ for est in clf.estimators_])
            self.coef_ = np.squeeze(np.asarray(
                [est.coef_ for est in clf.estimators_]))
            self.intercept_ = np.squeeze(np.array(
                [est.intercept_ for est in clf.estimators_]))
            self.n_iter_ = np.array(
                [est.n_iter_ for est in clf.estimators_])
        else:
            self._clf = clf
            self.shapelets_ = np.array([clf.shapelets_])
            self.coef_ = clf.coef_
            self.intercept_ = clf.intercept_
            self.n_iter_ = np.array([clf.n_iter_])
        return self

    def decision_function(self, X):
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
            predictions = np.vstack(
                [est.predict(X) for est in self._estimators]).T
            confidences = np.vstack(
                [est.decision_function(X) for est in self._estimators]).T
            X_new = _ovr_decision_function(
                predictions, confidences, len(self.classes_))
        else:
            X_new = self._clf.decision_function(X)

        return X_new

    def predict_proba(self, X):
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

    def predict(self, X):
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

    def _check_params(self, n_classes):
        if self.multi_class not in ('multinomial', 'ovr', 'ovo'):
            raise ValueError(
                "'multi_class' must be either 'multinomial', "
                "'ovr' or 'ovo' (got {}).".format(self.multi_class)
            )
        multi_class = 'binary' if n_classes == 2 else self.multi_class

        class_weight_dict = isinstance(self.class_weight, dict)
        if multi_class in ('ovr', 'ovo') and class_weight_dict:
            raise ValueError("'class_weight' must be None or 'balanced' if "
                             "'multi_class' is either 'ovr' or 'ovo'.")

        n_jobs_int = (isinstance(self.n_jobs, (int, np.integer)) and
                      self.n_jobs != 0)
        if not (self.n_jobs is None or n_jobs_int):
            raise ValueError("'n_jobs' must be None or an integer not equal "
                             "to zero (got {}).".format(self.n_jobs))

        return multi_class
