"""Code for Discrete Fourier Transform."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import ceil
from typing import Any, Self
from warnings import warn

import numpy as np
import numpy.typing as npt
from sklearn.base import BaseEstimator
from sklearn.feature_selection import f_classif
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from pyts._base import UnivariateTransformerMixin
from pyts.preprocessing import StandardScaler


class DiscreteFourierTransform(BaseEstimator, UnivariateTransformerMixin):
    """Discrete Fourier Transform.

    Parameters
    ----------
    n_coefs : None, int or float (default = None)
        The number of Fourier coefficients to keep. If None, all the Fourier
        coeeficients are kept. If an integer, the ``n_coefs`` most significant
        Fourier coefficients are returned if ``anova=True``, otherwise the
        first ``n_coefs`` Fourier coefficients are returned. If a float, it
        represents a percentage of the size of each time series and must be
        between 0 and 1. The number of coefficients will be computed as
        ``ceil(n_coefs * (n_timestamps - 1))`` if ``drop_sum=True`` and
        ``ceil(n_coefs * n_timestamps)`` if ``drop_sum=False``.

    drop_sum : bool (default = False)
        If True, the first Fourier coefficient (i.e. the sum of the subseries)
        is dropped. Otherwise, it is kept.

    anova : bool (default = False)
        If True, the Fourier coefficient selection is done via a one-way
        ANOVA test. If False, the first Fourier coefficients are selected.

    norm_mean : bool (default = False)
        If True, center each time series before scaling.

    norm_std : bool (default = False)
        If True, scale each time series to unit variance.

    Attributes
    ----------
    support_ : array, shape = (n_coefs,)
        Indices of the kept Fourier coefficients.

    References
    ----------
    .. [1] P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
           and Index for Similarity Search in High Dimensional Datasets",
           International Conference on Extending Database Technology,
           15, 516-527 (2012).

    Examples
    --------
    >>> from pyts.approximation import DiscreteFourierTransform
    >>> from pyts.datasets import load_gunpoint
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = DiscreteFourierTransform(n_coefs=4)
    >>> X_new = transformer.fit_transform(X)
    >>> X_new.shape
    (50, 4)

    """

    def __init__(
        self,
        n_coefs: int | float | None = None,
        drop_sum: bool = False,
        anova: bool = False,
        norm_mean: bool = False,
        norm_std: bool = False,
    ) -> None:
        self.n_coefs = n_coefs
        self.drop_sum = drop_sum
        self.anova = anova
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike | None = None) -> Self:
        """Learn indices of the Fourier coefficients to keep.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels for each data sample. Only used if ``anova=True``.

        Returns
        -------
        self : object

        """
        if self.anova:
            X, y = check_X_y(X, y, dtype=np.float64)
        else:
            X = check_array(X, dtype=np.float64)

        n_samples, n_timestamps = X.shape
        n_coefs = self._check_params(n_timestamps)
        if self.anova:
            ss = StandardScaler(self.norm_mean, self.norm_std)
            X = ss.fit_transform(X)
            X_fft = np.fft.rfft(X)
            X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
            if n_timestamps % 2 == 0:
                X_fft = X_fft.reshape(n_samples, n_timestamps + 2, order='F')
                X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:-1]]
            else:
                X_fft = X_fft.reshape(n_samples, n_timestamps + 1, order='F')
                X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:]]
            if self.drop_sum:
                X_fft = X_fft[:, 1:]
            self.support_ = self._anova(X_fft, y, n_coefs, n_timestamps)
        else:
            self.support_ = np.arange(n_coefs)
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return the selected Fourier coefficients for each sample.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Input data.

        Returns
        -------
        X_new : array, shape (n_samples, n_coefs)
            The selected Fourier coefficients for each sample.

        """
        check_is_fitted(self, 'support_')
        X = check_array(X, dtype=np.float64)
        n_samples, n_timestamps = X.shape

        ss = StandardScaler(self.norm_mean, self.norm_std)
        X = ss.fit_transform(X)
        X_fft = np.fft.rfft(X)
        X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
        if n_timestamps % 2 == 0:
            X_fft = X_fft.reshape(n_samples, n_timestamps + 2, order='F')
            X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:-1]]
        else:
            X_fft = X_fft.reshape(n_samples, n_timestamps + 1, order='F')
            X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:]]
        if self.drop_sum:
            X_fft = X_fft[:, 1:]
        return X_fft[:, self.support_]

    # This overrides UnivariateTransformerMixin.fit_transform with a fully
    # custom implementation (rather than delegating to fit().transform())
    # and, by pre-existing design, does not accept arbitrary **fit_params.
    def fit_transform(  # pyrefly: ignore[bad-override]
        self, X: npt.ArrayLike, y: npt.ArrayLike | None = None
    ) -> npt.NDArray[np.float64]:
        """Learn and return the Fourier coeeficients to keep.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : None or array-like, shape = (n_samples,) (default = None)
            Class labels for each data sample.

        Returns
        -------
        X_new : array, shape (n_samples, n_coefs)
            The selected Fourier coefficients for each sample.

        """
        if self.anova:
            X, y = check_X_y(X, y, dtype=np.float64)
        else:
            X = check_array(X, dtype=np.float64)
        n_samples, n_timestamps = X.shape
        n_coefs = self._check_params(n_timestamps)

        scaler = StandardScaler(self.norm_mean, self.norm_std)
        X = scaler.fit_transform(X)
        X_fft = np.fft.rfft(X)
        X_fft = np.vstack([np.real(X_fft), np.imag(X_fft)])
        if n_timestamps % 2 == 0:
            X_fft = X_fft.reshape(n_samples, n_timestamps + 2, order='F')
            X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:-1]]
        else:
            X_fft = X_fft.reshape(n_samples, n_timestamps + 1, order='F')
            X_fft = np.c_[X_fft[:, 0], X_fft[:, 2:]]
        if self.drop_sum:
            X_fft = X_fft[:, 1:]
        if self.anova:
            self.support_ = self._anova(X_fft, y, n_coefs, n_timestamps)
        else:
            self.support_ = np.arange(n_coefs)
        return X_fft[:, self.support_]

    def _anova(
        self,
        X_fft: npt.NDArray[np.float64],
        y: npt.NDArray[Any],
        n_coefs: int,
        n_timestamps: int,
    ) -> npt.NDArray[np.int64]:
        if n_coefs < X_fft.shape[1]:
            non_constant = np.where(
                ~np.isclose(X_fft.var(axis=0), np.zeros_like(X_fft.shape[1]))
            )[0]
            if non_constant.size == 0:
                raise ValueError(
                    "All the Fourier coefficients are constant. "
                    "Your input data is weirdly homogeneous."
                )
            elif non_constant.size < n_coefs:
                warn(
                    "The number of non constant Fourier coefficients "
                    f"({non_constant.size}) is lower than the number of "
                    f"coefficients to keep ({n_coefs}). The number of "
                    "coefficients to keep is truncated to "
                    f"{non_constant.size}"
                    ".",
                    stacklevel=2,
                )
                support = non_constant
            else:
                _, p = f_classif(X_fft[:, non_constant], y)
                support = non_constant[np.argsort(p)[:n_coefs]]
        else:
            support = np.arange(n_coefs)
        return support

    def _check_params(self, n_timestamps: int) -> int:
        if not (
            (isinstance(self.n_coefs, (int, np.integer, float, np.floating)))
            or (self.n_coefs is None)
        ):
            raise TypeError("'n_coefs' must be None, an integer or a float.")
        if isinstance(self.n_coefs, (int, np.integer)):
            if self.drop_sum and not (1 <= self.n_coefs <= n_timestamps - 1):
                raise ValueError(
                    "If 'n_coefs' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to (n_timestamps - 1) "
                    "if 'drop_sum=True'."
                )
            if not self.drop_sum and not (1 <= self.n_coefs <= n_timestamps):
                raise ValueError(
                    "If 'n_coefs' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to n_timestamps "
                    "if 'drop_sum=False'."
                )
            n_coefs = self.n_coefs
        elif isinstance(self.n_coefs, (float, np.floating)):
            if not 0 < self.n_coefs <= 1:
                raise ValueError(
                    "If 'n_coefs' is a float, it must be greater "
                    "than 0 and lower than or equal to 1."
                )
            if self.drop_sum:
                n_coefs = ceil(self.n_coefs * (n_timestamps - 1))
            else:
                n_coefs = ceil(self.n_coefs * n_timestamps)
        else:
            n_coefs = (n_timestamps - 1) if self.drop_sum else n_timestamps
        return n_coefs
