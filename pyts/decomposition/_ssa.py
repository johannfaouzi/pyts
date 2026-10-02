"""Code for Singular Spectrum Analysis."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import itertools
from math import ceil
from typing import Self

import numpy as np
import numpy.typing as npt
from joblib import Parallel, delayed
from numba import njit, prange
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array

from pyts._base import UnivariateTransformerMixin
from pyts.utils._utils import _windowed_view


@njit(
    "float64[:,:,:](float64[:,:,:], float64[:,:,::1], float64[:,:,:], "
    "int64, int64, int64, int64, int64)"
)
def _ssa_fused(
    v: npt.NDArray[np.float64],
    X_window: npt.NDArray[np.float64],
    membership: npt.NDArray[np.float64],
    n_samples: int,
    n_timestamps: int,
    window_size: int,
    n_windows: int,
    grouping_size: int,
) -> npt.NDArray[np.float64]:
    """Fuse the outer-product/grouping/diagonal-averaging pipeline.

    An unfused pipeline (computing each elementary matrix via an
    outer-product/dot, summing them per group, then diagonally averaging
    each group) would build the full
    ``(n_samples, window_size, window_size, n_windows)`` stack of
    elementary matrices -- one ``outer(v[:, j], v[:, j]) @ X_window`` per
    component ``j`` -- which is cubic in the input size and is the
    dominant peak-memory cost of ``SingularSpectrumAnalysis``. Since
    ``outer(v[:, j], v[:, j]) @ X_window == outer(v[:, j], y[j])`` with
    ``y = v.T @ X_window``, elementary component ``j`` is a rank-1 outer
    product; summing the components of one group is therefore computed
    directly into a single ``(window_size, n_windows)`` buffer (reused
    across groups) instead of ever holding all ``window_size`` components
    at once, and that buffer is diagonally averaged into its
    ``(n_timestamps,)`` output row immediately, before moving to the next
    group. ``membership[i, g, j]`` is 1 if elementary component ``j``
    belongs to group ``g`` for sample ``i``, 0 otherwise -- see
    ``SingularSpectrumAnalysis._grouping``.
    """
    if window_size >= n_windows:
        gap = window_size
    else:
        gap = n_windows
    first_row = [(0, col) for col in range(n_windows)]
    last_col = [(row, n_windows - 1) for row in range(1, window_size)]
    indices = first_row + last_col
    X_new = np.empty((n_samples, grouping_size, n_timestamps))
    for i in prange(n_samples):
        # Shared by every group: elementary component j is
        # outer(v[i, :, j], y[j]). ``.copy()`` makes the transposed
        # array contiguous, which BLAS needs to avoid falling back to a
        # slower path (the copy itself is cheap: it is only
        # (window_size, window_size), not the cubic elementary-matrix
        # stack).
        y = np.dot(np.transpose(v[i]).copy(), X_window[i])
        for g in range(grouping_size):
            buffer = np.zeros((window_size, n_windows))
            for j in range(window_size):
                m = membership[i, g, j]
                if m != 0.0:
                    buffer += m * np.outer(v[i, :, j], y[j])
            if window_size >= n_windows:
                buffer_t = np.transpose(buffer).copy()
            else:
                buffer_t = buffer
            for j, k in indices:
                X_new[i, g, j + k] = np.diag(buffer_t[:, ::-1], gap - j - k - 1).mean()
    return X_new


class SingularSpectrumAnalysis(BaseEstimator, UnivariateTransformerMixin):
    """Singular Spectrum Analysis.

    Parameters
    ----------
    window_size : int or float (default = 4)
        Size of the sliding window (i.e. the size of each word). If float, it
        represents the percentage of the size of each time series and must be
        between 0 and 1. The window size will be computed as
        ``max(2, ceil(window_size * n_timestamps))``.

    groups : None, int, 'auto', or array-like (default = None)
        The way the elementary matrices are grouped. If None, no grouping is
        performed. If an integer, it represents the number of groups and the
        bounds of the groups are computed as
        ``np.linspace(0, window_size, groups + 1).astype('int64')``.
        If 'auto', then three groups are determined, containing trend,
        seasonal, and residual. If array-like, each element must be array-like
        and contain the indices for each group.

    lower_frequency_bound : float (default = 0.075)
        The boundary of the periodogram to characterize trend, seasonal and
        residual components. It must be between 0 and 0.5.
        Ignored if ``groups`` is not set to 'auto'.

    lower_frequency_contribution : float (default = 0.85)
        The relative threshold to characterize trend, seasonal and
        residual components by considering the periodogram.
        It must be between 0 and 1. Ignored if ``groups`` is not set to 'auto'.

    chunksize : int or None (default = None)
        If int, the transformation of the whole dataset is performed using
        chunks (batches) and ``chunksize`` corresponds to the maximum size of
        each chunk (batch). If None, the transformation is performed on the
        whole dataset at once. Performing the transformation with chunks is
        likely to be a bit slower but requires less memory.

    n_jobs : None or int (default = None)
        The number of jobs to use for the computation. Only used if
        ``chunksize`` is set to an integer.

    References
    ----------
    .. [1] N. Golyandina, and A. Zhigljavsky, "Singular Spectrum Analysis for
           Time Series". Springer-Verlag Berlin Heidelberg (2013).

    .. [2] T. Alexandrov, "A Method of Trend Extraction Using Singular
           Spectrum Analysis", REVSTAT (2008).

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.decomposition import SingularSpectrumAnalysis
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = SingularSpectrumAnalysis(window_size=5)
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 5, 150)

    """

    def __init__(
        self,
        window_size: int | float = 4,
        groups: int | str | npt.ArrayLike | None = None,
        lower_frequency_bound: float = 0.075,
        lower_frequency_contribution: float = 0.85,
        chunksize: int | None = None,
        n_jobs: int | None = 1,
    ) -> None:
        self.window_size = window_size
        self.groups = groups
        self.lower_frequency_bound = lower_frequency_bound
        self.lower_frequency_contribution = lower_frequency_contribution
        self.chunksize = chunksize
        self.n_jobs = n_jobs

    def fit(
        self, X: npt.ArrayLike | None = None, y: npt.ArrayLike | None = None
    ) -> Self:
        """Pass.

        Parameters
        ----------
        X
            ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_splits, n_timestamps)
            Transformed data. ``n_splits`` value depends on the value of
            ``groups``. If ``groups=None``, ``n_splits`` is equal to
            ``window_size``. If ``groups`` is an integer, ``n_splits`` is
            equal to ``groups``. If ``groups='auto'``, ``n_splits`` is equal
            to three. If ``groups`` is array-like, ``n_splits`` is equal to
            the length of ``groups``. If ``n_splits=1``, ``X_new`` is squeezed
            and its shape is (n_samples, n_timestamps).

        """
        X = check_array(X, dtype=np.float64)
        n_samples, n_timestamps = X.shape
        window_size, grouping_size = self._check_params(n_timestamps)
        n_windows = n_timestamps - window_size + 1

        try:
            # Get a rough estimation of the required memory. ``_transform``
            # never materializes the (n_samples, window_size, window_size,
            # n_windows) stack of elementary matrices (see ``_ssa_fused``),
            # so the dominant arrays are now the windowed view / eigenvector
            # workspace (n_samples, window_size, window_size or n_windows)
            # and the (n_samples, grouping_size, n_timestamps) output.
            max_array = np.zeros(
                (
                    n_samples + 1,
                    window_size,
                    max(window_size, n_windows) + grouping_size,
                )
            )
            del max_array
        except MemoryError as err:
            msg = "The required memory is greater than the available memory. "
            if self.chunksize is None:
                msg += (
                    "Set the `chunksize` parameter to an integer to perform "
                    "the transformation using chunks (batches) to decrease "
                    "the required memory."
                )
            else:
                msg += (
                    "Decrease the value of the `chunksize` parameter to "
                    "to decrease the required memory."
                )
            raise MemoryError(msg) from err

        if self.chunksize is None:
            return self._transform(X)
        else:
            idx = np.r_[np.arange(0, n_samples, self.chunksize), n_samples]
            # Each chunk keeps its own (chunk_size, n_splits, n_timestamps)
            # array, so the per-chunk results must be concatenated along
            # the sample axis (axis=0), not stacked into a new leading
            # axis (which ``np.asarray`` on a list of arrays would do).
            return np.concatenate(
                Parallel(n_jobs=self.n_jobs)(
                    delayed(self._transform)(X[i:j]) for i, j in itertools.pairwise(idx)
                ),
                axis=0,
            )

    def _transform(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        n_samples, n_timestamps = X.shape
        window_size, grouping_size = self._check_params(n_timestamps)
        n_windows = n_timestamps - window_size + 1

        X_window = np.transpose(
            _windowed_view(X, n_samples, n_timestamps, window_size, window_step=1),
            axes=(0, 2, 1),
        ).copy()
        X_tranpose = np.matmul(X_window, np.transpose(X_window, axes=(0, 2, 1)))
        w, v = np.linalg.eigh(X_tranpose)
        w, v = w[:, ::-1], v[:, :, ::-1]

        del X_tranpose

        # The (n_samples, window_size, window_size, n_windows) stack of
        # elementary matrices that the old code built via ``_outer_dot``
        # (and then reduced via grouping + ``_diagonal_averaging``) is
        # never materialized: ``_grouping`` only determines which
        # elementary component belongs to which group (``membership``),
        # and ``_ssa_fused`` builds, sums, and diagonally averages one
        # group's (window_size, n_windows) matrix at a time.
        membership, grouping_size = self._grouping(
            v,
            n_samples,
            window_size,
            grouping_size,
        )
        X_ssa = _ssa_fused(
            v,
            X_window,
            membership,
            n_samples,
            n_timestamps,
            window_size,
            n_windows,
            grouping_size,
        )
        # Only squeeze away the n_splits axis (axis=1) when there is a
        # single group, per the documented behavior of ``transform``. An
        # unconditional ``np.squeeze(X_ssa)`` would also drop the samples
        # axis (axis=0) whenever this method is called on a single-sample
        # chunk, which corrupts the per-chunk concatenation done by
        # ``transform`` when ``chunksize`` is set.
        if grouping_size == 1:
            return np.squeeze(X_ssa, axis=1)
        return X_ssa

    def _grouping(
        self,
        v: npt.NDArray[np.float64],
        n_samples: int,
        window_size: int,
        grouping_size: int,
    ) -> tuple[npt.NDArray[np.float64], int]:
        """Determine, for each sample, which elementary component (of the
        ``window_size`` available) belongs to which group.

        Returns ``membership``, shape (n_samples, grouping_size,
        window_size): ``membership[i, g, j]`` is 1 if component ``j``
        belongs to group ``g`` for sample ``i``, 0 otherwise. This is
        deliberately *not* the (n_samples, grouping_size, window_size,
        n_windows) sum of elementary matrices that this method used to
        return -- that sum is now computed lazily, one group at a time,
        by ``_ssa_fused``, so it never needs to hold every group's
        elementary-matrix sum in memory simultaneously.
        """
        if self.groups is None:
            # Every component is its own group: membership is the
            # (window_size, window_size) identity, broadcast per sample.
            membership = np.tile(np.eye(window_size), (n_samples, 1, 1))
        elif self.groups == "auto":
            f = np.arange(0, 1 + window_size // 2) / window_size
            Pxx = np.abs(np.fft.rfft(v, axis=1, norm='ortho')) ** 2
            if Pxx.shape[-1] % 2 == 0:
                Pxx[:, 1:-1, :] *= 2
            else:
                Pxx[:, 1:, :] *= 2

            Pxx_cumsum = np.cumsum(Pxx, axis=1)
            idx_trend = np.where(f < self.lower_frequency_bound)[0][-1]
            idx_resid = Pxx_cumsum.shape[1] // 2

            c = self.lower_frequency_contribution
            trend = Pxx_cumsum[:, idx_trend, :] / Pxx_cumsum[:, -1, :] > c
            resid = Pxx_cumsum[:, idx_resid, :] / Pxx_cumsum[:, -1, :] < c
            season = np.logical_and(~trend, ~resid)

            membership = np.zeros((n_samples, grouping_size, window_size))
            for i in range(n_samples):
                for j, arr in enumerate((trend, season, resid)):
                    membership[i, j] = arr[i]
        elif isinstance(self.groups, (int, np.integer)):
            grouping = np.linspace(0, window_size, self.groups + 1).astype('int64')
            membership = np.zeros((n_samples, grouping_size, window_size))
            for i, (j, k) in enumerate(itertools.pairwise(grouping)):
                membership[:, i, j:k] = 1.0
        else:
            membership = np.zeros((n_samples, grouping_size, window_size))
            for i, group in enumerate(self.groups):
                membership[:, i, group] = 1.0
        return membership, grouping_size

    def _check_params(self, n_timestamps: int) -> tuple[int, int]:
        if not isinstance(self.window_size, (int, np.integer, float, np.floating)):
            raise TypeError("'window_size' must be an integer or a float.")
        if isinstance(self.window_size, (int, np.integer)):
            if not 2 <= self.window_size <= n_timestamps:
                raise ValueError(
                    "If 'window_size' is an integer, it must be greater "
                    "than or equal to 2 and lower than or equal to "
                    f"n_timestamps (got {self.window_size})."
                )
            window_size = self.window_size
        else:
            if not 0 < self.window_size <= 1:
                raise ValueError(
                    "If 'window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 "
                    f"(got {self.window_size})."
                )
            window_size = max(2, ceil(self.window_size * n_timestamps))

        if not (
            self.groups is None
            or (isinstance(self.groups, str) and self.groups == "auto")
            or isinstance(self.groups, (int, list, tuple, np.ndarray))
        ):
            raise TypeError(
                "'groups' must be either None, an integer, 'auto' or array-like."
            )
        if self.groups is None:
            grouping_size = window_size
        elif isinstance(self.groups, str) and self.groups == "auto":
            grouping_size = 3
        elif isinstance(self.groups, (int, np.integer)):
            if not 1 <= self.groups <= self.window_size:
                raise ValueError(
                    "If 'groups' is an integer, it must be greater than or "
                    "equal to 1 and lower than or equal to 'window_size'."
                )
            grouping = np.linspace(0, window_size, self.groups + 1).astype('int64')
            grouping_size = len(grouping) - 1
        else:
            # The TypeError check above guarantees 'groups' is None, 'auto',
            # an integer, or array-like (list, tuple, np.ndarray); the first
            # three are handled above, so this is the array-like case.
            assert isinstance(self.groups, (list, tuple, np.ndarray))
            idx = np.concatenate(self.groups)
            diff = np.setdiff1d(idx, np.arange(self.window_size))
            flat_list = [item for group in self.groups for item in group]
            if (diff.size > 0) or not (
                all(isinstance(x, (int, np.integer)) for x in flat_list)
            ):
                raise ValueError(
                    "If 'groups' is array-like, all the values in 'groups' "
                    "must be integers between 0 and ('window_size' - 1)."
                )
            grouping_size = len(self.groups)

        if not isinstance(self.lower_frequency_bound, (float, np.floating)):
            raise TypeError("'lower_frequency_bound' must be a float.")
        else:
            if not 0 < self.lower_frequency_bound < 0.5:
                raise ValueError(
                    "'lower_frequency_bound' must be greater than 0 and lower than 0.5."
                )

        if not isinstance(self.lower_frequency_contribution, (float, np.floating)):
            raise TypeError("'lower_frequency_contribution' must be a float.")
        else:
            if not 0 < self.lower_frequency_contribution < 1:
                raise ValueError(
                    "'lower_frequency_contribution' must be greater than 0 "
                    "and lower than 1."
                )

        if not (
            self.chunksize is None or isinstance(self.chunksize, (int, np.integer))
        ):
            raise TypeError("'chunksize' must be None or an integer.")
        if isinstance(self.chunksize, (int, np.integer)) and self.chunksize < 1:
            raise ValueError(
                "If 'chunksize' is an integer, it must be "
                f"positive (got {self.chunksize})"
            )

        if not (
            self.n_jobs is None
            or (isinstance(self.n_jobs, (int, np.integer)) and self.n_jobs != 0)
        ):
            raise ValueError(
                "'n_jobs' must be None or an integer not equal "
                f"to zero (got {self.n_jobs})."
            )
        return window_size, grouping_size
