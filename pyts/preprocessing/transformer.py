"""Code for transformers."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer as SklearnPowerTransformer
from sklearn.preprocessing import (QuantileTransformer as
                                   SklearnQuantileTransformer)
from sklearn.utils.validation import check_array


class PowerTransformer(BaseEstimator, TransformerMixin):
    """Apply a power transform sample-wise to make data more Gaussian-like.

    Power transforms are a family of parametric, monotonic transformations
    that are applied to make data more Gaussian-like. This is useful for
    modeling issues related to heteroscedasticity (non-constant variance),
    or other situations where normality is desired.

    Currently, PowerTransformer supports the Box-Cox transform and the
    Yeo-Johnson transform. The optimal parameter for stabilizing variance and
    minimizing skewness is estimated through maximum likelihood.

    Box-Cox requires input data to be strictly positive, while Yeo-Johnson
    supports both positive or negative data.

    By default, zero-mean, unit-variance normalization is applied to the
    transformed data.

    Parameters
    ----------
    method : 'yeo-johnson' or 'box-cox' (default = 'yeo-johnson')
        The power transform method. Available methods are:

        - 'yeo-johnson' [1]_, works with positive and negative values
        - 'box-cox' [2]_, only works with strictly positive values

    standardize : boolean (default = True)
        Set to True to apply zero-mean, unit-variance normalization to the
        transformed output.

    Notes
    -----
    NaNs are treated as missing values: disregarded in ``fit``, and maintained
    in ``transform``.

    References
    ----------
    .. [1] I.K. Yeo and R.A. Johnson, "A new family of power transformations to
           improve normality or symmetry." Biometrika, 87(4), pp.954-959,
           (2000).
    .. [2] G.E.P. Box and D.R. Cox, "An Analysis of Transformations", Journal
           of the Royal Statistical Society B, 26, 211-252 (1964).

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.preprocessing import PowerTransformer
    >>> X = [[1, 3, 4], [2, 2, 5]]
    >>> pt = PowerTransformer()
    >>> print(pt.transform(X))
    [[-1.316...  0.209...  1.106...]
     [-0.707... -0.707...  1.414...]]

    """

    def __init__(self, method='yeo-johnson', standardize=True):
        self.method = method
        self.standardize = standardize

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Transformed data.

        """
        X = check_array(X, dtype='float64', force_all_finite='allow-nan')
        transformer = SklearnPowerTransformer(
            method=self.method, standardize=self.standardize
        )
        X_new = transformer.fit_transform(X.T).T
        return X_new


class QuantileTransformer(BaseEstimator, TransformerMixin):
    """Transform samples using quantiles information.

    This method transforms the samples to follow a uniform or a normal
    distribution. Therefore, for a given sample, this transformation tends
    to spread out the most frequent values. It also reduces the impact of
    (marginal) outliers: this is therefore a robust preprocessing scheme.
    The transformation is applied on each sample independently.

    The cumulative distribution function of a feature is used to project the
    original values. Note that this transform is non-linear.

    Parameters
    ----------
    n_quantiles : int, optional (default = 1000)
        Number of quantiles to be computed. It corresponds to the number
        of landmarks used to discretize the cumulative distribution function.

    output_distribution : 'uniform' or 'normal' (default = 'uniform')
        Marginal distribution for the transformed data. The choices are
        'uniform' (default) or 'normal'.

    subsample : int, optional (default = 1e5)
        Maximum number of timestamps used to estimate the quantiles for
        computational efficiency.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by np.random. Note that this is used by subsampling and smoothing
        noise.

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.preprocessing import QuantileTransformer
    >>> X, _, _, _  = load_gunpoint(return_X_y=True)
    >>> qt = QuantileTransformer(n_quantiles=10)
    >>> qt.transform(X)
    array([...])

    """

    def __init__(self, n_quantiles=1000, output_distribution='uniform',
                 subsample=int(1e5), random_state=None):
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample
        self.random_state = random_state

    def fit(self, X=None, y=None):
        """Pass.

        Parameters
        ----------
        X
            Ignored

        y
            Ignored

        Returns
        -------
        self : object

        """
        return self

    def transform(self, X):
        """Transform the data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Data to transform.

        Returns
        -------
        X_new : array-like, shape = (n_samples, n_timestamps)
            Transformed data.

        """
        X = check_array(X, dtype='float64')
        transformer = SklearnQuantileTransformer(
            n_quantiles=self.n_quantiles,
            output_distribution=self.output_distribution,
            subsample=self.subsample,
            random_state=self.random_state
        )
        X_new = transformer.fit_transform(X.T).T
        return X_new
