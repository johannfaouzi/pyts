"""Functions to make datasets."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
from sklearn.utils import check_array, check_random_state
from sklearn.utils import shuffle as utils_shuffle


def make_cylinder_bell_funnel(
    n_samples=30, weights=None, shuffle=True, random_state=None,
    return_params=False
):
    r"""Make a Cylinder-Bell-Funnel dataset.

    The classes are coded with the following meaning:

        - 0: Cylinder
        - 1: Bell
        - 2: Funnel

    Parameters
    ----------
    n_samples : int (default = 30)
        The number of time series.

    weights : list of floats or None (default=None)
        The proportions of samples assigned to each class. If None, then
        classes are balanced. Note that if ``len(weights) == 2``,
        then the last class weight is automatically inferred.
        More than ``n_samples`` samples may be returned if the sum of
        ``weights`` exceeds 1.

    shuffle : bool (default = True)
        If True, shuffle the samples.

    return_params : bool (default = False)
        If True, a dictionary containing the parameters used to make the
        time series is returned.

    random_state : int, RandomState instance or None (default = None)
        Determines random number generation for dataset creation. Pass an int
        for reproducible output across multiple function calls.

    Returns
    -------
    X : array, shape = (n_samples, 128)
        Generated time series.

    y : array, shape = (n_samples,)
        Generated class labels. Labels have the following meaning:

        - 0: Cylinder
        - 1: Bell
        - 2: Funnel

    params : dict
        The parameters used to generate the data. The keys are 'a',
        'b', 'eta' and 'epsilon' and the values are the corresponding
        values. Only returned if ``return_params=True``.

    Notes
    -----
    The time series are generated from the following distributions:

    .. math::

        c(t) = (6 + \eta) \cdot 1_{[a, b]}(t) + \epsilon(t)

        b(t) = (6 + \eta) \cdot 1_{[a, b]}(t) \cdot (t - a) / (b - a) +
        \epsilon(t)

        f(t) = (6 + \eta) \cdot 1_{[a, b]}(t) \cdot (b - t) / (b - a) +
        \epsilon(t)

    where :math:`t=1,\ldots,128`, :math:`a` is an integer-valued uniform random
    variable on the inverval :math:`[16, 32]`, :math:`b-a` is an integer-valued
    uniform distribution on the inveral :math:`[32, 96]`, :math:`\eta` and
    :math:`\epsilon(t)` are standard normal variables,
    :math:`{1}_{[a, b]}` is the characteristic function on the interval
    :math:`[a, b]`. :math:`c`, :math:`b`, and :math:`f` stand for "cylinder",
    "bell", and "funnel" respectively.

    References
    ----------
    .. [1] N. Saito, "Local feature extraction and its application
           using a library of bases". Ph.D. thesis, Department of
           Mathematics, Yale University, 1994.

    Examples
    --------
    >>> import numpy as np
    >>> from pyts.datasets import make_cylinder_bell_funnel
    >>> X, y = make_cylinder_bell_funnel()
    >>> X.shape
    (30, 128)
    >>> print(np.bincount(y))
    [10 10 10]

    """
    if not isinstance(n_samples, (int, np.integer)):
        raise TypeError("'n_samples' must be an integer.")
    if not n_samples > 0:
        raise ValueError("'n_samples' must be a positive integer.")
    if weights is None:
        weights = [1.0 / 3] * 3
        weights[-1] = 1.0 - sum(weights[:-1])
    else:
        weights = check_array(weights, ensure_2d=False)
        if len(weights) not in [2, 3]:
            raise ValueError("'weights' must be None or a list with 2 or "
                             "3 elements (got {}).".format(len(weights)))
        if len(weights) == 2:
            if sum(weights) > 1:
                raise ValueError(
                    "'sum(weights)' cannot be larger than 1 if "
                    "len(weights) == 2 (got {})".format(sum(weights)))
            weights = np.append(weights, 1.0 - sum(weights))
    rng = check_random_state(random_state)

    n_samples_per_class = [int(n_samples * weights[k]) for k in range(3)]
    for i in range(n_samples - sum(n_samples_per_class)):
        n_samples_per_class[i] += 1
    n_samples = sum(n_samples_per_class)

    y = ([0] * n_samples_per_class[0] +
         [1] * n_samples_per_class[1] +
         [2] * n_samples_per_class[2])
    y = np.asarray(y)

    a = rng.randint(16, 33)
    b = rng.randint(32, 97) + a
    eta = rng.randn()
    epsilon = rng.randn(n_samples, 128)

    arange = np.tile(np.arange(1, 129), n_samples).reshape(n_samples, 128)
    additional_term = (6 + eta) * np.logical_and(arange >= a, arange <= b)
    i, j = n_samples_per_class[0], sum(n_samples_per_class[0:2])
    additional_term[i:j] *= (np.arange(1, 129) - a) / (b - a)
    additional_term[j:] *= (b - np.arange(1, 129)) / (b - a)

    X = epsilon + additional_term

    if shuffle:
        X, y = utils_shuffle(X, y, random_state=rng)

    if return_params:
        params = {'a': a, 'b': b, 'eta': eta, 'epsilon': epsilon}
        return X, y, params
    else:
        return X, y
