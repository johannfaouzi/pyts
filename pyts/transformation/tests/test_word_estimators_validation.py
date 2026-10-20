"""Validation of the alphabet and window sizes of word-based estimators."""

# License: BSD-3-Clause

import re

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_parameters_default_constructible

from pyts.classification import BOSSVS
from pyts.datasets import load_basic_motions, load_gunpoint
from pyts.multivariate.transformation import WEASELMUSE
from pyts.transformation import BOSS, WEASEL

X_uni, _, y_uni, _ = load_gunpoint(return_X_y=True)
X_multi, _, y_multi, _ = load_basic_motions(return_X_y=True)

ORDINAL_ERR_MSG = (
    "'alphabet' must be None or array-like with shape (n_bins,) (got ordinal)."
)


@pytest.mark.parametrize(
    'estimator, X, y',
    [
        (BOSS(word_size=2, n_bins=2, window_size=10), X_uni, None),
        (WEASEL(word_size=2, n_bins=2, window_sizes=[12, 36]), X_uni, y_uni),
        (BOSSVS(word_size=2, n_bins=2, window_size=28), X_uni, y_uni),
        (WEASELMUSE(word_size=2, n_bins=2, window_sizes=[12, 36]), X_multi, y_multi),
    ],
    ids=['BOSS', 'WEASEL', 'BOSSVS', 'WEASELMUSE'],
)
def test_ordinal_alphabet_is_rejected(estimator, X, y):
    """Words are built by joining symbols, so integer symbols are rejected.

    This used to fail with a TypeError from ``str.join`` deep inside ``fit``.
    """
    estimator = clone(estimator).set_params(alphabet='ordinal')
    with pytest.raises(TypeError, match=re.escape(ORDINAL_ERR_MSG)):
        estimator.fit(X, y)


@pytest.mark.parametrize('Estimator', [WEASEL, WEASELMUSE])
def test_window_sizes_not_mutated_on_init(Estimator):
    """The default ``window_sizes`` is resolved at fit time, not in __init__."""
    estimator = Estimator()
    assert estimator.get_params()['window_sizes'] is None
    check_parameters_default_constructible(Estimator.__name__, estimator)


def test_weasel_default_window_sizes():
    """``window_sizes=None`` uses [0.1, 0.3, 0.5, 0.7, 0.9]."""
    explicit = WEASEL(window_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], sparse=False)
    default = WEASEL(window_sizes=[12, 36], sparse=False).set_params(window_sizes=None)
    np.testing.assert_array_equal(
        default.fit(X_uni, y_uni).transform(X_uni),
        explicit.fit(X_uni, y_uni).transform(X_uni),
    )


def test_weasel_muse_default_window_sizes():
    """WEASELMUSE forwards ``window_sizes=None`` to WEASEL."""
    explicit = WEASELMUSE(window_sizes=[0.1, 0.3, 0.5, 0.7, 0.9], sparse=False)
    default = WEASELMUSE(sparse=False)
    np.testing.assert_array_equal(
        default.fit(X_multi, y_multi).transform(X_multi),
        explicit.fit(X_multi, y_multi).transform(X_multi),
    )
