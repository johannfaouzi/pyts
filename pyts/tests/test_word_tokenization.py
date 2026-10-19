"""Words are counted as built, whatever the alphabet or word size."""

# License: BSD-3-Clause

import numpy as np
import pytest

from pyts.classification import BOSSVS, SAXVSM
from pyts.datasets import load_gunpoint
from pyts.transformation import BOSS, WEASEL, BagOfPatterns

X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)


def _output(estimator, alphabet=None, word_size=None):
    params = {}
    if alphabet is not None:
        params['alphabet'] = alphabet
    if word_size is not None:
        params['word_size'] = word_size
    estimator = estimator.set_params(**params)
    if isinstance(estimator, (BOSSVS, SAXVSM)):
        return estimator.fit(X_train, y_train).decision_function(X_test)
    if isinstance(estimator, WEASEL):
        return estimator.fit(X_train, y_train).transform(X_test)
    return estimator.fit(X_train).transform(X_test)


ESTIMATORS = [
    BOSS(word_size=2, n_bins=2, window_size=10, sparse=False),
    WEASEL(word_size=2, n_bins=2, window_sizes=[12, 36], sparse=False),
    BagOfPatterns(window_size=12, word_size=2, n_bins=2, sparse=False),
    BOSSVS(word_size=2, n_bins=2, window_size=28),
    SAXVSM(window_size=15, word_size=2, n_bins=2),
]
IDS = [type(estimator).__name__ for estimator in ESTIMATORS]


@pytest.mark.parametrize('estimator', ESTIMATORS, ids=IDS)
@pytest.mark.parametrize('alphabet', [['x', 'y'], ['a', 'A'], ['+', '-']])
def test_alphabet_symbols_do_not_change_the_result(estimator, alphabet):
    """Renaming the symbols, including case or punctuation, is harmless."""
    actual = _output(estimator, alphabet=alphabet)
    expected = _output(estimator, alphabet=['a', 'b'])
    if not isinstance(estimator, (BOSSVS, SAXVSM)):
        # The vocabulary is sorted, so renaming the symbols can permute the
        # columns of the document-term matrix: compare them as a multiset.
        actual, expected = sorted(map(tuple, actual.T)), sorted(map(tuple, expected.T))
    np.testing.assert_allclose(actual, expected)


@pytest.mark.parametrize('estimator', ESTIMATORS, ids=IDS)
def test_single_symbol_words(estimator):
    """Words of a single symbol are kept rather than silently dropped."""
    if isinstance(estimator, (BagOfPatterns, SAXVSM)):
        # These use PAA, which does not support a single segment yet.
        pytest.skip("PiecewiseAggregateApproximation needs at least 2 segments.")
    X_new = _output(estimator, word_size=1)
    assert X_new.shape[0] == X_test.shape[0]
    assert X_new.shape[1] > 0
