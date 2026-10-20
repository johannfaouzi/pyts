"""Regression tests for dataset generation and local dataset loading."""

# License: BSD-3-Clause

import os
import shutil

import numpy as np
import pytest

import pyts
from pyts.datasets import (
    fetch_ucr_dataset,
    fetch_uea_dataset,
    make_cylinder_bell_funnel,
)

_BUNDLED_UEA_DIR = os.path.join(
    os.path.dirname(pyts.__file__), 'datasets', 'cached_datasets', 'UEA'
)


@pytest.mark.parametrize('shuffle', [True, False])
def test_cylinder_bell_funnel_params_aligned_with_samples(shuffle):
    """The returned parameters describe the sample at the same index.

    Outside of ``[a, b]``, each time series is its noise ``epsilon`` only.
    """
    X, _, params = make_cylinder_bell_funnel(
        n_samples=30, shuffle=shuffle, random_state=0, return_params=True
    )
    t = np.arange(1, 129)
    outside = (t < params['a']) | (t > params['b'])
    np.testing.assert_array_equal((X - params['epsilon'])[outside], 0)
    assert params['a'].shape == params['b'].shape == params['eta'].shape == (30, 1)


@pytest.mark.parametrize('shuffle', [True, False])
def test_cylinder_bell_funnel_samples_unchanged_by_return_params(shuffle):
    """Returning the parameters does not change the samples themselves."""
    X, y = make_cylinder_bell_funnel(n_samples=30, shuffle=shuffle, random_state=0)
    X_p, y_p, _ = make_cylinder_bell_funnel(
        n_samples=30, shuffle=shuffle, random_state=0, return_params=True
    )
    np.testing.assert_array_equal(X, X_p)
    np.testing.assert_array_equal(y, y_p)


def _write_ts(path, rows, n_channels=1):
    """Write a minimal ``.ts`` file, each row being (channels, label)."""
    lines = [
        '# A tiny variable-length dataset',
        '@problemName tiny',
        '@univariate ' + ('true' if n_channels == 1 else 'false'),
        '@equalLength false',
        '@classLabel true 0 1',
        '@data',
    ]
    for channels, label in rows:
        lines.append(
            ':'.join(','.join(str(v) for v in channel) for channel in channels)
            + f':{label}'
        )
    path.write_text('\n'.join(lines) + '\n')


def test_ucr_variable_length_splits_share_the_same_width(tmp_path):
    """Both splits are padded to the length of the longest series."""
    dataset_dir = tmp_path / 'PLAID'
    dataset_dir.mkdir()
    _write_ts(dataset_dir / 'PLAID_TRAIN.ts', [([[1, 2, 3, 4, 5]], 0), ([[1, 2]], 1)])
    _write_ts(dataset_dir / 'PLAID_TEST.ts', [([[1, 2, 3]], 0), ([[1]], 1)])
    res = fetch_ucr_dataset('PLAID', use_cache=True, data_home=str(tmp_path))
    assert res.data_train.shape == (2, 5)
    assert res.data_test.shape == (2, 5)
    np.testing.assert_array_equal(res.data_test[0], [1, 2, 3, np.nan, np.nan])
    np.testing.assert_array_equal(res.data_train[0], [1, 2, 3, 4, 5])


def test_ucr_padding_when_the_test_set_is_longer(tmp_path):
    """The training set is padded too when the longest series is in test."""
    dataset_dir = tmp_path / 'PLAID'
    dataset_dir.mkdir()
    _write_ts(dataset_dir / 'PLAID_TRAIN.ts', [([[1, 2]], 0), ([[3]], 1)])
    _write_ts(dataset_dir / 'PLAID_TEST.ts', [([[1, 2, 3, 4]], 0), ([[5]], 1)])
    res = fetch_ucr_dataset('PLAID', use_cache=True, data_home=str(tmp_path))
    assert res.data_train.shape == res.data_test.shape == (2, 4)
    np.testing.assert_array_equal(res.data_train[1], [3, np.nan, np.nan, np.nan])


def test_uea_variable_length_splits_share_the_same_width(tmp_path):
    """Both splits of a multivariate dataset share the same width."""
    dataset_dir = tmp_path / 'JapaneseVowels'
    dataset_dir.mkdir()
    _write_ts(
        dataset_dir / 'JapaneseVowels_TRAIN.ts',
        [([[1, 2, 3, 4], [5, 6, 7, 8]], 0), ([[1, 2], [3, 4]], 1)],
        n_channels=2,
    )
    _write_ts(
        dataset_dir / 'JapaneseVowels_TEST.ts',
        [([[1, 2], [3, 4]], 0), ([[1], [2]], 1)],
        n_channels=2,
    )
    res = fetch_uea_dataset('JapaneseVowels', use_cache=True, data_home=str(tmp_path))
    assert res.data_train.shape == (2, 2, 4)
    assert res.data_test.shape == (2, 2, 4)
    np.testing.assert_array_equal(res.data_test[1, 1], [2, np.nan, np.nan, np.nan])


def test_uea_legacy_arff_without_description(tmp_path):
    """A legacy ARFF dataset without a description file can be loaded."""
    dataset_dir = tmp_path / 'BasicMotions'
    dataset_dir.mkdir()
    for split in ('TRAIN', 'TEST'):
        name = f'BasicMotions_{split}.arff'
        shutil.copy(os.path.join(_BUNDLED_UEA_DIR, 'BasicMotions', name), dataset_dir)
    res = fetch_uea_dataset('BasicMotions', use_cache=True, data_home=str(tmp_path))
    assert res.DESCR is None
    assert res.data_train.shape == (40, 6, 100)
    assert res.data_test.shape == (40, 6, 100)
