"""Testing for the UCR utility functions."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import os
import re

import pytest

import pyts
from pyts.datasets import fetch_ucr_dataset, ucr_dataset_info, ucr_dataset_list
from pyts.datasets._ucr import (
    _correct_ucr_name_description,
    _correct_ucr_name_download,
    _load_ucr_dataset,
)

# The small set of UCR datasets bundled as package data (see
# pyproject.toml's [tool.setuptools.package-data]), used to test
# 'fetch_ucr_dataset's use_cache=True code path offline. Not pyts's
# download cache (see pyts.datasets._cache._default_data_home) -- these
# ship with pyts itself and are unrelated to where new downloads land.
_BUNDLED_UCR_DIR = os.path.join(
    os.path.dirname(pyts.__file__), 'datasets', 'cached_datasets', 'UCR'
)


@pytest.mark.parametrize(
    'dataset, err_msg',
    [
        (
            'Hey',
            "Hey is not a valid name. The list of available names can be obtained "
            "by calling the 'pyts.datasets.ucr_dataset_list' function.",
        ),
        (
            ['Hey', 'ACSF1'],
            "The following names are not valid: ['Hey']. The list of available "
            "names can be obtained by calling the "
            "'pyts.datasets.ucr_dataset_list' function.",
        ),
        (
            ['Hey', 'Hi', 'ACSF1'],
            "The following names are not valid: ['Hey' 'Hi']. The list of "
            "available names can be obtained by calling the "
            "'pyts.datasets.ucr_dataset_list' function.",
        ),
    ],
)
def test_parameter_check_uea_dataset_info(dataset, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        ucr_dataset_info(dataset)


def test_type_check_ucr_dataset_info():
    """Test that an unsupported 'dataset' type raises a TypeError."""
    with pytest.raises(TypeError, match="'dataset' must be"):
        ucr_dataset_info(42)


@pytest.mark.parametrize(
    'dataset, length_expected',
    [(None, 128), ('CBF', 5), (['CBF'], 1), (['CBF', 'Beef'], 2)],
)
def test_dictionary_length_uea_dataset_info(dataset, length_expected):
    """Test that the length of the dictionart is the expected one."""
    assert len(ucr_dataset_info(dataset)) == length_expected


@pytest.mark.parametrize(
    'dataset', [None, 'Computers', ['Computers'], ['Computers', 'Chinatown']]
)
def test_dictionary_keys_ucr_dataset_info(dataset):
    """Test that the length of the dictionart is the expected one."""
    keys_expected = [
        'n_classes',
        'n_timestamps',
        'test_size',
        'train_size',
        'type',
    ]
    dictionary = ucr_dataset_info(dataset)
    if 'train_size' in dictionary.keys():
        assert sorted(list(dictionary.keys())) == keys_expected
    else:
        for key in dictionary.keys():
            assert sorted(list(dictionary[key].keys())) == keys_expected


def test_length_ucr_dataset_list():
    """Test that the length of the list is equal to the number of datasets."""
    assert len(ucr_dataset_list()) == 128


@pytest.mark.parametrize(
    'dataset, output',
    [
        ('CinCECGtorso', 'CinCECGTorso'),
        ('MixedShapes', 'MixedShapesRegularTrain'),
        ('NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax1'),
        ('NonInvasiveFetalECGThorax2', 'NonInvasiveFetalECGThorax2'),
        ('StarlightCurves', 'StarLightCurves'),
        ('Hey', 'Hey'),
        ('Hello World', 'Hello World'),
    ],
)
def test_correct_ucr_name_download(dataset, output):
    """Test that the results are the expected ones."""
    assert _correct_ucr_name_download(dataset) == output


@pytest.mark.parametrize(
    'dataset, output',
    [
        ('CinCECGTorso', 'CinCECGtorso'),
        ('MixedShapesRegularTrain', 'MixedShapes'),
        ('NonInvasiveFetalECGThorax1', 'NonInvasiveFetalECGThorax1'),
        ('NonInvasiveFetalECGThorax2', 'NonInvasiveFetalECGThorax2'),
        ('StarLightCurves', 'StarlightCurves'),
        ('Hey', 'Hey'),
        ('Hello World', 'Hello World'),
    ],
)
def test_correct_ucr_name_description(dataset, output):
    """Test that the results are the expected ones."""
    assert _correct_ucr_name_description(dataset) == output


def test_fetch_cached_ucr_dataset():
    """Test that a cached dataset can be loaded using 'fetch_ucr_dataset'."""
    res = fetch_ucr_dataset('GunPoint', use_cache=True, data_home=_BUNDLED_UCR_DIR)
    assert res.data_train.shape == (50, 150)
    assert res.data_test.shape == (150, 150)
    assert res.target_train.shape == (50,)
    assert res.target_test.shape == (150,)


def test_load_ucr_dataset_description_url(tmp_path):
    """The description URL must use the name 'description.php' expects.

    ``_load_ucr_dataset`` is always called with the download-corrected name
    (see ``_correct_ucr_name_download``), which differs from the name the
    'description.php' page on timeseriesclassification.com expects for
    'CinCECGtorso', 'MixedShapes' and 'StarlightCurves'. Using the download
    name there 401s, so the URL must be built from the description name
    instead (see ``_correct_ucr_name_description``).
    """
    download_name = 'MixedShapesRegularTrain'
    dataset_dir = tmp_path / download_name
    dataset_dir.mkdir()
    (dataset_dir / f'{download_name}.txt').write_text('fake description')
    (dataset_dir / f'{download_name}_TRAIN.txt').write_text('0 1.0 2.0\n1 3.0 4.0\n')
    (dataset_dir / f'{download_name}_TEST.txt').write_text('0 1.0 2.0\n1 3.0 4.0\n')

    res = _load_ucr_dataset(download_name, str(tmp_path))
    assert res.url == (
        "https://timeseriesclassification.com/description.php?Dataset=MixedShapes"
    )
