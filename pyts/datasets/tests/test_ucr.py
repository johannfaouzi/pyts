"""Testing for the UCR utility functions."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import pytest
import re
from pyts.datasets import fetch_ucr_dataset, ucr_dataset_info, ucr_dataset_list
from pyts.datasets.ucr import (_correct_ucr_name_description,
                               _correct_ucr_name_download)


@pytest.mark.parametrize(
    'dataset, err_msg',
    [('Hey',
      "Hey is not a valid name. The list of available names can be obtained "
      "by calling the 'pyts.datasets.ucr_dataset_list' function."),

     (['Hey', 'ACSF1'],
      "The following names are not valid: ['Hey']. The list of available "
      "names can be obtained by calling the "
      "'pyts.datasets.ucr_dataset_list' function."),

     (['Hey', 'Hi', 'ACSF1'],
      "The following names are not valid: ['Hey' 'Hi']. The list of "
      "available names can be obtained by calling the "
      "'pyts.datasets.ucr_dataset_list' function.")]
)
def test_parameter_check_uea_dataset_info(dataset, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        ucr_dataset_info(dataset)


@pytest.mark.parametrize(
    'dataset, length_expected',
    [(None, 128),
     ('CBF', 5),
     (['CBF'], 1),
     (['CBF', 'Beef'], 2)]
)
def test_dictionary_length_uea_dataset_info(dataset, length_expected):
    """Test that the length of the dictionart is the expected one."""
    assert len(ucr_dataset_info(dataset)) == length_expected


@pytest.mark.parametrize(
    'dataset',
    [None,
     'Computers',
     ['Computers'],
     ['Computers', 'Chinatown']]
)
def test_dictionary_keys_ucr_dataset_info(dataset):
    """Test that the length of the dictionart is the expected one."""
    keys_expected = ['n_classes', 'n_timestamps', 'test_size', 'train_size',
                     'type']
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
    [('CinCECGtorso', 'CinCECGTorso'),
     ('MixedShapes', 'MixedShapesRegularTrain'),
     ('NonInvasiveFetalECGThorax1', 'NonInvasiveFatalECGThorax1'),
     ('NonInvasiveFetalECGThorax2', 'NonInvasiveFatalECGThorax2'),
     ('StarlightCurves', 'StarLightCurves'),
     ('Hey', 'Hey'),
     ('Hello World', 'Hello World')]
)
def test_correct_ucr_name_download(dataset, output):
    """Test that the results are the expected ones."""
    assert _correct_ucr_name_download(dataset) == output


@pytest.mark.parametrize(
    'dataset, output',
    [('CinCECGTorso', 'CinCECGtorso'),
     ('MixedShapesRegularTrain', 'MixedShapes'),
     ('NonInvasiveFatalECGThorax1', 'NonInvasiveFetalECGThorax1'),
     ('NonInvasiveFatalECGThorax2', 'NonInvasiveFetalECGThorax2'),
     ('StarLightCurves', 'StarlightCurves'),
     ('Hey', 'Hey'),
     ('Hello World', 'Hello World')]
)
def test_correct_ucr_name_description(dataset, output):
    """Test that the results are the expected ones."""
    assert _correct_ucr_name_description(dataset) == output


def test_fetch_cached_ucr_dataset():
    """Test that a cached dataset can be loaded using 'fetch_ucr_dataset'."""
    res = fetch_ucr_dataset('GunPoint', use_cache=True)
    assert res.data_train.shape == (50, 150)
    assert res.data_test.shape == (150, 150)
    assert res.target_train.shape == (50,)
    assert res.target_test.shape == (150,)
