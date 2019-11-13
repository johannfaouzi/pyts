"""Testing for the UEA utility functions."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import pytest
import re
from pyts.datasets import fetch_uea_dataset, uea_dataset_info, uea_dataset_list
from pyts.datasets.uea import _correct_uea_name_download


@pytest.mark.parametrize(
    'dataset, err_msg',
    [('Hey',
      "Hey is not a valid name. The list of available names can be obtained "
      "by calling the 'pyts.datasets.uea_dataset_list' function."),

     (['Hey', 'ArticularyWordRecognition'],
      "The following names are not valid: ['Hey']. The list of available "
      "names can be obtained by calling the "
      "'pyts.datasets.uea_dataset_list' function."),

     (['Hey', 'Hi', 'ArticularyWordRecognition'],
      "The following names are not valid: ['Hey' 'Hi']. The list of "
      "available names can be obtained by calling the "
      "'pyts.datasets.uea_dataset_list' function.")]
)
def test_parameter_check_uea_dataset_info(dataset, err_msg):
    """Test parameter validation."""
    with pytest.raises(ValueError, match=re.escape(err_msg)):
        uea_dataset_info(dataset)


@pytest.mark.parametrize(
    'dataset, length_expected',
    [(None, 30),
     ('ArticularyWordRecognition', 5),
     (['ArticularyWordRecognition'], 1),
     (['ArticularyWordRecognition', 'Epilepsy'], 2)]
)
def test_dictionary_length_uea_dataset_info(dataset, length_expected):
    """Test that the length of the dictionart is the expected one."""
    assert len(uea_dataset_info(dataset)) == length_expected


@pytest.mark.parametrize(
    'dataset',
    [None,
     'ArticularyWordRecognition',
     ['ArticularyWordRecognition'],
     ['ArticularyWordRecognition', 'Epilepsy']]
)
def test_dictionary_keys_uea_dataset_info(dataset):
    """Test that the length of the dictionart is the expected one."""
    keys_expected = ['n_classes', 'n_timestamps', 'test_size', 'train_size',
                     'type']
    dictionary = uea_dataset_info(dataset)
    if 'train_size' in dictionary.keys():
        assert sorted(list(dictionary.keys())) == keys_expected
    else:
        for key in dictionary.keys():
            assert sorted(list(dictionary[key].keys())) == keys_expected


def test_length_uea_dataset_list():
    """Test that the length of the list is equal to the number of datasets."""
    assert len(uea_dataset_list()) == 30


@pytest.mark.parametrize(
    'dataset, output',
    [('Ering', 'ERing'),
     ('AtrialFibrillation', 'AtrialFibrillation'),
     ('BasicMotions', 'BasicMotions')]
)
def test_correct_uea_name_download(dataset, output):
    """Test that the results are the expected ones."""
    assert _correct_uea_name_download(dataset) == output


def test_fetch_cached_uea_dataset():
    """Test that a cached dataset can be loaded using 'fetch_uea_dataset'."""
    res = fetch_uea_dataset('BasicMotions', use_cache=True)
    assert res.data_train.shape == (40, 6, 100)
    assert res.data_test.shape == (40, 6, 100)
    assert res.target_train.shape == (40,)
    assert res.target_test.shape == (40,)
