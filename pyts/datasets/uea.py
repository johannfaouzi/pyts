"""
Utility functions for the UEA multivariate time series classification
archive.
"""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import numpy as np
import os
import pickle
from scipy.io.arff import loadarff
from sklearn.utils import Bunch
from urllib.request import urlretrieve
import zipfile


def _correct_uea_name_download(dataset):
    if dataset == 'Ering':
        return 'ERing'
    else:
        return dataset


def uea_dataset_list():
    """List of available UEA datasets.

    Returns
    -------
    datasets : list
        List of available datasets from the UEA Time Series
        Classification Archive.

    References
    ----------
    .. [1] `List of datasets on the UEA & UCR archive
           <http://www.timeseriesclassification.com/dataset.php>`_

    Examples
    --------
    >>> from pyts.datasets import uea_dataset_list
    >>> uea_dataset_list()[:3]
    ['ArticularyWordRecognition', 'AtrialFibrillation', 'BasicMotions']

    """
    module_path = os.path.dirname(__file__)
    finfo = os.path.join(module_path, 'info', 'uea.pickle')
    dictionary = pickle.load(open(finfo, 'rb'))
    datasets = sorted(dictionary.keys())
    return datasets


def uea_dataset_info(dataset=None):
    """Information about the UEA datasets.

    Parameters
    ----------
    dataset : str, list of str or None (default = None)
        The data sets for which the information will be returned.
        If None, the information for all the datasets is returned.

    Returns
    -------
    dictionary : dict
        Dictionary with the information for each dataset.

    References
    ----------
    .. [1] `List of datasets on the UEA & UCR archive
           <http://www.timeseriesclassification.com/dataset.php>`_

    Examples
    --------
    >>> from pyts.datasets import uea_dataset_info
    >>> uea_dataset_info('AtrialFibrillation')['n_timestamps']
    640

    """
    module_path = os.path.dirname(__file__)
    finfo = os.path.join(module_path, 'info', 'uea.pickle')
    dictionary = pickle.load(open(finfo, 'rb'))
    datasets = list(dictionary.keys())

    if dataset is None:
        return dictionary
    elif isinstance(dataset, str):
        if dataset not in datasets:
            raise ValueError(
                "{0} is not a valid name. The list of available names "
                "can be obtained by calling the "
                "'pyts.datasets.uea_dataset_list' function."
                .format(dataset)
            )
        else:
            return dictionary[dataset]
    elif isinstance(dataset, (list, tuple, np.ndarray)):
        dataset = np.asarray(dataset)
        invalid_datasets = np.setdiff1d(dataset, datasets)
        if invalid_datasets.size > 0:
            raise ValueError(
                "The following names are not valid: {0}. The list of "
                "available names can be obtained by calling the "
                "'pyts.datasets.uea_dataset_list' function."
                .format(invalid_datasets)
            )
        else:
            info = {}
            for data in dataset:
                info[data] = dictionary[data]
            return info


def fetch_uea_dataset(dataset, use_cache=True, data_home=None,
                      return_X_y=False):  # noqa 207
    """Fetch dataset from UEA TSC Archive by name.

    Fetched data sets are saved by default in the
    ``pyts/datasets/cached_datasets/UEA/`` folder. To avoid
    downloading the same data set several times, it is
    highly recommended not to change the default values
    of ``use_cache`` and ``path``.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    use_cache : bool (default = True)
        If True, look if the data set has already been fetched
        and load the fetched version if it is the case. If False,
        download the data set from the UCR Time Series Classification
        Archive.

    data_home : None or str (default = None)
        The path of the folder containing the cached data set.
        If None, the ``pyts.datasets.cached_datasets/UEA/`` folder is
        used. If the data set is not found, it is downloaded and cached
        in this path.

    return_X_y : bool (default = False)
        If True, returns ``(data_train, data_test, target_train, target_test)``
        instead of a Bunch object. See below for more information about the
        `data` and `target` object.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array of integers
            The classification labels in the training set.
        target_test : array of integers
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    (data_train, data_test, target_train, target_test) : tuple if \
``return_X_y`` is True

    Notes
    -----
    Missing values are represented as NaN's.

    References
    ----------
    .. [1] A. Bagnall et al, "The UEA multivariate time series
           classification archive, 2018". arXiv:1811.00075 [cs, stat],
           2018.

    .. [2] A. Bagnall et al, "The UEA & UCR Time Series Classification
           Repository", www.timeseriesclassification.com.

    """
    if dataset not in uea_dataset_list():
        raise ValueError(
            "{0} is not a valid name. The list of available names "
            "can be obtained with ``pyts.datasets.uea_dataset_list()``"
            .format(dataset)
        )
    if data_home is None:
        import pyts
        home = '/'.join(pyts.__file__.split('/')[:-2]) + '/'
        relative_path = 'pyts/datasets/cached_datasets/UEA/'
        path = home + relative_path
    else:
        path = data_home
    if not os.path.exists(path):
        os.makedirs(path)

    correct_dataset = _correct_uea_name_download(dataset)
    if use_cache and os.path.exists(path + correct_dataset):
        bunch = _load_uea_dataset(correct_dataset, path)
    else:
        url = ("http://www.timeseriesclassification.com/Downloads/{0}.zip"
               .format(correct_dataset))
        filename = 'temp_{}'.format(correct_dataset)
        _ = urlretrieve(url, path + filename)
        zipfile.ZipFile(path + filename).extractall(path + correct_dataset)
        os.remove(path + filename)
        bunch = _load_uea_dataset(correct_dataset, path)

    if return_X_y:
        return (bunch.data_train, bunch.data_test,
                bunch.target_train, bunch.target_test)
    return bunch


def _load_uea_dataset(dataset, path):
    """Load a UEA data set from a local folder.

    Parameters
    ----------
    dataset : str
        Name of the dataset.

    path : str
        The path of the folder containing the cached data set.

    Returns
    -------
    data : Bunch
        Dictionary-like object, with attributes:

        data_train : array of floats
            The time series in the training set.
        data_test : array of floats
            The time series in the test set.
        target_train : array
            The classification labels in the training set.
        target_test : array
            The classification labels in the test set.
        DESCR : str
            The full description of the dataset.
        url : str
            The url of the dataset.

    Notes
    -----
    Missing values are represented as NaN's.

    """
    new_path = path + dataset + '/'
    try:
        description_file = [
            file for file in os.listdir(new_path)
            if ('Description.txt' in file
                or dataset + '.txt' in file)
        ][0]
    except IndexError:
        description_file = None

    if description_file is not None:
        try:
            with(open(new_path + description_file, encoding='utf-8')) as f:
                description = f.read()
        except UnicodeDecodeError:
            with(open(new_path + description_file,
                      encoding='ISO-8859-1')) as f:
                description = f.read()
    else:
        description = None

    data_train = loadarff(new_path + dataset + '_TRAIN.arff')
    X_train, y_train = _parse_relational_arff(data_train)

    data_test = loadarff(new_path + dataset + '_TEST.arff')
    X_test, y_test = _parse_relational_arff(data_test)

    bunch = Bunch(
        data_train=X_train, target_train=y_train,
        data_test=X_test, target_test=y_test,
        DESCR=description,
        url=("http://www.timeseriesclassification.com/"
             "description.php?Dataset={}".format(dataset))
    )

    return bunch


def _parse_relational_arff(data):
    X_data = np.asarray(data[0])
    n_samples = len(X_data)
    X, y = [], []

    if X_data[0][0].dtype.names is None:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][name] for name in X_data[i].dtype.names]
            )
            X.append(X_sample.T)
            y.append(X_data[i][1])
    else:
        for i in range(n_samples):
            X_sample = np.asarray(
                [X_data[i][0][name] for name in X_data[i][0].dtype.names]
            )
            X.append(X_sample.T)
            y.append(X_data[i][1])

    X = np.asarray(X).astype('float64')
    y = np.asarray(y)

    try:
        y = y.astype('float64').astype('int64')
    except ValueError:
        pass

    return X, y
