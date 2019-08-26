"""Functions to load datasets."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

import os
from .uea import _load_uea_dataset
from .ucr import _load_ucr_dataset


def _load_dataset(name, archive, return_X_y):
    r"""Load and return dataset.

    Parameters
    ----------
    name : str
        Name of the dataset.

    archive : 'UCR' or 'UEA'
        Archive the dataset belongs to.

    return_X_y : bool
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

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

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    """  # noqa: E501
    module_path = os.path.dirname(__file__)
    folder = os.path.join(module_path, 'cached_datasets', archive, '')
    if archive == 'UCR':
        bunch = _load_ucr_dataset(name, folder)
    else:
        bunch = _load_uea_dataset(name, folder)
    if return_X_y:
        return (bunch.data_train, bunch.data_test,
                bunch.target_train, bunch.target_test)
    return bunch


def load_basic_motions(return_X_y=False):
    r"""Load and return the Basic Motions dataset.

    The data was generated as part of a student project where four students
    performed four activities whilst wearing a smart watch. The watch collects
    3D accelerometer and a 3D gyroscope It consists of four classes, which are
    walking, resting, running and badminton. Participants were required to
    record motion a total of five times, and the data is sampled once every
    tenth of a second, for a ten second period.

    ================   ==============
    Training samples               40
    Test samples                   40
    Dimensionality                  6
    Timestamps                    100
    Classes                         4
    ================   ==============

    Parameters
    ----------
    return_X_y : bool (default = False)
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

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

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    References
    ----------
    .. [1] `UCR archive entry for the PigCVP dataset
           <http://www.timeseriesclassification.com/description.php?Dataset=BasicMotions>`_

    Examples
    --------
    >>> from pyts.datasets import load_basic_motions
    >>> bunch = load_basic_motions()
    >>> bunch.data_train.shape
    (40, 6, 100)
    >>> X_train, X_test, y_train, y_test = load_basic_motions(return_X_y=True)
    >>> X_train.shape
    (40, 6, 100)

    """  # noqa: E501
    return _load_dataset('BasicMotions', 'UEA', return_X_y)


def load_coffee(return_X_y=False):
    r"""Load and return the Coffee dataset.

    Food spectrographs are used in chemometrics to classify food types, a task
    that has obvious applications in food safety and quality assurance. The
    coffee data set is a two class problem to distinguish between Robusta and
    Aribica coffee beans.

    ================   ==============
    Training samples               28
    Test samples                   28
    Timestamps                    286
    Classes                         2
    ================   ==============

    Parameters
    ----------
    return_X_y : bool (default = False)
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

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

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    References
    ----------
    .. [1] R. Briandet, E.K. Kemsley, and R.H. Wilson, "Discrimination of
           Arabica and Robusta in Instant Coffee by Fourier Transform Infrared
           Spectroscopy and Chemometrics". Journal of Agricultural and Food
           Chemistry (1996).

    .. [2] A. Bagnall, L. Davis, J. Hills and J. Lines, "Transformation Based
           Ensembles for Time Series Classification". SDM (2012).

    .. [3] `UCR archive entry for the PigCVP dataset
           <http://www.timeseriesclassification.com/description.php?Dataset=Coffee>`_

    Examples
    --------
    >>> from pyts.datasets import load_coffee
    >>> bunch = load_coffee()
    >>> bunch.data_train.shape
    (28, 286)
    >>> X_train, X_test, y_train, y_test = load_coffee(return_X_y=True)
    >>> X_train.shape
    (28, 286)

    """  # noqa: E501
    return _load_dataset('Coffee', 'UCR', return_X_y)


def load_gunpoint(return_X_y=False):
    r"""Load and return the GunPoint dataset.

    This dataset involves one female actor and one male actor making a motion
    with their hand. The two classes are: Gun-Draw and Point: For Gun-Draw the
    actors have their hands by their sides. They draw a replicate gun from a
    hip-mounted holster, point it at a target for approximately one second,
    then return the gun to the holster, and their hands to their sides. For
    Point the actors have their gun by their sides. They point with their index
    fingers to a target for approximately one second, and then return their
    hands to their sides. For both classes, we tracked the centroid of the
    actor's right hands in both X- and Y-axes, which appear to be highly
    correlated. The data in the archive is just the X-axis.

    ================   ==============
    Training samples               50
    Test samples                  150
    Timestamps                    150
    Classes                         2
    ================   ==============

    Parameters
    ----------
    return_X_y : bool (default = False)
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

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

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    References
    ----------
    .. [1] `UCR archive entry for the PigCVP dataset
           <http://www.timeseriesclassification.com/description.php?Dataset=GunPoint>`_

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> bunch = load_gunpoint()
    >>> bunch.data_train.shape
    (50, 150)
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> X_train.shape
    (50, 150)

    """  # noqa: E501
    return _load_dataset('GunPoint', 'UCR', return_X_y)


def load_pig_central_venous_pressure(return_X_y=False):
    r"""Load and return the PigCVP dataset.

    In the test set, a class is represented by four examples, the second and
    third 2000 data points of the before time series and the second and third
    2000 data points of the after time series. Data created by Mathieu
    Guillame-Bert et al. Data edited by Shaghayegh Gharghabi and Eamonn Keogh.

    ================   ==============
    Training samples              104
    Test samples                  208
    Timestamps                   2000
    Classes                        52
    ================   ==============

    Parameters
    ----------
    return_X_y : bool (default = False)
        If True, return
        ``(data_train, data_test, target_train, target_test)`` instead of a
        Bunch object.

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

    (data_train, data_test, target_train, target_test) : tuple if ``return_X_y`` is True

    References
    ----------
    .. [1] M. Guillame-Bert and A. Dubrawski, "Classification of Time Sequences
           using Graphs of Temporal Constraints". Journal of Machine Learning
           Research, 2017.

    .. [2] `UCR archive entry for the PigCVP dataset
           <http://www.timeseriesclassification.com/description.php?Dataset=PigCVP>`_

    Examples
    --------
    >>> from pyts.datasets import load_pig_central_venous_pressure
    >>> bunch = load_pig_central_venous_pressure()
    >>> bunch.data_train.shape
    (104, 2000)
    >>> X_train, X_test, y_train, y_test = load_pig_central_venous_pressure(
    ...    return_X_y=True)
    >>> X_train.shape
    (104, 2000)

    """  # noqa: E501
    return _load_dataset('PigCVP', 'UCR', return_X_y)
