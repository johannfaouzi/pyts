.. _reproducibility:

===============
Reproducibility
===============

**pyts** provides many algorithms for time series classification that have
been published in the literature. Alongside high code coverage, we want to
provide users confidence about our implementations of these algorithms.
To do so, we created another
`repository <https://github.com/johannfaouzi/pyts-repro>`_ where we compare
the performance of several algorithms using pyts with the performance published
in the original papers or on the
`UEA & UCR Time Series Classification Repository <http://www.timeseriesclassification.com>`_.
We summarize the results on this page. The scripts to generate these results
are notebooks that are made available on the
`repository <https://github.com/johannfaouzi/pyts-repro>`_.

**Note: Most algorithms have hyper-parameters that need to be fine-tuned for
each dataset. If the values of these hyper-parameters are not directly
available, a grid search is performed using the testing set. For each of those
algorithms, the accuracy reported in the pyts column is the minimum of the
accuracy reported in the article and the highest accuracy obtained with the
grid search (to avoid any overestimation of the performance of the algorithm
because of data leakage). The same grid searches as the ones presented in the
articles are usually not done for computational reasons and randomness.**


UEA & UCR Time Series Classification Repository
-----------------------------------------------

The `UEA & UCR Time Series Classification Repository <http://www.timeseriesclassification.com>`_
is an ongoing project to develop a comprehensive repository for research into
time series classification providing datasets as well as code and results for
many algorithms.


Datasets
--------

The datasets used are taken from this repository.
On this website, you can download the datasets (a password is required to
unzip the file, you can find it by reading the PDF or the PowerPoint).
Convenience functions are also provided in pyts to download a dataset from this
repository:

* Univariate time series dataset: :func:`pyts.datasets.fetch_ucr_dataset`,
* Multivariate time series dataset: :func:`pyts.datasets.fetch_uea_dataset`.

For computational reasons, the algorithms are only tested on the smallest
datasets. This way, anyone can run the scripts by themselves on a single
machine and verify the results. The selected datasets are presented in the
table below.

+-------------+------------------+-------+------+-------+--------+
| Type        | Name             | Train | Test | Class | Length |
+=============+==================+=======+======+=======+========+
| Image       | Adiac            | 390   | 391  | 37    | 176    |
+-------------+------------------+-------+------+-------+--------+
| ECG         | ECG200           | 100   | 100  | 2     | 96     |
+-------------+------------------+-------+------+-------+--------+
| Motion      | GunPoint         | 50    | 150  | 2     | 150    |
+-------------+------------------+-------+------+-------+--------+
| Image       | MiddlePhalanxTW  | 399   | 154  | 6     | 80     |
+-------------+------------------+-------+------+-------+--------+
| Sensor      | Plane            | 105   | 105  | 7     | 144    |
+-------------+------------------+-------+------+-------+--------+
| Simulated   | SyntheticControl | 300   | 300  | 6     | 60     |
+-------------+------------------+-------+------+-------+--------+


Results
-------

1NN classifier with several metrics
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The metrics used are:

* Euclidean Distance (ED),
* Dynamic Time Warping (DTW), and
* Dynamic Time Warping with a learned warping window (DTW(w)).

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/KNN.ipynb>`__

+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| Name             | ED (reported) | ED (pyts) | DTW (reported) | DTW (pyts) | DTW(w) (reported) | DTW(w) (pyts)  |
+==================+===============+===========+================+============+===================+================+
| Adiac            | 0.6113        | 0.6113    | 0.6036         | 0.6036     | 0.6087            | 0.6087         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| ECG200           | 0.8800        | 0.8800    | 0.7700         | 0.7700     | 0.8800            | 0.8800         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| GunPoint         | 0.9133        | 0.9133    | 0.9067         | 0.9067     | 0.9133            | 0.9133         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| MiddlePhalanxTW  | 0.5130        | 0.5130    | 0.5065         | 0.5065     | 0.5065            | 0.5065         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| Plane            | 0.9619        | 0.9619    | 1.0000         | 1.0000     | 1.0000            | 1.0000         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+
| SyntheticControl | 0.8800        | 0.8800    | 0.9933         | 0.9933     | 0.9833            | 0.9833         |
+------------------+---------------+-----------+----------------+------------+-------------------+----------------+


Bag-of-Patterns transformer followed by a 1NN classifier using Euclidean distance
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/Bag-of-Patterns.ipynb>`__

+------------------+----------------------------+------------------------+
| Name             | Bag-of-Patterns (reported) | Bag-of-Patterns (pyts) |
+==================+============================+========================+
| Adiac            | 0.5916                     | 0.614                  |
+------------------+----------------------------+------------------------+
| ECG200           | 0.7857                     | 0.786                  |
+------------------+----------------------------+------------------------+
| GunPoint         | 0.9703                     | 0.980                  |
+------------------+----------------------------+------------------------+
| MiddlePhalanxTW  | 0.4914                     | 0.474                  |
+------------------+----------------------------+------------------------+
| Plane            | 0.9871                     | 1.000                  |
+------------------+----------------------------+------------------------+
| SyntheticControl | 0.9258                     | 0.926                  |
+------------------+----------------------------+------------------------+


BOSS transformer followed by a 1NN classifier using the BOSS metric
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/BOSS.ipynb>`__

+------------------+-----------------+-------------+
| Name             | BOSS (reported) | BOSS (pyts) |
+==================+=================+=============+
| Adiac            | 0.765           | 0.752       |
+------------------+-----------------+-------------+
| ECG200           | 0.870           | 0.870       |
+------------------+-----------------+-------------+
| GunPoint         | 1.000           | 1.000       |
+------------------+-----------------+-------------+
| MiddlePhalanxTW  | 0.526           | 0.526       |
+------------------+-----------------+-------------+
| Plane            | 1.000           | 1.000       |
+------------------+-----------------+-------------+
| SyntheticControl | 0.967           | 0.963       |
+------------------+-----------------+-------------+


BOSSVS classifier
^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/BOSSVS.ipynb>`__

+------------------+-------------------+---------------+
| Name             | BOSSVS (reported) | BOSSVS (pyts) |
+==================+===================+===============+
| Adiac            | 0.698             | 0.698         |
+------------------+-------------------+---------------+
| ECG200           | 0.820             | 0.820         |
+------------------+-------------------+---------------+
| GunPoint         | 1.000             | 1.000         |
+------------------+-------------------+---------------+
| MiddlePhalanxTW  | 0.586             | 0.545         |
+------------------+-------------------+---------------+
| Plane            | Unreported        | 1.000         |
+------------------+-------------------+---------------+
| SyntheticControl | 0.960             | 0.960         |
+------------------+-------------------+---------------+


Learning-Shapelet classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/LearningShapelet.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | LearningShapelet (reported)  | LearningShapelet (pyts)  |
+==================+==============================+==========================+
| Adiac            | 0.5274                       | 0.537                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.8714                       | 0.860                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 0.9826                       | 0.973                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5403                       | 0.494                    |
+------------------+------------------------------+--------------------------+
| Plane            | 0.9948                       | 0.981                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.9946                       | 0.990                    |
+------------------+------------------------------+--------------------------+


ROCKET transformer followed by a Ridge Classifier with built-in cross-validation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/ROCKET.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | ROCKET (reported)            | ROCKET (pyts)            |
+==================+==============================+==========================+
| Adiac            | 0.7847                       | 0.808                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.9060                       | 0.850                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 1.0000                       | 0.987                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5558                       | 0.571                    |
+------------------+------------------------------+--------------------------+
| Plane            | 1.0000                       | 1.000                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.8733                       | 0.983                    |
+------------------+------------------------------+--------------------------+


SAXVSM classifier
^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/SAXVSM.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | SAXVSM (reported)            | SAXVSM (pyts)            |
+==================+==============================+==========================+
| Adiac            | 0.4574                       | 0.458                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.8354                       | 0.840                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 0.9930                       | 0.993                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5393                       | 0.545                    |
+------------------+------------------------------+--------------------------+
| Plane            | 0.9799                       | 0.981                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.8691                       | 0.869                    |
+------------------+------------------------------+--------------------------+


ShapeletTransform transformer followed by a Support Vector Machine with a linear kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/ShapeletTransform.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | ShapeletTransform (reported) | ShapeletTransform (pyts) |
+==================+==============================+==========================+
| Adiac            | 0.2379                       | 0.238                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.8402                       | 0.840                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 1.0000                       | 0.967                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5793                       | 0.579                    |
+------------------+------------------------------+--------------------------+
| Plane            | 1.0000                       | 1.000                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.8733                       | 0.873                    |
+------------------+------------------------------+--------------------------+


TimeSeriesForest classifier
^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/TimeSeriesForest.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | TimeSeriesForest (reported)  | TimeSeriesForest (pyts)  |
+==================+==============================+==========================+
| Adiac            | 0.7072                       | 0.706                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.8682                       | 0.880                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 0.9617                       | 0.969                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5770                       | 0.591                    |
+------------------+------------------------------+--------------------------+
| Plane            | 0.9941                       | 1.000                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.9903                       | 0.987                    |
+------------------+------------------------------+--------------------------+


TSBF classifier
^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/TSBF.ipynb>`__

+------------------+------------------------------+--------------------------+
| Name             | TSBF (reported)              | TSBF (pyts)              |
+==================+==============================+==========================+
| Adiac            | 0.7268                       | 0.703                    |
+------------------+------------------------------+--------------------------+
| ECG200           | 0.8468                       | 0.820                    |
+------------------+------------------------------+--------------------------+
| GunPoint         | 0.9645                       | 0.967                    |
+------------------+------------------------------+--------------------------+
| MiddlePhalanxTW  | 0.5682                       | 0.558                    |
+------------------+------------------------------+--------------------------+
| Plane            | 0.9932                       | 1.000                    |
+------------------+------------------------------+--------------------------+
| SyntheticControl | 0.9865                       | 0.993                    |
+------------------+------------------------------+--------------------------+


WEASEL transformer followed by a logistic regression
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Link to the notebook <https://github.com/johannfaouzi/pyts-repro/blob/master/0.13.0/WEASEL.ipynb>`__

+------------------+-------------------+---------------+
| Name             | WEASEL (reported) | WEASEL (pyts) |
+==================+===================+===============+
| Adiac            | 0.8312            | 0.788         |
+------------------+-------------------+---------------+
| ECG200           | 0.8500            | 0.850         |
+------------------+-------------------+---------------+
| GunPoint         | 1.0000            | 0.960         |
+------------------+-------------------+---------------+
| MiddlePhalanxTW  | 0.5390            | 0.539         |
+------------------+-------------------+---------------+
| Plane            | 1.0000            | 1.000         |
+------------------+-------------------+---------------+
| SyntheticControl | 0.9933            | 0.973         |
+------------------+-------------------+---------------+
