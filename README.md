[![Build Status](https://travis-ci.org/johannfaouzi/pyts.svg?branch=master)](https://travis-ci.org/johannfaouzi/pyts)
[![Build Status](https://img.shields.io/appveyor/ci/johannfaouzi/pyts/master.svg)](https://ci.appveyor.com/project/johannfaouzi/pyts)
[![Build Status](https://img.shields.io/circleci/project/:vcsType/johann.faouzi/pyts/master.svg)](https://circleci.com/gh/johannfaouzi/pyts)
[![Codecov](https://codecov.io/gh/johannfaouzi/pyts/branch/master/graph/badge.svg)](https://codecov.io/gh/johannfaouzi/pyts)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyts.svg)](https://img.shields.io/pypi/pyversions/pyts.svg)
[![PyPI version](https://badge.fury.io/py/pyts.svg)](https://badge.fury.io/py/pyts)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1244152.svg)](https://doi.org/10.5281/zenodo.1244152)

## pyts: a Python package for time series transformation and classification

pyts is a Python package for time series transformation and classification. It
aims to provide state-of-the-art as well as recently published algorithms
for time series classification. Most of these algorithms transform time series,
thus pyts provides several tools to perform these transformations.


### Installation

#### Dependencies

pyts requires:

- Python (>= 3.5)
- NumPy (>= 1.15.4)
- SciPy (>= 1.1.0)
- Scikit-Learn (>=0.20.1)
- Numba (>0.41.0)

To run the examples Matplotlib (>=2.0.0) is required.


#### User installation

If you already have a working installation of numpy, scipy, scikit-learn and
numba, you can easily install pyts using ``pip``

    pip install pyts

You can also get the latest version of pyts by cloning the repository

    git clone https://github.com/johannfaouzi/pyts.git
    cd pyts
    pip install .


#### Testing

After installation, you can launch the test suite from the source
directory using pytest:

    pytest


### Changelog

See the [changelog](https://johannfaouzi.github.io/pyts/changelog.html)
for a history of notable changes to pyts.

### Development

The development of this package is in line with the one of the scikit-learn
community. Therefore, you can refer to their
[Development Guide](https://scikit-learn.org/stable/developers/). A slight
difference is the use of Numba instead of Cython for optimization.

### Documentation

The section below gives some information about the implemented algorithms in pyts.
For more information, you can have a look at the
[HTML documentation available via ReadTheDocs](https://johannfaouzi.github.io/pyts/)

### Implemented features

pyts consists of the following modules:

- `approximation`: This module provides implementations of algorithms that
approximate time series. The available algorithms are
[Piecewise Aggregate Approximation]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation),
[Symbolic Aggregate approXimation]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation),
[Discrete Fourier Transform]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation),
[Multiple Coefficient Binning]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
and
[Symbolic Fourier Approximation]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
.

- `bag_of_words`: This module consists of a class
[BagOfWords]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
that transforms
time series into bags of words. This approach is quite common in time series
classification.

- `classification`: This module provides implementations of algorithms that
can classify time series. The available algorithms are
[kNN]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation),
[SAX-VSM]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
and
[BOSSVS]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
.

- `decomposition`: This module provides implementations of algorithms that
decompose a time series into several time series. The only available algorithm
is
[Singular Spectrum Analysis]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
.

- `image`: This module provides implementations of algorithms that transform
time series into images. The available algorithms are
[Recurrence Plot]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation),
[Gramian Angular Field]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
and
[Markov Transition Field]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
.

- `metrics`: This module provides implementations of metrics that are specific
to time series. The available metrics are
[Dynamic Time Warping]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
with several variants and the
[BOSS](https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
metric.

- `preprocessing`: This module provides most of the scikit-learn preprocessing
tools but applied sample-wise (i.e. to each time series independently) instead
of feature-wise, as well as an
[imputer]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
of missing values using interpolation.

- `transformation`: This module provides implementations of algorithms that
transform a data set of time series with shape `(n_samples, n_timestamps)` into
a data set with shape `(n_samples, n_features)`. The available algorithms are
[BOSS]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
and
[WEASEL]
(https://johannfaouzi.github.io/pyts/approximation.html#pyts.approximation.PiecewiseAggregateApproximation)
.

- `utils`: a simple module with utility functions.
