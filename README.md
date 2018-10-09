Welcome to pyts, a Python package for time series transformation and classification !
=========================================================================================
[![Build Status](https://travis-ci.org/johannfaouzi/pyts.svg?branch=dev)](https://travis-ci.org/johannfaouzi/pyts)
[![codecov](https://codecov.io/gh/johannfaouzi/pyts/branch/master/graph/badge.svg)](https://codecov.io/gh/johannfaouzi/pyts)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/pyts.svg)](https://img.shields.io/pypi/pyversions/pyts.svg)
[![PyPI version](https://badge.fury.io/py/pyts.svg)](https://badge.fury.io/py/pyts)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1244152.svg)](https://doi.org/10.5281/zenodo.1244152)

pyts is a Python package for time series transformation and classification. It
aims to provide state-of-the-art as well as recently published algorithms
for time series classification. Most of these algorithms transform time series,
thus pyts provides several tools to perform these transformations.

# Installation

## Dependencies

pyts has been tested on Python 2.7 and 3.5 with the following dependencies:

- numpy (>= 1.13.3)
- scipy (>= 0.13.3)
- scikit-learn (>=0.17.0)
- future (>=0.13.0) (for Python 2 compatibility)

To run the examples matplotlib is required (matplotlib >= 2.0.0 has
been tested).

## User installation

If you already have a working installation of numpy, scipy and
scikit-learn, you can easily install pyts using ``pip``

    pip install pyts

You can also get the latest version of pyts by cloning the repository

    git clone https://github.com/johannfaouzi/pyts.git
    cd pyts
    pip install .


# Documentation

For more information about the algorithms implemented in pyts as well as
how to use them, you can have a look at the
[HTML documentation](https://johannfaouzi.github.io/pyts/)

# Citation

pyts is registered on [Zenodo](https://doi.org/10.5281/zenodo.1244152).
If you use it in a scientific publication, please cite us

    @misc{johann_faouzi_2018_1244152,
    author       = {Johann Faouzi},
    title        = {{pyts: a Python package for time series transformation and classification}},
    month        = may,
    year         = 2018,
    doi          = {10.5281/zenodo.1244152},
    url          = {https://doi.org/10.5281/zenodo.1244152}
		}
