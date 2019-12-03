.. _datasets:

=========================
Dataset loading utilities
=========================

.. currentmodule:: pyts.datasets

The `UEA & UCR Time Series Classification Repository <http://www.timeseriesclassification.com>`_
hosts a lot of datasets for time series classification. A few datasets are
available in the *pyts* repository itself, and functions to download the
other datasets are made available.


Simulated datasets
------------------

The :func:`make_cylinder_bell_funnel` function makes a synthetic dataset
of univariate time series with three classes: cylinder, bell and funnel.
This dataset was introduced by N. Saito in his PhD thesis
"Local feature extraction and its application using a library of bases".

The time series are generated from the following distributions:

.. math::

    c(t) = (6 + \eta) \cdot 1_{[a, b]}(t) + \epsilon(t)

    b(t) = (6 + \eta) \cdot 1_{[a, b]}(t) \cdot (t - a) / (b - a) +
    \epsilon(t)

    f(t) = (6 + \eta) \cdot 1_{[a, b]}(t) \cdot (b - t) / (b - a) +
    \epsilon(t)

where:

- :math:`t=1,\ldots,128`,
- :math:`a` is an integer-valued uniform random variable on the interval :math:`[16, 32]`,
- :math:`b-a` is an integer-valued uniform distribution on the interval :math:`[32, 96]`,
- :math:`\eta` and :math:`\epsilon(t)` are standard normal variables,
- :math:`{1}_{[a, b]}` is the characteristic function on the interval :math:`[a, b]`.

:math:`c`, :math:`b`, and :math:`f` stand for "cylinder", "bell", and "funnel" respectively.


Univariate time series: UCR repository
--------------------------------------

*pyts* comes with a copy of three univariate time series datasets:

- :func:`load_coffee`: load the *Coffee* dataset,
- :func:`load_gunpoint`: load the *GunPoint* dataset,
- :func:`load_pig_central_venous_pressure`: load the *Pig Central Venous Pressure* dataset.

The characteristics of these datasets are summarized in the following table:

+--------------+------------------+-------+------+-------+--------+
| Type         | Name             | Train | Test | Class | Length |
+==============+==================+=======+======+=======+========+
| SPECTRO      | Coffee           | 100   | 100  | 2     | 96     |
+--------------+------------------+-------+------+-------+--------+
| MOTION       | GunPoint         | 50    | 150  | 2     | 150    |
+--------------+------------------+-------+------+-------+--------+
| HEMODYNAMICS | PigCVP           | 104   | 208  | 52    | 2000   |
+--------------+------------------+-------+------+-------+--------+

Three functions are made available to fetch other datasets from this repository:

- :func:`ucr_dataset_list`: return the list of available datasets,
- :func:`ucr_dataset_info`: return a dictionary with the characteristics
  of each dataset,
- :func:`fetch_ucr_dataset`: fetch a dataset given its name.


Multivariate time series: UEA repository
----------------------------------------

*pyts* comes with a copy of one multivariate time series dataset:

- :func:`load_basic_motions`: load the *Basic Motions* dataset.

Three functions are made available to fetch other datasets from this repository:

- :func:`uea_dataset_list`: return the list of available datasets,
- :func:`uea_dataset_info`: return a dictionary with the characteristics
  of each dataset,
- :func:`fetch_uea_dataset`: fetch a dataset given its name.
