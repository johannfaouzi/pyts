.. _introduction:

============
Introduction
============

Introduction
------------

A time series is a sequence of values indexed in time order. Given their
nature, they are very common in many real-world applications. With the high
availability of sensors and the development of *Internet of things* devices,
the amount of time series data and the number of applications is continuously
increasing. Traditional domains using this type of data include finance and
econometrics, and these domains have recently been joined by smart grid,
earthquake prediction and weather forecasting.

One specific analysis is time series classification: given a time series and
a set of classes, one would like to classify this time series. Real-world
problems include disease detection using electrocardiogram data, household
device classification to reduce carbon footprint, and image classification.
Standard machine learning classification is not always well suited for time
series because of the possibly high correlation between back-to-back time
points. One typical example is the Naive Bayes algorithm, which assumes a
conditional independence between each feature given the class. For this reason,
algorithms dedicated to time series classification have been developed.

As the Python programming language is becoming more and more popular in
the fields of machine learning and data science, the objective of the **pyts**
Python package is to make time series classification easily accessible by
providing preprocessing and utility tools, and implementations of
several algorithms for time series classification.


Mathematical formulation
------------------------

A time series is defined as an ordered sequence :math:`(x_1,\ldots,x_n)`.
There are two kinds of time series: univariate time series and multivariate
time series.
Univariate time series have one single feature, that is
:math:`\forall i, x_i \in \mathbb{R}`.
Multivariate time series have several features, that is
:math:`\forall i, x_i \in \mathbb{R}^d`, where :math:`d` is
the number of features.

Note that the term *features* has a different meaning
for time series than in standard machine learning: it refers to the different
components of a given time series. For instance, a sensor of a GPS navigation
device will output a multivariate time series with two features: one feature
for the latitude coordinates and one feature for the longitude coordinates.

Most of the literature is focused on univariate time series classification,
therefore most of this package is also focused on univariate time series.
Nonetheless, we provide tools for multivariate time series in the
:mod:`pyts.multivariate` module.

A single label :math:`y` is associated with a time series. The objective is to
predict this label given a time series.


Challenge: lengths of time series
---------------------------------

One important challenge with time series is their number of time points.
A dataset of **equal-length** time series consists of time series that all have
the same number of time points. A dataset of **varying-length** time series
consists of time series that may have different numbers of time points.

For computational efficiency, **most algorithms implemented in pyts can only
deal with datasets of equal-length time series**. One exception is the
:func:`pyts.metrics.dtw` function that computes the Dynamic Time Warping
score between two time series that may have different lengths.
We will try to extend most implementations to datasets of varying-length time
series while preserving computational efficiency in the near future.


Notations
---------

In pyts, we use the NumPy package and more specifically the
`numpy.ndarray <https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.html>`_ class
to represent data.

Input data
^^^^^^^^^^

The input of a dataset of univariate time series is represented as a
two-dimensional array with shape ``(n_samples, n_timestamps)``, where the
first axis represents the samples and the second axis represents time.

The input of a dataset of multivariate time series is represented as a
three-dimensional array with shape ``(n_samples, n_features, n_timestamps)``,
where the first axis represents the samples, the second axis represents the
features, and the third axis represents time.

The name of this variable is usually ``X``, ``X_train``, or ``X_test`` if
cross-validation splits have already been performed.

Output data
^^^^^^^^^^^

The set of labels is always represented as a one-dimensional array with shape
``(n_samples,)``. The name of this variable is usually ``y``, ``y_train``, or
``y_test`` if cross-validation splits have already been performed.


.. topic:: References

    * Travis E, Oliphant. A guide to NumPy, USA: Trelgol Publishing, (2006).
