.. _transformation:

====================================
Extracting features from time series
====================================

.. currentmodule:: pyts.transformation

Standard machine learning algorithms are not always well suited for raw
time series because they cannot capture the high correlation between
consecutive time points: treating time points as features may not be optimal.
Therefore, algorithms that extract features from time series have been
developed. These algorithms transforms a dataset of time series with shape
``(n_samples, n_timestamps)`` into a dataset of features with shape
``(n_samples, n_features)`` that can be used to fit a standard classifier.
They can be found in the :mod:`pyts.transformation` module.
The following sections describe the algorithms made available.


ShapeletTransform
-----------------

:class:`ShapeletTransform` is a shapelet-based approach to extract features.
A shapelet is defined as a contiguous subsequence of a time series.
The distance between a shapelet and a time series is defined as the minimum
of the distances between this shapelet and all the shapelets of identical
length extracted from this time series. :class:`ShapeletTransform` extracts
the ``n_shapelets`` most discriminative shapelets given a criterion (mutual
information or F-scores) from a dataset of time series when ``fit`` is called.
The indices of the selected shapelets are made available via the ``indices_``
attribute.

.. figure:: ../auto_examples/transformation/images/sphx_glr_plot_shapelet_transform_001.png
   :target: ../auto_examples/transformation/plot_shapelet_transform.html
   :align: center
   :scale: 80%

:class:`ShapeletTransform` derives the distances between the selected shapelets
and a dataset of time series when ``transform`` is called. ``fit_transform``
is an optimized version of ``fit`` followed by ``transform`` since the distances
between the shapelets and the time series must be computed when ``fit`` is
called::

    >>> from pyts.transformation import ShapeletTransform
    >>> X = [[0, 2, 3, 4, 3, 2, 1],
    ...      [0, 1, 3, 4, 3, 4, 5],
    ...      [2, 1, 0, 2, 1, 5, 4],
    ...      [1, 2, 2, 1, 0, 3, 5]]
    >>> y = [0, 0, 1, 1]
    >>> st = ShapeletTransform(n_shapelets=2, window_sizes=[3])
    >>> X_new = st.fit_transform(X, y)
    >>> X_new.shape()
    (4, 2)

.. topic:: References

    * J. Lines, L. M. Davis, J. Hills and A. Bagnall, "A Shapelet Transform for
      Time Series Classification". Data Mining and Knowledge Discovery,
      289-297 (2012).


BOSS
----

BOSS stands for **B**\ ag **O**\ f **S**\ ymbolic-Fourier-Approximation
**S**\ ymbols. :class:`BOSS` extracts words from time series using the
:ref:`approximation_sfa` algorithm and derives their frequencies for each time
series.

.. figure:: ../auto_examples/transformation/images/sphx_glr_plot_boss_001.png
   :target: ../auto_examples/transformation/plot_boss.html
   :align: center
   :scale: 80%

The ``vocabulary_`` attribute is a mapping from the feature indices to the
corresponding words::

    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.transformation import BOSS
    >>> X_train, X_test, _, _ = load_gunpoint(return_X_y=True)
    >>> boss = BOSS(word_size=2, n_bins=2, sparse=False)
    >>> boss.fit(X_train) # doctest: +ELLIPSIS
    BOSS(...)
    >>> sorted(boss.vocabulary_.values())
    ['aa', 'ab', 'ba', 'bb']
    >>> boss.transform(X_test) # doctest: +ELLIPSIS
    array(...)

.. topic:: References

    * P. Schäfer, "The BOSS is concerned with time series classification
      in the presence of noise". Data Mining and Knowledge Discovery,
      29(6), 1505-1530 (2015).

.. _transformation_weasel:

WEASEL
------

WEASEL stands for **W**\ ord **E**\ xtr\ **A**\ ction for time **SE**\ ries
c\ **L**\ assification. While :class:`BOSS` extracts words with a single sliding
window, :class:`WEASEL` extracts words with several sliding windows of different
sizes, and selects the most discriminative words according to the chi-squared
test. The ``vocabulary_`` attribute is a mapping from the feature indices to the
corresponding words.

.. figure:: ../auto_examples/transformation/images/sphx_glr_plot_weasel_001.png
   :target: ../auto_examples/transformation/plot_weasel.html
   :align: center
   :scale: 80%

For new input data, the frequencies of each selected word are derived::

    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.transformation import WEASEL
    >>> X_train, X_test, y_train, _ = load_gunpoint(return_X_y=True)
    >>> weasel = WEASEL(sparse=False)
    >>> weasel.fit(X_train, y_train)
    WEASEL(...)
    >>>len(weasel.vocabulary_)
    73
    >>> weasel.transform(X_test).shape
    (150, 73)

.. topic:: References

    * P. Schäfer, and U. Leser, "Fast and Accurate Time Series Classification
      with WEASEL". Conference on Information and Knowledge Management,
      637-646 (2017).
