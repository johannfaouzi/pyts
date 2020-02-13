.. _classification:

=================================
Classification of raw time series
=================================

.. currentmodule:: pyts.classification

Algorithms that can directly classify time series have been developed.
The following sections will describe the ones that are available in pyts.
They can be found in the :mod:`pyts.classification` module.


KNeighborsClassifier
--------------------

The k-nearest neighbors algorithm is a relatively simple algorithm.
:class:`KNeighborsClassifier` finds the k nearest neighbors of a time series
and the predicted class is determined with majority voting. A key parameter
of this algorithm is the ``metric`` used to find the nearest neighbors.
A popular metric for time series is the Dynamic Time Warping metric
(see :ref:`metrics`).
The one-nearest-neighbor algorithm with this metric can be considered as
a good baseline for time series classification::

    >>> from pyts.classification import KNeighborsClassifier
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = KNeighborsClassifier(metric='dtw')
    >>> clf.fit(X_train, y_train) # doctest: +ELLIPSIS
    KNeighborsClassifier(...)
    >>> clf.score(X_test, y_test)
    0.91...


.. topic:: References

  * `Wikipedia entry on the k-nearest neighbors algorithm <https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm>`_


SAX-VSM
-------

SAX-VSM stands for **S**\ ymbolic **A**\ ggregate appro\ **X**\ imation in
**V**\ ector **S**\ pace **M**\ odel.
:class:`SAXVSM` is an algorithm based on the SAX representation of time
series in a vector space model. It first transforms a time series of floats
into a sequence of letters using the :ref:`approximation_sax` algorithm.
Then each sequence of letters is transformed into a bag of words using a sliding
window. Finally, a term-frequency inverse-term-frequency (tf-idf) vector is computed
for each class. Predictions are made using the cosine similarity between
the time series and the tf-idf vectors for each class. The predicted class
is the class yielding the highest cosine similarity.

.. figure:: ../auto_examples/classification/images/sphx_glr_plot_saxvsm_001.png
   :target: ../auto_examples/classification/plot_saxvsm.html
   :align: center
   :scale: 60%

.. code-block:: python

    >>> from pyts.classification import SAXVSM
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = SAXVSM(window_size=34, sublinear_tf=False, use_idf=False)
    >>> clf.fit(X_train, y_train) # doctest: +ELLIPSIS
    SAXVSM(...)
    >>> clf.score(X_test, y_test)
    0.76

.. topic:: References

   * P. Senin, and S. Malinchik, "SAX-VSM: Interpretable Time Series
     Classification Using SAX and Vector Space Model". International
     Conference on Data Mining, 13, 1175-1180 (2013).


BOSSVS
------

BOSSVS stands for **B**\ ag of **S**\ ymbolic **F**\ ourier **S**\ ymbols in
**V**\ ector **S**\ pace.
:class:`BOSSVS` is another bag-of-words approach for time series classification.
:class:`BOSSVS` is relatively similar to SAX-VSM: it builds a term-frequency
inverse-term-frequency vector for each class, but the symbols used to create
the words are generated with the :ref:`approximation_sfa` algorithm.

.. figure:: ../auto_examples/classification/images/sphx_glr_plot_bossvs_001.png
   :target: ../auto_examples/classification/plot_bossvs.html
   :align: center
   :scale: 60%

.. code-block:: python

    >>> from pyts.classification import BOSSVS
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = BOSSVS(window_size=28)
    >>> clf.fit(X_train, y_train) # doctest: +ELLIPSIS
    BOSSVS(...)
    >>> clf.score(X_test, y_test)
    0.98

.. topic:: References

  * P. Schäfer, "Scalable Time Series Classification". Data Mining and
    Knowledge Discovery, 30(5), 1273-1298 (2016).


LearningShapelets
-----------------

:class:`LearningShapelets` is a shapelet-based classifier.
A shapelet is defined as a contiguous subsequence of a time series.
The distance between a shapelet and a time series is defined as the minimum
of the distances between this shapelet and all the shapelets of identical
length extracted from this time series.
This estimator consists of two steps: computing the distances between the
shapelets and the time series, then computing a logistic regression using
these distances as features. This algorithm learns the shapelets as well as
the coefficients of the logistic regression.

.. figure:: ../auto_examples/classification/images/sphx_glr_plot_learning_shapelets_001.png
   :target: ../auto_examples/classification/plot_learning_shapelets.html
   :align: center
   :scale: 60%

.. code-block:: python

    >>> from pyts.classification import LearningShapelets
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = LearningShapelets(random_state=42, tol=0.01)
    >>> clf.fit(X_train, y_train)
    LearningShapelets(...)
    >>> clf.score(X_test, y_test)
    0.766...

.. topic:: References

  * J. Grabocka, N. Schilling, M. Wistuba and L. Schmidt-Thieme,
    "Learning Time-Series Shapelets". International Conference on Data
    Mining, 14, 392-401 (2014).
