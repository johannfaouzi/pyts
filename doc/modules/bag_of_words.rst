.. _bag_of_words:

============================
Bag of words for time series
============================

.. currentmodule:: pyts.bag_of_words

Several algorithms for time series classification are based on bag-of-words
approaches: a sequence of symbols is transformed into a bag of words.
Utilities to derive bag of words are provided in the :mod:`pyts.bag_of_words`
module.


Bag of words
------------

:class:`BagOfWords` extracts words from a sequence of symbols. This sequence
of symbols is usually a discretized time series or discretized Fourier
coefficients of a time series. Words are extracted using a sliding window
that can be controlled with the ``window_size`` and ``window_step`` parameters.
The ``numerosity_reduction`` parameter controls the removal of all but one
occurrence of identical consecutive words. The impact of this parameter is illustrated
in the following example: when a time series has low variation over several
time points, the discretized time series is constant, which leads to several
identical consecutive words. Removed words are almost transparent.

.. figure:: ../auto_examples/bag_of_words/images/sphx_glr_plot_bow_001.png
   :target: ../auto_examples/bag_of_words/plot_bow.html
   :align: center
   :scale: 50%

.. code-block:: python

    >>> from pyts.bag_of_words import BagOfWords
    >>> X = [['a', 'a', 'b', 'a', 'b', 'b', 'b', 'b', 'a'],
    ...      ['a', 'b', 'c', 'c', 'c', 'c', 'a', 'a', 'c']]
    >>> bow = BagOfWords(window_size=2)
    >>> print(bow.transform(X))
    ['aa ab ba ab bb ba' 'ab bc cc ca aa ac']
    >>> bow = BagOfWords(window_size=2, numerosity_reduction=False)
    >>> print(bow.transform(X))
    ['aa ab ba ab bb bb bb ba' 'ab bc cc cc cc ca aa ac']
