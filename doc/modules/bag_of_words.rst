.. _bag_of_words:

============================
Bag of words for time series
============================

.. currentmodule:: pyts.bag_of_words

Several algorithms for time series classification are based on bag-of-words
approaches: a sequence of symbols is transformed into a bag of words.
Utilities to derive bag of words are provided in the :mod:`pyts.bag_of_words`
module.

.. _bag_of_words_w:

Bag of words
------------

:class:`BagOfWords` extracts subseries using a sliding window, then transforms
each subseries into a word using the :ref:`approximation_paa` and
:ref:`approximation_sax` algorithms. Therefore :class:`BagOfWords` trasnforms
each time series into a bag of words. The sliding window can be controlled
with the ``window_size`` and ``window_step`` parameters. The length of
each word can be set with the ``word_size`` parameter, while the ``n_bins``
parameter controls the size of the alphabet to discretize time series.
The ``numerosity_reduction`` parameter controls the removal of all but one
occurrence of identical consecutive words.

.. figure:: ../auto_examples/bag_of_words/images/sphx_glr_plot_bow_001.png
   :target: ../auto_examples/bag_of_words/plot_bow.html
   :align: center
   :scale: 80%

.. code-block:: python

    >>> import numpy as np
    >>> from pyts.bag_of_words import BagOfWords
    >>> X = np.arange(12).reshape(2, 6)
    >>> bow = BagOfWords(window_size=4, word_size=4)
    >>> bow.transform(X)
    array(['abcd', 'abcd'], dtype='<U4')
    >>> bow.set_params(numerosity_reduction=False)
    BagOfWords(...)
    >>> bow.transform(X)
    array(['abcd abcd abcd', 'abcd abcd abcd'], dtype='<U14')
