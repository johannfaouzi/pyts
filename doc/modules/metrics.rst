.. _metrics:

=======================
Metrics for time series
=======================

.. currentmodule:: pyts.metrics

Traditional metrics like the Euclidean distance are not always well suited for
time series.
Imagine a person saying a word slowly and another person saying the same quickly,
and both persons are recorded. The output will be two unequal-length time series.
Since both persons pronounce the same word, we can expect a metric between
these two time series to be (close to) zero.

Specific metrics for time series have been developed, the most well-known
being the Dynamic Time Warping (DTW) metric. Several variants have then
been proposed because of downsides with the original formulation.

Metrics for time series can be found in the :mod:`pyts.metrics` module.


Classic Dynamic Time Warping
------------------------------

Given two time series :math:`X=(x_1,\ldots,x_n)` and :math:`Y=(y_1,\ldots,y_m)`,
the cost matrix :math:`C` is defined as the cost for each pair of values:

.. math::

    C_{i, j} = f(x_i, y_j), \quad
    \forall i \in \{1, \ldots, n\}, \forall j \in \{1, \ldots, m\}

where :math:`f` is the cost function.
:math:`f` is typically the squared difference: :math:`f(x,y)=(x-y)^2`.

A warping path is a sequence :math:`p=(p_1,\ldots,p_L)` such that:

- Value condition: :math:`p_l = (i_l, j_l) \in \{1, \ldots, n\} \times \{1, \ldots, m\}, \quad \forall l \in \{1, \ldots, L\}`
- Boundary condition: :math:`p_1 = (1, 1)` and :math:`p_L = (n, m)`
- Monotonicity and step-size condition: :math:`p_{l+1} - p_l \in \{(0, 1), (1, 0), (1, 1)\}, \quad \forall l \in \{1, \ldots, L-1\}`

The cost associated with a warping path, denoted :math:`C_p`, is the sum of
the elements of the cost matrix that belong to the warping path:

.. math::

    C_p(X, Y) = \sum_{l=1}^L C_{i_l, j_l}

The Dynamic Time Warping score is defined as the minimum of these costs
among all the warping paths:

.. math::

    DTW(X, Y) = \min_{p \in P} C_p(X, Y)

where :math:`P` is the set of warping paths. This score can be computed using
the accumulated matrix, denoted as :math:`D`, defined as:

.. math::

    D_{1, j} = \sum_{k=1}^j C_{1, k}, \quad \forall j ∈ \{1,\ldots,m\}

    D_{i, 1} = \sum_{k=1}^i C_{k, 1}, \quad \forall i ∈ \{1,\ldots,n\}

.. math::

    D_{i, j} = \min\{ D(i−1,j−1), D(i−1,j), D(i,j−1) \} + C_{i, j},
    \quad \forall i \in \{2, \ldots, n\}, \forall j \in \{2, \ldots, m\}

The last entry of the accumulated cost matrix is the Dynamic Time Warping score:

.. math::   DTW(X, Y) = D_{n, m}

The Dynamic Time Warping metric has two main downsides:

- Complexity: Computational complexity is :math:`O(nm)`.
- It does not obey the triangle inequality, which means that a brute search
  must be performed when using a k-nearest neighbors algorithm for instance.

Several variants have been developed to address both downsides. We will
describe the variants that address the first issue since the variants
addressing the other issue are works in progress.


Variants of Dynamic Time Warping
--------------------------------

The main idea is to reduce the set of warping paths in order to decrease
the complexity of the algorithm. To do so, a region constraint is used.
Two kinds of regions exist: global regions and adaptive regions.


Global regions
^^^^^^^^^^^^^^

Global regions are regions that do not depend on the values of the time series
:math:`X` and :math:`Y`, but only on their lengths.

The most well-known global constraint region is the Sakoe-Chiba band and is
implemented as :func:`sakoe_chiba_band` is characterized by a ``window_size``
parameter.
Indices that belong to the band are indices that are not too far away from
the diagonal, that is:

.. math::

    R = \{(i, j), |i - j| \leq r \}

where :math:`r` is the ``window_size``. The higher, the wider the region is.


.. figure:: ../auto_examples/metrics/images/sphx_glr_plot_sakoe_chiba_001.png
   :target: ../auto_examples/metrics/plot_sakoe_chiba.html
   :align: center
   :scale: 50%

Another popular global constraint is the Itakura parallelogram.
:func:`itakura_parallelogram` is characterized by the ``max_slope`` parameter,
which defines how far away two points can be. The higher, the wider the region is.


.. figure:: ../auto_examples/metrics/images/sphx_glr_plot_itakura_001.png
   :target: ../auto_examples/metrics/plot_itakura.html
   :align: center
   :scale: 50%


Adaptive regions
^^^^^^^^^^^^^^^^

Adaptive regions are regions that depend on the values of the time series
:math:`X` and :math:`Y`. The idea is to find a constraint region that is more
specific to both time series.

One approach is called *MultiscaleDTW*. The idea is to down-sample both
time series, so that they have fewer time points. The optimal path is computed
for the down-sampled time series, then projected in the original space.
This projection is the constraint region.

Another approach is called *FastDTW*. The idea is to repeat the process of
down-sampling and defining the projected optimal path as the constraint region
several times in a recursive fashion.

Implementations
---------------

The most convenient way to derive any Dynamic Time Warping score is to use
the :func:`dtw` function. :func:`dtw` has a ``method`` parameter that lets
you choose which variant to use (default is ``method='classic'``, which is
the classic DTW score). Options for each method can be provided with the
``options`` parameter.

.. figure:: ../auto_examples/metrics/images/sphx_glr_plot_dtw_001.png
   :target: ../auto_examples/metrics/plot_dtw.html
   :align: center
   :scale: 70%

.. code-block:: python

    >>> from pyts.metrics import dtw
    >>> x = [0, 1, 1]
    >>> y = [2, 0, 1]
    >>> dtw(x, y, method='sakoechiba', options={'window_size': 0.5})
    2.0


.. topic:: References

    * H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
      for spoken word recognition". IEEE Transactions on Acoustics,
      Speech, and Signal Processing, 26(1), 43-49 (1978).

    * F. Itakura, "Minimum prediction residual principle applied to
      speech recognition". IEEE Transactions on Acoustics,
      Speech, and Signal Processing, 23(1), 67–72 (1975).

    * M. Müller, H. Mattes and F. Kurth, "An efficient multiscale approach
      to audio synchronization". International Conference on Music
      Information Retrieval, 6(1), 192-197 (2006).

    * S. Salvador ans P. Chan, "FastDTW: Toward Accurate Dynamic Time
      Warping in Linear Time and Space". KDD Workshop on Mining Temporal
      and Sequential Data, 70–80 (2004).
