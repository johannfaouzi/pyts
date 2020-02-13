.. _approximation:

=========================
Approximating time series
=========================

.. currentmodule:: pyts.approximation

Raw time series can be noisy or have a lot of time points. Approximating them
can be useful so that the most important information is kept while reducing
noise. The :mod:`pyts.approximation` module provides simple tools for
approximating time series.


.. _approximation_paa:

Piecewise Aggregate Approximation
---------------------------------

:class:`PiecewiseAggregateApproximation` (PAA) reduces the number of time points
of a time series using a sliding window and taking the mean value in this
window. It transforms a time series :math:`X=(x_1,\ldots,x_n)` into another
time series :math:`\tilde{X}=(\tilde{x}_1,\ldots,\tilde{x}_m)` with
:math:`m<n` such that, if :math:`m` divides :math:`n`, then

.. math::

    \tilde{x}_i = \frac{m}{n} \sum_{j={(n/m)\cdot (i-1)+1}}^{(n/m)\cdot i} x_j

If :math:`m` does not divides :math:`n`, two options are made available with
the ``overlapping`` parameter:

- If ``overlapping=True``, the length of the sliding window is constant and
  some windows are overlapping.
- If ``overlapping=False``, the length of the sliding window may vary and
  the windows are non-overlapping.


.. figure:: ../auto_examples/approximation/images/sphx_glr_plot_paa_001.png
   :target: ../auto_examples/approximation/plot_paa.html
   :align: center
   :scale: 70%

.. code-block:: python

    >>> from pyts.approximation import PiecewiseAggregateApproximation
    >>> X = [[0, 4, 2, 1, 7, 6, 3, 5],
    ...      [2, 5, 4, 5, 3, 4, 2, 3]]
    >>> transformer = PiecewiseAggregateApproximation(window_size=2)
    >>> transformer.transform(X)
    array([[2. , 1.5, 6.5, 4. ],
           [3.5, 4.5, 3.5, 2.5]])

.. topic:: References

    * E. Keogh, K. Chakrabarti, M. Pazzani, and S. Mehrotra,
      "Dimensionality reduction for fast similarity search in large
      time series databases". Knowledge and information Systems,
      3(3), 263-286 (2001).

.. _approximation_sax:

Symbolic Aggregate approXimation
--------------------------------

:class:`SymbolicAggregateApproximation` (SAX) reduces the dimension of
the feature space by discretizing each time series independently. Several
strategies can be used to derive the edges of the bins with the ``strategy``
parameter:

- ``strategy='uniform'``: all bins in each sample have identical widths,
- ``strategy='quantile'``: all bins in each sample have the same number of points,
- ``strategy='normal'``: bin edges are quantiles from a standard normal distribution.

.. figure:: ../auto_examples/approximation/images/sphx_glr_plot_sax_001.png
   :target: ../auto_examples/approximation/plot_sax.html
   :align: center
   :scale: 70%

.. code-block:: python

     >>> from pyts.approximation import SymbolicAggregateApproximation
     >>> X = [[0, 4, 2, 1, 7, 6, 3, 5],
     ...      [2, 5, 4, 5, 3, 4, 2, 3]]
     >>> transformer = SymbolicAggregateApproximation()
     >>> print(transformer.transform(X))
     [['a' 'c' 'b' 'a' 'd' 'd' 'b' 'c']
      ['a' 'd' 'c' 'd' 'b' 'c' 'a' 'b']]

.. topic:: References

    * J. Lin, E. Keogh, L. Wei, and S. Lonardi, "Experiencing SAX: a
      novel symbolic representation of time series". Data Mining and
      Knowledge Discovery, 15(2), 107-144 (2007).

.. _approximation_dft:

Discrete Fourier Transform
--------------------------

:class:`DiscreteFourierTransform` extracts Fourier coefficients from each
time series. The ``n_coefs`` parameter controls the number of Fourier
coefficients to keep.

.. figure:: ../auto_examples/approximation/images/sphx_glr_plot_dft_001.png
   :target: ../auto_examples/approximation/plot_dft.html
   :align: center
   :scale: 70%

.. code-block:: python

   >>> from pyts.approximation import DiscreteFourierTransform
   >>> from pyts.datasets import load_gunpoint
   >>> X, _, _, _ = load_gunpoint(return_X_y=True)
   >>> transformer = DiscreteFourierTransform(n_coefs=4)
   >>> X_new = transformer.fit_transform(X)
   >>> X_new.shape
   (50, 4)

.. topic:: References

    * P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
      and Index for Similarity Search in High Dimensional Datasets",
      International Conference on Extending Database Technology,
      15, 516-527 (2012).

.. _approximation_mcb:

Multiple Coefficient Binning
----------------------------

:class:`MultipleCoefficientBinning` is similar to :ref:`approximation_sax`,
but it discretizes each time point independently. Therefore, it is very close
to `sklearn.prepreocessing.KBinsDiscretizer <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html#sklearn-preprocessing-kbinsdiscretizer>`_
but the ``strategy`` parameter can take different values:

- 'uniform': all bins in each sample have identical widths,
- 'quantile': all bins in each sample have the same number of points,
- 'normal': bin edges are quantiles from a standard normal distribution,
- 'entropy': bin edges are computed using information gain.

.. figure:: ../auto_examples/approximation/images/sphx_glr_plot_mcb_001.png
   :target: ../auto_examples/approximation/plot_mcb.html
   :align: center
   :scale: 70%

.. code-block:: python

   >>> from pyts.approximation import MultipleCoefficientBinning
   >>> X = [[0, 4],
   ...      [2, 7],
   ...      [1, 6],
   ...      [3, 5]]
   >>> transformer = MultipleCoefficientBinning(n_bins=2)
   >>> print(transformer.fit_transform(X))
   [['a' 'a']
    ['b' 'b']
    ['a' 'b']
    ['b' 'a']]

.. topic:: References

    * P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
      and Index for Similarity Search in High Dimensional Datasets",
      International Conference on Extending Database Technology,
      15, 516-527 (2012).

.. _approximation_sfa:

Symbolic Fourier Approximation
------------------------------

:class:`SymbolicFourierApproximation` consists in combining
:ref:`approximation_dft` and :ref:`approximation_mcb`:

- First Fourier coefficients are derived using :class:`DiscreteFourierTransform`,
- Then Fourier coefficients are discretized using :class:`MultipleCoefficientBinning`.

.. topic:: References

    * P. Schäfer, and M. Högqvist, "SFA: A Symbolic Fourier Approximation
      and Index for Similarity Search in High Dimensional Datasets",
      International Conference on Extending Database Technology,
      15, 516-527 (2012)
