.. _decomposition:

=======================
Decomposing time series
=======================

.. currentmodule:: pyts.decomposition

Decomposing time series consists in extracting several time series from a
time series. These extracted time series can represent different components,
such as the trend, the seasonality or noise. These kinds of algorithms can be
found in the :mod:`pyts.decomposition` module.


Singular Spectrum Analysis
--------------------------

:class:`SingularSpectrumAnalysis` is an algorithm that decomposes a time
series :math:`X` of length :math:`n` into several time series :math:`X^j` of
length :math:`n` such that :math:`X = \sum_{j=1}^n X^j`. The smaller the index
:math:`j`, the more information about :math:`X` it contains. The higher the index
:math:`j`, the more noise it contains. Taking the first extracted time series
can be used as a preprocessing step to remove noise.

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ssa_001.png
   :target: ../auto_examples/image/plot_ssa.html
   :align: center
   :scale: 50%

.. code-block:: python

    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.decomposition import SingularSpectrumAnalysis
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = SingularSpectrumAnalysis(window_size=5)
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 5, 150)

.. topic:: References

    * N. Golyandina, and A. Zhigljavsky, "Singular Spectrum Analysis for
      Time Series". Springer-Verlag Berlin Heidelberg (2013).
