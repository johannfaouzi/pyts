.. _image:

===================
Imaging time series
===================

.. currentmodule:: pyts.image

Imaging time series, that is transforming time series into images, is another
popular transformation. One important upside of this transformation is retrieving
information for any pair of time points :math:`(x_i, x_j)` given a time series
:math:`(x_1, \ldots, x_n)`.
Deep neural networks, especially convolutional neural networks, have been
used to classify these imaged time series. While *pyts* does not provide
deep neural networks, it provides algorithms to transform time series into
images in the :mod:`pyts.image` module.

.. _image_rp:

Recurrence Plot
---------------

:class:`RecurrencePlot` extracts trajectories from time series and computes the
pairwise distances between these trajectories. The trajectories are defined
as:

.. math::

      \vec{x}_i = (x_i, x_{i + \tau}, \ldots, x_{i + (m - 1)\tau}), \quad
      \forall i \in \{1, \ldots, n - (m - 1)\tau \}

where :math:`m` is the ``dimension`` of the trajectories and :math:`\tau`
is the ``time_delay``. The recurrence plot, denoted :math:`R`, is the
binarized pairwise distance matrix between the trajectories:

.. math::

    R_{i, j} = \Theta(\varepsilon - \| \vec{x}_i - \vec{x}_j \|), \quad
    \forall i,j \in \{1, \ldots, n - (m - 1)\tau \}

where :math:`\Theta` is the Heaviside function and :math:`\varepsilon`
is the ``threshold``. Different strategies can be used to choose the threshold,
such as a given float or a quantile of the distances.

.. figure:: ../auto_examples/image/images/sphx_glr_plot_rp_001.png
   :target: ../auto_examples/image/plot_rp.html
   :align: center
   :scale: 80%

.. code-block:: python

     >>> from pyts.datasets import load_gunpoint
     >>> from pyts.image import RecurrencePlot
     >>> X, _, _, _ = load_gunpoint(return_X_y=True)
     >>> transformer = RecurrencePlot()
     >>> X_new = transformer.transform(X)
     >>> X_new.shape
     (50, 150, 150)

.. topic:: References

    * J.-P Eckmann, S. Oliffson Kamphorst and D Ruelle, "Recurrence
      Plots of Dynamical Systems". Europhysics Letters (1987).

    * N. Hatami, Y. Gavet and J. Debayle, "Classification of Time-Series Images
      Using Deep Convolutional Neural Networks". https://arxiv.org/abs/1710.00886


Gramian Angular Field
---------------------

:class:`GramianAngularField` creates a matrix of temporal correlations for each
:math:`(x_i, x_j)`. First it rescales the time series in a range :math:`[a, b]`
where :math:`-1 \leq a < b \leq 1`. Then it computes the polar coordinates of
the scaled time series by taking the :math:`arccos`. Finally it computes the
cosine of the sum of the angles for the Gramian Angular Summation Field
(GASF) or the sine of the difference of the angles for the Gramian Angular
Difference Field (GADF).

.. math::

    \tilde{x}_i = a + (b - a) \times \frac{x_i - \min(x)}{\max(x) - \min(x)},
    \quad \forall i \in \{1, \ldots, n\}

    \phi_i = \arccos(\tilde{x}_i), \quad \forall i \in \{1, \ldots, n\}

    GASF_{i, j} = \cos(\phi_i + \phi_j), \quad \forall i, j \in \{1, \ldots, n\}

    GADF_{i, j} = \sin(\phi_i - \phi_j), \quad \forall i, j \in \{1, \ldots, n\}

The ``method`` parameter controls which type of Gramian angular fields are
computed.

.. figure:: ../auto_examples/image/images/sphx_glr_plot_gaf_001.png
   :target: ../auto_examples/image/plot_gaf.html
   :align: center
   :scale: 80%

.. code-block:: python

    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.image import GramianAngularField
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = GramianAngularField()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 150, 150)

.. topic:: References

    * Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
      Inspection and Classification Using Tiled Convolutional Neural
      Networks." AAAI Workshop (2015).


Markov Transition Field
-----------------------

:class:`MarkovTransitionField` discretizes a time series into bins.
It then computes the Markov Transition Matrix of the discretized time series.
Finally it spreads out the transition matrix to a field in order to reduce
the loss of temporal information.

.. figure:: ../auto_examples/image/images/sphx_glr_plot_mtf_001.png
   :target: ../auto_examples/image/plot_mtf.html
   :align: center
   :scale: 80%

.. code-block:: python

    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.image import MarkovTransitionField
    >>> X, _, _, _ = load_gunpoint(return_X_y=True)
    >>> transformer = MarkovTransitionField()
    >>> X_new = transformer.transform(X)
    >>> X_new.shape
    (50, 150, 150)

.. topic:: References

    * Z. Wang and T. Oates, "Encoding Time Series as Images for Visual
      Inspection and Classification Using Tiled Convolutional Neural
      Networks." AAAI Workshop (2015).
