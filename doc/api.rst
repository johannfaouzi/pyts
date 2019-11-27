.. _api:

=================
API Documentation
=================

Full API documentation of the :code:`pyts` Python package.

:mod:`pyts.approximation`: Approximation algorithms
===================================================

.. automodule:: pyts.approximation
    :no-members:
    :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
   :toctree: generated/
   :template: class.rst

   approximation.PiecewiseAggregateApproximation
   approximation.SymbolicAggregateApproximation
   approximation.DiscreteFourierTransform
   approximation.MultipleCoefficientBinning
   approximation.SymbolicFourierApproximation


:mod:`pyts.bag_of_words`: Bag-of-words algorithms
=================================================

.. automodule:: pyts.bag_of_words
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  bag_of_words.BagOfWords


:mod:`pyts.classification`: Classification algorithms
=====================================================

.. automodule:: pyts.classification
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  classification.KNeighborsClassifier
  classification.SAXVSM
  classification.BOSSVS


:mod:`pyts.datasets`: Dataset loading utilities
===============================================

.. automodule:: pyts.datasets
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: function.rst

  datasets.fetch_ucr_dataset
  datasets.fetch_uea_dataset
  datasets.load_basic_motions
  datasets.load_coffee
  datasets.load_gunpoint
  datasets.load_pig_central_venous_pressure
  datasets.make_cylinder_bell_funnel
  datasets.ucr_dataset_info
  datasets.ucr_dataset_list
  datasets.uea_dataset_info
  datasets.uea_dataset_list


:mod:`pyts.decomposition`: Decomposition algorithms
===================================================

.. automodule:: pyts.decomposition
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  decomposition.SingularSpectrumAnalysis


:mod:`pyts.image`: Imaging algorithms
=====================================

.. automodule:: pyts.image
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  image.RecurrencePlot
  image.GramianAngularField
  image.MarkovTransitionField


:mod:`pyts.metrics`: Metrics
============================

.. automodule:: pyts.metrics
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: function.rst

  metrics.boss
  metrics.dtw
  metrics.dtw_classic
  metrics.dtw_region
  metrics.dtw_sakoechiba
  metrics.dtw_itakura
  metrics.dtw_multiscale
  metrics.dtw_fast
  metrics.itakura_parallelogram
  metrics.sakoe_chiba_band


:mod:`pyts.multivariate`: Multivariate time series tools
========================================================

.. automodule:: pyts.multivariate
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

Classification
--------------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  multivariate.classification.MultivariateClassifier

Image
-----

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  multivariate.image.JointRecurrencePlot

Transformation
--------------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  multivariate.transformation.MultivariateTransformer
  multivariate.transformation.WEASELMUSE

Utils
-----

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: function.rst

  multivariate.utils.check_3d_array


:mod:`pyts.preprocessing`: Preprocessing tools
==============================================

.. automodule:: pyts.preprocessing
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

Scaling
-------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  preprocessing.StandardScaler
  preprocessing.MinMaxScaler
  preprocessing.MaxAbsScaler
  preprocessing.RobustScaler

Transformation
--------------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  preprocessing.PowerTransformer
  preprocessing.QuantileTransformer

Discretizing
------------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  preprocessing.KBinsDiscretizer

Imputation
----------

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  preprocessing.InterpolationImputer


:mod:`pyts.transformation`: Transformation algorithms
=====================================================

.. automodule:: pyts.transformation
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: class.rst

  transformation.BOSS
  transformation.WEASEL


:mod:`pyts.utils`: Utility tools
================================

.. automodule:: pyts.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: pyts

.. autosummary::
  :toctree: generated/
  :template: function.rst

  utils.segmentation
  utils.windowed_view
