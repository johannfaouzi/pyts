.. _changelog:

==========
Change Log
==========

Version 0.13.0
--------------

- Add automatic components-grouping in the *Singular Spectrum Analysis*
  for trend-seasonal decomposition with suitable example (by Lucas Plagwitz).

- Add two new parameters in :class:`pyts.decomposition.SingularSpectrumAnalysis`:
  ``chunksize`` allows for computing the decomposition of all the input time
  series using chunks (it should be a bit slower but use less memory), and
  ``n_jobs`` allows for running the decomposition of each chunk in parallel.


Version 0.12.0
--------------

- Add support for Python 3.9 and drop support for Python 3.6.

- Add the *Time Series Forest* algorithm implemented as
  :class:`pyts.classification.TimeSeriesForest`.

- Add the *Time Series Bag-of-Features* algorithm implemented as
  :class:`pyts.classification.TSBF`.

- Replace ``scikit-learn`` mixin classes with ``pyts`` mixin classes to have
  standardized docstrings.

- Update the examples in the **Imaging time series** section of the gallery of
  examples.

- Remove some constraints when discretizing time series (number of bins, time
  series with low variance) that impact the following classes:

  + :class:`pyts.preprocessing.KBinsDiscretizer`
  + :class:`pyts.approximation.SymbolicAggregateApproximation`
  + :class:`pyts.bag_of_words.BagOfWords`
  + :class:`pyts.classification.SAXVSM`

- Remove specific functions for the different variants of Dynamic Time Warping
  (all ``dtw_*`` functions), only the main :func:`pyts.metrics.dtw` is kept.


Version 0.11.0
--------------

- Add support for Python 3.8 and drop support for Python 3.5.

- Rework the *BagOfWords* algorithm to match the description of the algorithm
  in the original paper. The former version of *BagOfWords* is available
  as *WordExtractor* in the :mod:`pyts.bag_of_words` module.

- Update the *SAXVSM* classifier with the new version of *BagOfWords*.

- Add the *BagOfPatterns* algorithm in the :mod:`pyts.transformation` module.

- Add the *ROCKET* algorithm in the :mod:`pyts.transformation` module.

- Add the *LearningShapelets* algorithm in the :mod:`pyts.classification`
  module.

- Deprecated specific functions for Dynamic Time Warping (all ``dtw_*`` functions),
  only the main :func:`pyts.metrics.dtw` is kept.


Version 0.10.0
--------------

- Adapt DTW functions to compare time series with different lengths
  (by Hicham Janati)

- Add a ``precomputed_cost`` parameter in DTW variants that are compatible
  with a precomputed cost matrix, that is classical DTW and DTW with global
  constraint regions like Sakoe-Chiba band and Itakura parallelogram
  (by Hicham Janati)

- Add a new algorithm called *ShapeletTransform* in the :mod:`pyts.transformation`
  module.

- Add a new dependency, the *joblib* Python package, since it has been vendored
  from scikit-learn and it is used in ShapeletTransform.

- [DOC] Revamp documentation in most sections:

  * User guide is much more detailed
  * A *Scikit-learn compatibility* page has been added to highlight the compatibility
    of pyts estimators with scikit-learn tools like model selection and pipelines.
  * A *Reproducibility* page has been added to highlight the work done in the
    `pyts-repro <https://github.com/johannfaouzi/pyts-repro>`_ repository,
    where we compare the performance of our implementations to the literature.
  * A *Contributing guide* has been added.


Version 0.9.0
-------------

- Add `datasets` module with dataset loading utilities

- Add `multivariate` module with utilities for multivariate time series

- Revamp the tests using `pytest.mark.parametrize`

- Add an `Examples` section in most of the public functions and classes

- Require version 1.3.0 of scipy: this is required to load ARFF files
  with relational attributes using `scipy.io.arff.loadarff`


Version 0.8.0
-------------

- No more Python 2 support

- New package required: numba

- Updated required versions of packages

- Modification of the API:

  - `quantization` module merged in `approximation` and removed

  - `bow` module renamed `bag_of_words`

  - Fewer acronyms used for the names of the classes: if an algorithm has a name
    with three words or fewer, the whole name is used.

  - More preprocessing tools in `preprocessing` module

  - New module `metrics` with metrics specific to time series

- Improved tests using pytest tools

- Reworked documentation

- Updated continuous integration scripts

- More optimized code using numba
