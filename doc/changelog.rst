.. _changelog:

==========
Change Log
==========

Version 0.14.1
--------------

- Fix :meth:`pyts.classification.LearningShapelets.decision_function`, which
  dropped the sample axis for a single sample in binary classification
  (by Raashish Aggarwal).

- Fix :class:`pyts.decomposition.SingularSpectrumAnalysis`:

  + validate ``groups`` against the resolved window size when
    ``window_size`` is a float (by Anton Soloviev);
  + accept ``groups`` given as a NumPy array (by Anton Soloviev);
  + skip the memory check when ``chunksize`` is set, which raised a
    ``MemoryError`` even when the chunked transformation would fit
    (by sean0407).

- Fix :class:`pyts.classification.KNeighborsClassifier` with
  ``metric='dtw_sakoechiba'`` or ``metric='dtw_itakura'``, which ignored the
  other entries of ``metric_params`` (by Anton Soloviev).

- Fix :class:`pyts.approximation.MultipleCoefficientBinning` with
  ``strategy='quantile'``, which checked for equal consecutive quantiles
  across timestamps instead of within each timestamp (by LI).

- Convert numeric inputs (e.g. float32, float16, narrow or unsigned integers)
  to the dtypes supported by the compiled numba kernels, instead of raising an
  error:

  + :class:`pyts.image.RecurrencePlot` (by LI);
  + the lower bounds in :mod:`pyts.metrics` (by Anton Soloviev);
  + integer ``weights`` in
    :class:`pyts.multivariate.classification.MultivariateClassifier`
    (by Anton Soloviev);
  + :class:`pyts.approximation.PiecewiseAggregateApproximation` and
    :class:`pyts.image.GramianAngularField` (by Anton Soloviev);
  + :class:`pyts.classification.LearningShapelets`, array-like ``bins`` in
    :class:`pyts.classification.TSBF`, ``region`` in
    :func:`pyts.metrics.lower_bound_keogh` and
    :func:`pyts.metrics.lower_bound_improved`, ``window_sizes`` and
    ``window_steps`` in :class:`pyts.transformation.ShapeletTransform`, and
    uint64 inputs in :class:`pyts.image.RecurrencePlot`.

- Fix wrong results:

  + constrained DTW (``'sakoechiba'``, ``'itakura'``, ``'multiscale'`` and
    ``'fast'`` methods of :func:`pyts.metrics.dtw`) when the first time
    series is longer than the second, which often gave infinite distances;
  + ``sample_weight``, which was ignored by
    :class:`pyts.classification.LearningShapelets`;
  + the windows of :class:`pyts.classification.TimeSeriesForest`, which never
    started at the last valid position and failed when ``min_window_size``
    is the length of the time series (results change for a given
    ``random_state``);
  + estimators listed in
    :class:`pyts.multivariate.classification.MultivariateClassifier` and
    :class:`pyts.multivariate.transformation.MultivariateTransformer`, which
    were not cloned, so that the same instance listed several times was only
    fitted on the last feature;
  + words in :class:`pyts.transformation.BOSS`,
    :class:`pyts.transformation.WEASEL`,
    :class:`pyts.transformation.BagOfPatterns`,
    :class:`pyts.classification.BOSSVS` and
    :class:`pyts.classification.SAXVSM`, where one-symbol words, symbols
    that only differ by case and symbols such as ``'+'`` were dropped or
    merged;
  + shapelets with an undefined F-score, which were preferred by
    :class:`pyts.transformation.ShapeletTransform` with
    ``criterion='anova'``;
  + the first Fourier coefficient, which is now dropped by
    :class:`pyts.approximation.SymbolicFourierApproximation` (and BOSS) when
    ``norm_mean=True`` and ``anova=False``, as documented;
  + integer inputs of :class:`pyts.multivariate.transformation.WEASELMUSE`,
    whose derivatives overflowed for small or unsigned integers
    (by Anton Soloviev);
  + the automatic groups of
    :class:`pyts.decomposition.SingularSpectrumAnalysis`, which could share
    a component when ``lower_frequency_bound`` is large;
  + multi-character symbols, which were truncated by
    :class:`pyts.bag_of_words.BagOfWords` when ``threshold_std`` is set;
  + the parameters returned by :func:`pyts.datasets.make_cylinder_bell_funnel`
    with ``return_params=True``, which were not shuffled with the samples.

- Make scikit-learn recognize the classifiers of pyts as classifiers, so that
  for instance cross-validation uses stratified folds by default.

- Support inputs that used to fail: ``method='region'`` in
  :func:`pyts.metrics.dtw`, time series of length 1 (and of length 2 with
  the Itakura parallelogram), a single segment in
  :class:`pyts.approximation.PiecewiseAggregateApproximation` and
  :class:`pyts.image.GramianAngularField`, alphabets given as NumPy arrays,
  small-scale data with ``anova=True`` in
  :class:`pyts.approximation.DiscreteFourierTransform`, NumPy integer
  ``groups`` in :class:`pyts.decomposition.SingularSpectrumAnalysis`,
  variable-length datasets (whose training and test sets now have the same
  length) and legacy UEA datasets without a description file.

- Raise clear errors for invalid inputs: a different number of features (or
  of timestamps) than during ``fit``, ``weights`` of the wrong shape,
  ``alphabet='ordinal'`` in the estimators that build words, unknown
  parameters of :class:`pyts.classification.KNeighborsClassifier`, kernel
  sizes of 1, a non-positive number of kernels or too short time series in
  :class:`pyts.transformation.ROCKET`, and boundary values of several
  parameters. :class:`pyts.transformation.WEASEL` and WEASELMUSE also no
  longer modify ``window_sizes`` when they are created.

Version 0.14.0
--------------

- Add support for Python 3.12, 3.13 and 3.14, and drop support for Python
  3.8, 3.9 and 3.10 (all past end-of-life).

- Update the minimal versions required of the dependencies to support
  current releases of the scientific Python stack:

  * NumPy (>= 1.24.0)
  * SciPy (>= 1.15.0, for ``scipy.interpolate.make_interp_spline`` to build
    an even-degree spline from exactly the minimum number of points without
    explicit knots)
  * Scikit-Learn (>= 1.6.0)
  * Joblib (>= 1.3.0)
  * Numba (>= 0.60.0, for NumPy 2 support)

- Update the codebase for compatibility with NumPy 2.x, recent scikit-learn
  releases and recent numba releases:

  + Replace the removed ``force_all_finite`` parameter of
    ``sklearn.utils.check_array`` with its replacement,
    ``ensure_all_finite`` (renamed in scikit-learn 1.6, removed thereafter),
    in :class:`pyts.preprocessing.InterpolationImputer`,
    :class:`pyts.preprocessing.PowerTransformer` and :func:`pyts.metrics.dtw`.

  + Fix :func:`pyts.metrics.show_options`, whose internal docstring parser
    assumed a fixed indentation that no longer matches the docstring
    whitespace produced by the Python 3.13+ compiler.

  + Update the :class:`pyts.preprocessing.QuantileTransformer` doctest
    example to check the output's ``shape`` instead of printing the array
    itself, since NumPy's ``repr`` of large truncated arrays (and in
    particular whether it appends a ``shape=(...)`` suffix) is not stable
    across the supported NumPy version range.

  + Remove a ``distutils`` import from the Sphinx configuration
    (``doc/conf.py``); ``distutils`` was removed from the standard library
    in Python 3.12.

- Migrate packaging from ``setup.py``/``setup.cfg`` to a PEP 621
  ``pyproject.toml``, and replace the Azure Pipelines CI configuration
  with GitHub Actions.

Version 0.13.0
--------------

- Add support for Python 3.10 and 3.11, and drop support for Python 3.7.

- Update the minimal versions required of the dependencies:
  * NumPy (>= 1.22.4)
  * SciPy (>= 1.8.1)
  * Scikit-Learn (>=1.2.0)
  * Joblib (>=1.1.1)
  * Numba (>=0.55.2)

- Add an example illustrating time series clustering using
  :class:`pyts.transformation.BOSS` transformation with different metrics
  (by Lucas Plagwitz).

- Add automatic components-grouping in the *Singular Spectrum Analysis*
  for trend-seasonal decomposition with suitable example (by Lucas Plagwitz).

- Add two new parameters in :class:`pyts.decomposition.SingularSpectrumAnalysis`:
  ``chunksize`` allows for computing the decomposition of all the input time
  series using chunks (it should be a bit slower but use less memory), and
  ``n_jobs`` allows for running the decomposition of each chunk in parallel.

- Set the number of initiations of K-means to compute the initial shapelets
  in :class:`pyts.classification.LearningShapelets`: to 10 (to prevent a change
  of the default value in scikit-learn).

- Replace ``base_estimator_`` attribute with ``estimator_`` in
  :class:`pyts.classification.TimeSeriesForest` and
  :class:`pyts.classification.TSBF` (to match the changes made in scikit-learn).

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
