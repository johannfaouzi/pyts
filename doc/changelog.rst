.. _changelog:

==========
Change Log
==========


Version 0.10.0
-------------

- Adapt `pyts.metrics.dtw` to compare time series with different lengths (Hicham Janati)



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
