.. _changelog:

==========
Change Log
==========

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
