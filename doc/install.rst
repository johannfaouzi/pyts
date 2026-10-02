.. _install:

=====================================
Installation, testing and development
=====================================

Dependencies
------------

pyts requires:

- Python (>= 3.11, < 3.16)
- NumPy (>= 1.24.0)
- SciPy (>= 1.15.0)
- Scikit-Learn (>= 1.6.0)
- Joblib (>= 1.3.0)
- Numba (>= 0.60.0)

To run the examples, Matplotlib (>= 3.7) is required.


User installation
------------------

If you already have a working installation of numpy, scipy, scikit-learn,
joblib and numba, you can easily install pyts using ``pip``::

    pip install pyts

or ``conda`` via the ``conda-forge`` channel::

    conda install -c conda-forge pyts

You can also get the latest version of pyts by cloning the repository::

    git clone https://github.com/johannfaouzi/pyts.git
    cd pyts
    pip install .


Testing
-------

pyts uses `pytest <https://docs.pytest.org>`_ for testing. If you don't
already have it installed, you can get it (along with ``pytest-cov``) with
the ``tests`` extra::

    pip install "pyts[tests]"

After installation, you can launch the test suite from outside the source
directory::

    pytest pyts


Development
-----------

The development of this package is in line with the one of the scikit-learn
community. For more information about our contributing guidelines, please
refer to the :ref:`contribute` guide, which covers setting up an editable
installation with the development extras (``pip install -e ".[dev]"``),
running the linter and type checker (``ruff``, ``pyrefly``), and building
the documentation.
