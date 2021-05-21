.. _contribute:

==================
Contributing guide
==================

Thank you for considering contributing to pyts. Contributions from anyone
are welcomed. There are many ways to contribute to the package, such as
reporting bugs, adding new features and improving the documentation. The
following sections give more details on how to contribute.

**Important links**:

- The project is hosted on GitHub: https://github.com/johannfaouzi/pyts
- The documentation is hosted on Read the Docs: https://pyts.readthedocs.io/en/latest/


Submitting a bug report or a feature request
--------------------------------------------

If you experience a bug using pyts or if you would like to see a new
feature being added to the package, feel free to open an issue on GitHub:
https://github.com/johannfaouzi/pyts/issues

Bug report
^^^^^^^^^^

A good bug report usually contains:

- a description of the bug,
- a self-contained example to reproduce the bug if applicable,
- a description of the difference between the actual and expected results,
- the versions of the dependencies of pyts.

The last point can easily be done with the following commands::

    import numpy; print("NumPy", numpy.__version__)
    import scipy; print("SciPy", scipy.__version__)
    import sklearn; print("Scikit-Learn", sklearn.__version__)
    import numba; print("Numba", numba.__version__)
    import pyts; print("Pyts", pyts.__version__)

These guidelines make reproducing the bug easier, which make fixing it easier.


Feature request
^^^^^^^^^^^^^^^

A good feature request usually contains:

- a description of the requested feature,
- a description of the relevance of this feature to time series classification,
- references if applicable, with links to the papers if they are in open access.

This makes reviewing the relevance of the requested feature easier.


Contributing code
-----------------

In order to contribute code, you need to create a pull request on
GitHub: https://github.com/johannfaouzi/pyts/pulls

How to contribute
^^^^^^^^^^^^^^^^^

To contribute to pyts, you need to fork the repository then submit a
pull request:

1. Fork the repository.

2. Clone your fork of the pyts repository from your GitHub account to your
   local disk::

     git clone https://github.com/yourusername/pyts.git
     cd pyts

   where ``yourusername`` is your GitHub username.

3. Install the development dependencies::

      pip install pytest flake8

4. Install pyts in editable mode::

      pip install -e .

5. Add the ``upstream`` remote. It creates a reference to the main repository
   that can be used to keep your repository synchronized with the latest changes
   on the main repository::

      git remote add upstream https://github.com/johannfaouzi/pyts.git

6. Fetch the ``upstream`` remote then create a new branch where you will make
   your changes and switch to it::

      git fetch upstream
      git checkout -b my-feature upstream/main

   where ``my-feature`` is the name of your new branch (it's good practice to have
   an explicit name). You can now start making changes.

7. Make the changes that you want on your new branch on your new local machine.
   When you are done, add the changed files using ``git add`` and then
   ``git commit``::

      git add modified_files
      git commit

   Then push your commits to your GitHub account using ``git push``::

      git push origin my-feature

8. Create a pull request from your work. The base fork is the fork you
   would like to merge changes into, that is ``johannfaouzi/pyts`` on the
   ``main`` branch. The head fork is the repository where you made your
   changes, that is ``yourusername/pyts`` on the ``my-feature`` branch.
   Add a title and a description of your pull request, then click on
   **Create Pull Request**.


Pull request checklist
^^^^^^^^^^^^^^^^^^^^^^

Before pushing to your GitHub account, there are a few rules that are
usually worth complying with.

- **Make sure that your code passes tests**. You can do this by running the
  whole test suite with the ``pytest`` command. If you are experienced with
  ``pytest``, you can run specific tests that are relevant for your changes.
  It is still worth it running the whole test suite when you are done making
  changes since it does not take very long.
  For more information, please refer to the
  `pytest documentation <http://doc.pytest.org/en/latest/usage.html>`_.
  If your code does not pass tests but you are looking for help, feel free
  to do so (but mention it in your pull request).

- **Make sure to add tests if you add new code**. It is important to test
  new code to make sure that it behaves as expected. Ideally code coverage
  should increase with any new pull request. You can check code coverage
  using ``pytest-cov``::

    pip install pytest-cov
    pytest --cov pyts

- **Make sure that the documentation renders properly**. To build the
  documentation, please refer to the :ref:`contribute_documentation` guidelines.

- **Make sure that your PR does not add PEP8 violations**. On a Unix-like
  system, you can run ``make flake8-diff`` to only test the modified code.
  On any platform, you can run ``flake8`` to test the whole package, but it
  is better to only fix PEP8 violations that are related to your changes.
  Feel free to submit another pull request if you find other PEP8 violations.

.. _contribute_documentation:

Contributing to the documentation
---------------------------------

Documentation is as important as code. If you see typos, find docstrings
unclear or want to add examples illustrating functionalities provided in
pyts, feel free to open an issue to report it or a pull request if you
want to fix it.


Building the documentation
^^^^^^^^^^^^^^^^^^^^^^^^^^

Building the documentation requires installing some additional packages::

    pip install docutils=0.14 sphinx==1.8.5 sphinx-gallery numpydoc matplotlib

To build the documentation, you must be in the ``doc`` folder::

    cd doc

To generate the website with the example gallery, run the following command::

    make html

The documentation will be generated in the ``_build/html``. You can double
click on ``index.html`` to open the index page, which will look like
the first page that you see on the online documentation. Then you can move to
the pages that you modified and have a look at your changes.

Finally, repeat this process until you are satisfied with your changes and
open a pull request describing the changes you made.
