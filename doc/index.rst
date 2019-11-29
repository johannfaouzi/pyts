Welcome to pyts documentation!
==============================

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Getting Started

   install
   contribute

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Documentation

   user_guide
   api
   scikit_learn_compatibility

.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Tutorial - Examples

   auto_examples/index

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Additional Information

   reproducibility
   changelog


**pyts** is a Python package dedicated to time series classification.
It aims to make time series classification easily accessible by providing
preprocessing and utility tools, and implementations of several time series
classification algorithms.
The package comes up with many unit tests and continuous integration ensures
new code integration and backward compatibility.
The package is distributed under the 3-clause BSD license.


Minimal example
---------------

The following code snippet illustrates the basic usage of pyts:

    >>> from pyts.classification import BOSSVS
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = BOSSVS(window_size=28)
    >>> clf.fit(X_train, y_train)
    BOSSVS(...)
    >>> clf.score(X_test, y_test)
    0.98

1. First we import:

  - a class defining a classifier (``BOSSVS``),
  - a function that loads the *GunPoint* dataset (``load_gunpoint``).

2. Then we load the training and test sets by calling the ``load_gunpoint`` function.

3. Next we define a classifier by creating an instance of the class.

4. Finally we fit the classifier on the training set and evaluate its
   performance by computing the accuracy on the test set.

People familiar with scikit-learn API should feel comfortable with pyts as
its API is heavily inspired from it, and pyts estimators are compatible
with scikit-learn tools like model selection and pipelines. For more
information, please refer to the
`Scikit-learn compatibility <scikit_learn_compatibility.html>`_ page.


`Getting started <install.html>`_
---------------------------------

Information to install, test, and contribute to the package.

`User Guide <user_guide.html>`_
-------------------------------

The main documentation. This contains an in-depth description of all
algorithms and how to apply them.

`API Documentation <api.html>`_
-------------------------------

The exact API of all functions and classes, as given in the
docstrings. The API documents expected types and allowed features for
all functions, and all parameters available for the algorithms.

`Examples <auto_examples/index.html>`_
--------------------------------------

A set of examples illustrating the use of the different algorithms. It
complements the `User Guide <user_guide.html>`_.

`Changelog <changelog.html>`_
------------------------------

History of notable changes to the pyts.

See the `README <https://github.com/johannfaouzi/pyts/blob/master/README.md>`_
for more information.


Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
