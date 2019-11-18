.. _scikit_learn_compatibility:

==========================
Scikit-learn compatibility
==========================

`Scikit-learn <https://scikit-learn.org/>`_ is a very popular Python package
for machine learning. If you are familiar with scikit-learn API, you should
feel comfortable with pyts API as it is heavily inspired from it. The following
sections illustrate the compatibility between pyts and scikit-learn.

Estimator API
-------------

pyts provides two types of estimators:

- *transformers*: estimators that transform the input data,
- *classifiers*: estimators that classify the input data.

These estimators have the same basic methods as the ones from scikit-learn:

- Transformers

  + ``fit``: fit the transformer,
  + ``transform``: transform the input data.

- Classifiers:

  + ``fit``: fit the classifier,
  + ``predict``: make predictions given the input data.


Compatibility with existing tools from scikit-learn
---------------------------------------------------

Scikit-learn provides a lot of utilities such as model selection and pipelines.
These tools are often used in machine learning. By having an API compatible
with scikit-learn API, we do not need to reimplement them, and can use them
directly. We will illustrate this compatibility with two popular modules from
scikit-learn:
`Model selection <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection>`_ and
`Pipeline <https://scikit-learn.org/stable/modules/classes.html#module-sklearn.pipeline>`_

Model selection
^^^^^^^^^^^^^^^

Model selection is a core concept of machine learning. With a wide range of
algorithms and several hyper-parameters for each algorithm, there needs a way
to select the best model. One popular approach is to perform cross validation
over a grid of possible values for each hyper-parameter.
The corresponding scikit-learn implementation is
`sklearn.model_selection.GridSearchCV <https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn-model-selection-gridsearchcv>`_.

We will illustrate the use of GridSearchCV with a classifier from pyts.
Let's say that we want to use the
`SAX-VSM <https://pyts.readthedocs.io/en/latest/generated/pyts.classification.SAXVSM.html#pyts-classification-saxvsm>`_
classifier and tune the value for two of its hyper-parameters:

- *window_size* : 0.3, 0.5 or 0.7

- *strategy*: 'quantile' or 'uniform'

We can define a GridSearchCV instance to find the best combination:

  >>> clf = GridSearchCV(
  ...     SAXVSM(),
  ...     {'window_size': (0.3, 0.5, 0.7), 'strategy': ('uniform', 'quantile')},
  ...     iid=False, cv=5
  ... )

Then we can simply:

- fit on the training set by calling ``clf.fit(X_train, y_train)``,

- derive predictions on the test set by calling ``clf.predict(X_test)``,

- directly evaluate the performance on the test set by calling ``clf.score(X_test, y_test)``.

Here is a self-contained example:

    >>> from pyts.classification import SAXVSM
    >>> from pyts.datasets import load_gunpoint
    >>> from sklearn.model_selection import GridSearchCV
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = GridSearchCV(
    ...     SAXVSM(),
    ...     {'window_size': (0.3, 0.5, 0.7), 'strategy': ('uniform', 'quantile')},
    ...     iid=False, cv=5
    ... )
    >>> clf.fit(X_train, y_train)
    # GridSearchCV(...)
    >>> clf.best_params_
    # {'strategy': 'uniform', 'window_size': 0.5}
    >>> clf.score(X_test, y_test)
    # 0.846...


Pipeline
^^^^^^^^

Transformers are usually combined with a classifier to build a composite
estimator. It is possible to build such an estimator in scikit-learn using
`sklearn.pipeline.Pipeline <https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html#sklearn.pipeline.Pipeline>`_.
You can use estimators from both pyts and scikit-learn to build your own
composite estimator to classify time series.

We will illustrate this functionality with the following example. Let's say
that we want to build a composite estimator with the following steps:

1. Standardization of each time series using
`pyts.preprocessing.StandardScaler <https://pyts.readthedocs.io/en/latest/generated/pyts.preprocessing.StandardScaler.html#pyts.preprocessing.StandardScaler>`_


2. Feature extraction using
`pyts.transformation.WEASEL <https://pyts.readthedocs.io/en/latest/generated/pyts.transformation.WEASEL.html#pyts-transformation-weasel>`_


3. Scaling of each feature using
`sklearn.preprocessing.MinMaxScaler <https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html#sklearn-preprocessing-minmaxscaler>`_


4. Classification using
`sklearn.ensemble.RandomForestClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html#sklearn-ensemble-randomforestclassifier>`_

We just have to create a Pipeline instance with these estimators:

    >>> clf = Pipeline([('scaler_1', StandardScaler()),
    ...                 ('boss', BOSS(sparse=False)),
    ...                 ('scaler_2', MinMaxScaler()),
    ...                 ('forest', RandomForestClassifier())])

Then we can simply:

- fit on the training set by calling ``clf.fit(X_train, y_train)``,

- derive predictions on the test set by calling ``clf.predict(X_test)``,

- directly evaluate the performance on the test set by calling ``clf.score(X_test, y_test)``.

Here is a self-contained example:

    >>> from pyts.datasets import load_pig_central_venous_pressure
    >>> from pyts.preprocessing import StandardScaler
    >>> from pyts.transformation import BOSS
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> X_train, X_test, y_train, y_test = load_pig_central_venous_pressure(return_X_y=True)
    >>> clf = Pipeline([('scaler_1', StandardScaler()),
    ...                 ('boss', BOSS(sparse=False)),
    ...                 ('scaler_2', MinMaxScaler()),
    ...                 ('forest', RandomForestClassifier(random_state=42))])
    >>> clf.fit(X_train, y_train)
    # Pipeline(...)
    >>> clf.score(X_test, y_test)
    # 0.543...
