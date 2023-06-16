"""Code for Time Series Forest."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import ceil
from numba import njit
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state)
from ..base import UnivariateClassifierMixin, UnivariateTransformerMixin


@njit()
def extract_features(X, n_samples, n_windows, indices):
    X_new = np.empty((n_samples, 3 * n_windows))
    for j in range(n_windows):
        start, end = indices[j]
        arange = np.arange((start - end + 1) / 2, (end + 1 - start) / 2)
        if end - start == 1:
            var_arange = 1.
        else:
            var_arange = np.sum(arange ** 2)

        for i in range(n_samples):
            mean = np.mean(X[i, start:end])
            X_new[i, 3 * j] = mean
            X_new[i, 3 * j + 1] = np.std(X[i, start:end])
            X_new[i, 3 * j + 2] = (
                np.sum((X[i, start:end] - mean) * arange) / var_arange
            )

    return X_new


class WindowFeatureExtractor(BaseEstimator, UnivariateTransformerMixin):
    """Feature extractor over a window.

    This transformer extracts 3 features from each window: the mean, the
    standard deviation and the slope.

    Parameters
    ----------
    n_windows : int or float (default = 1.)
        The number of windows from which features are extracted.

    min_window_size : int or float (default = 1)
        The minimum length of the windows. If float, it represents a percentage
        of the size of each time series.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    indices_ : array, shape = (n_windows, 2)
        The indices for the windows.
        The first column consists of the starting indices (included)
        of the windows. The second column consists of the ending indices
        (excluded) of the windows.

    """

    def __init__(self, n_windows=1., min_window_size=1, random_state=None):
        self.n_windows = n_windows
        self.min_window_size = min_window_size
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model according to the given training data.

        It generates the indices of the windows from which the features will be
        extracted.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        y
            Ignored

        Returns
        -------
        self : object

        """
        # Check
        X = check_array(X, dtype='float64')
        n_timestamps = X.shape[1]
        n_windows, min_window_size, rng = self._check_params(X)

        # Generate the start and end indices
        start = rng.randint(0, n_timestamps - min_window_size, size=n_windows)
        end = rng.randint(start + min_window_size, n_timestamps + 1,
                          size=n_windows)
        self.indices_ = np.c_[start, end]
        return self

    def transform(self, X):
        """Transform the provided data.

        It extracts the three features from all the selected windows
        for all the samples.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        Returns
        -------
        X_new : array, shape = (n_samples, 3 * n_windows)
            Extracted features.

        """
        X = check_array(X, dtype='float64')
        check_is_fitted(self)

        # Extract the features from each window
        n_samples = X.shape[0]
        n_windows = self.indices_.shape[0]
        return extract_features(X, n_samples, n_windows, self.indices_)

    def _check_params(self, X):
        n_samples, n_timestamps = X.shape

        if not isinstance(self.n_windows,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'n_windows' must be an integer or a float.")
        if isinstance(self.n_windows, (int, np.integer)):
            if self.n_windows < 1:
                raise ValueError(
                    "If 'n_windows' is an integer, it must be positive "
                    "(got {0}).".format(self.n_windows)
                )
            n_windows = self.n_windows
        else:
            if self.n_windows <= 0:
                raise ValueError(
                    "If 'n_windows' is a float, it must be greater "
                    "than 0 (got {0}).".format(self.n_windows)
                )
            n_windows = ceil(self.n_windows * n_timestamps)

        if not isinstance(self.min_window_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'min_window_size' must be an integer or a float.")
        if isinstance(self.min_window_size, (int, np.integer)):
            if not 1 <= self.min_window_size <= n_timestamps:
                raise ValueError(
                    "If 'min_window_size' is an integer, it must be greater "
                    "than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.min_window_size)
                )
            min_window_size = self.min_window_size
        else:
            if not 0 < self.min_window_size <= 1:
                raise ValueError(
                    "If 'min_window_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {}).".
                    format(self.min_window_size)
                )
            min_window_size = ceil(self.min_window_size * n_timestamps)

        rng = check_random_state(self.random_state)

        return n_windows, min_window_size, rng


class TimeSeriesForest(BaseEstimator, UnivariateClassifierMixin):
    """A random forest classifier for time series.

    A random forest is a meta estimator that fits a number of decision tree
    classifiers on various sub-samples of the dataset and uses averaging to
    improve the predictive accuracy and control over-fitting.

    This transformer extracts 3 features from each window: the mean, the
    standard deviation and the slope. The total number of features is thus
    equal to ``3 * n_windows``. Then a random forest is built using
    these features as input data.

    Parameters
    ----------
    n_estimators : int (default = 500)
        The number of trees in the forest.

    n_windows : int or float (default = 1.)
        The number of windows from which features are extracted.

    min_window_size : int or float (default = 1)
        The minimum length of the windows. If float, it represents a percentage
        of the size of each time series.

    criterion : str (default = "entropy")
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
        Note: this parameter is tree-specific.

    max_depth : integer or None (default = None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        ``min_samples_split`` samples.

    min_samples_split : int or float (default = 2)
        The minimum number of samples required to split an internal node:

        - If int, then consider ``min_samples_split`` as the minimum number.
        - If float, then ``min_samples_split`` is a fraction and
          ``ceil(min_samples_split * n_samples)`` are the minimum
          number of samples for each split.

    min_samples_leaf : int or float (default = 1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model.

        - If int, then consider ``min_samples_leaf`` as the minimum number.
        - If float, then ``min_samples_leaf`` is a fraction and
          ``ceil(min_samples_leaf * n_samples)`` are the minimum
          number of samples for each node.

    min_weight_fraction_leaf : float (default = 0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node.

    max_features : int, float, str or None (default = "sqrt")
        The number of features to consider when looking for the best split:

        - If int, then consider ``max_features`` features at each split.
        - If float, then ``max_features`` is a fraction and
          ``int(max_features * n_features)`` features are considered at each
          split.
        - If "sqrt", then ``max_features=sqrt(n_features)``.
        - If "log2", then ``max_features=log2(n_features)``.
        - If None, then ``max_features=n_features``.

    max_leaf_nodes : int or None (default = None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.

    min_impurity_decrease : float (default = 0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::

            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)

        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.

    bootstrap : bool (default = True)
        Whether bootstrap samples are used when building trees. If False, the
        whole datset is used to build each tree.

    oob_score : bool (default = False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.

    n_jobs : int or None, optional (default = None)
        The number of jobs to run in parallel. :meth:`fit`, :meth:`predict`,
        :meth:`decision_path` and :meth:`apply` are all parallelized over the
        trees. ``None`` means 1 unless in a ``joblib.parallel_backend``
        context. ``-1`` means using all processors.

    random_state : int, RandomState instance or None (default = None)
        Controls both the randomness of the bootstrapping of the samples used
        when building trees (if ``bootstrap=True``) and the sampling of the
        features to consider when looking for the best split at each node
        (if ``max_features < n_features``).

    verbose : int (default = 0)
        Controls the verbosity when fitting and predicting.

    class_weight : dict, "balanced", "balanced_subsample" or None (default = None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one.

        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.

    ccp_alpha : float (default = 0.)
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed.
        It must be non-negative.

    max_samples : int, float or None (default = None)
        If bootstrap is True, the number of samples to draw from X
        to train each base estimator:

        - If None (default), then draw ``X.shape[0]`` samples.
        - If int, then draw ``max_samples`` samples.
        - If float, then draw ``max_samples * X.shape[0]`` samples. Thus,
          ``max_samples`` should be in the interval `(0, 1)`.

    Attributes
    ----------
    estimator_ : DecisionTreeClassifier
        The child estimator template used to create the collection of fitted
        sub-estimators.

    classes_ : array, shape = (n_classes,)
        The classes labels.

    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    feature_importances_ : array, shape = (n_features,)
        The feature importances (the higher, the more important the feature).

    indices_ : array, shape = (n_windows, 2)
        The indices for the windows.
        The first column consists of the starting indices (included)
        of the windows. The second column consists of the ending indices
        (excluded) of the windows.

    n_features_in_ : int
        The number of features when ``fit`` is performed.

    oob_decision_function_ : None or array, shape = (n_samples, n_classes)
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        `oob_decision_function_` might contain NaN. This attribute is not None
        only when ``oob_score`` is True.

    oob_score_ : None or float
        Score of the training dataset obtained using an out-of-bag estimate.
        This attribute is not None only when ``oob_score`` is True.

    Examples
    --------
    >>> from pyts.datasets import load_gunpoint
    >>> from pyts.classification import TimeSeriesForest
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = TimeSeriesForest(random_state=43)
    >>> clf.fit(X_train, y_train)
    TimeSeriesForest(...)
    >>> clf.score(X_test, y_test)
    0.97333...

    Notes
    -----
    The default values for the parameters controlling the size of the trees
    (e.g. ``max_depth``, ``min_samples_leaf``, etc.) lead to fully grown and
    unpruned trees which can potentially be very large on some data sets. To
    reduce memory consumption, the complexity and size of the trees should be
    controlled by setting those parameter values.

    The features are always randomly permuted at each split. Therefore,
    the best found split may vary, even with the same training data,
    ``max_features=n_features`` and ``bootstrap=False``, if the improvement
    of the criterion is identical for several splits enumerated during the
    search of the best split. To obtain a deterministic behaviour during
    fitting, ``random_state`` has to be fixed.

    References
    ----------
    .. [1] H. Deng, G. Runger, E. Tuv and M. Vladimir, "A Time Series
           Forest for Classification and Feature Extraction".
           Information Sciences, 239, 142-153 (2013).

    .. [2] Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    """  # noqa: E501
    def __init__(self,
                 n_estimators=500,
                 n_windows=1.,
                 min_window_size=1,
                 criterion="entropy",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="sqrt",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 class_weight=None,
                 ccp_alpha=0.0,
                 max_samples=None):
        self.n_estimators = n_estimators
        self.n_windows = n_windows
        self.min_window_size = min_window_size
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha
        self.max_samples = max_samples

    def apply(self, X):
        """Apply trees in the forest to X, return leaf indices.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        Returns
        -------
        X_leaves : array_like, shape = (n_samples, n_estimators)
            For each datapoint x in X and for each tree in the forest,
            return the index of the leaf x ends up in.

        """
        check_is_fitted(self)
        X = check_array(X, dtype='float64')
        X_new = self._pipeline['fe'].transform(X)
        return self._pipeline['rfc'].apply(X_new)

    def decision_path(self, X):
        """Return the decision path in the forest.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        Returns
        -------
        indicator : sparse csr array, shape = (n_samples, n_nodes)
            Return a node indicator matrix where non zero elements
            indicates that the samples goes through the nodes.

        n_nodes_ptr : array, shape = (n_estimators + 1,)
            The columns from indicator[n_nodes_ptr[i]:n_nodes_ptr[i+1]]
            gives the indicator value for the i-th estimator.

        """
        check_is_fitted(self)
        X = check_array(X, dtype='float64')
        X_new = self._pipeline['fe'].transform(X)
        return self._pipeline['rfc'].decision_path(X_new)

    def fit(self, X, y):
        """Fit the model according to the given training data.

        It build a forest of trees from the training set.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        y : array-like, shape = (n_samples,)
            Class labels for each sample.

        Returns
        -------
        self : object

        """
        # Create and fit the pipeline
        feature_extractor = WindowFeatureExtractor(
            n_windows=self.n_windows, min_window_size=self.min_window_size,
            random_state=self.random_state
        )
        rfc = RandomForestClassifier(
            n_estimators=self.n_estimators,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            min_weight_fraction_leaf=self.min_weight_fraction_leaf,
            max_features=self.max_features,
            max_leaf_nodes=self.max_leaf_nodes,
            min_impurity_decrease=self.min_impurity_decrease,
            bootstrap=self.bootstrap,
            oob_score=self.oob_score,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            warm_start=False
        )
        self._pipeline = Pipeline([('fe', feature_extractor), ('rfc', rfc)])
        self._pipeline.fit(X, y)

        # Get attributes
        self.estimator_ = self._pipeline['rfc'].estimator_
        self.classes_ = self._pipeline['rfc'].classes_
        self.estimators_ = self._pipeline['rfc'].estimators_
        self.feature_importances_ = self._pipeline['rfc'].feature_importances_
        self.indices_ = self._pipeline['fe'].indices_
        self.n_features_in_ = (
            self._pipeline['rfc'].n_features_in_
            if hasattr(self._pipeline['rfc'], 'n_features_in_')
            else self._pipeline['rfc'].n_features_
        )
        self.oob_decision_function_ = getattr(
            self._pipeline['rfc'], 'oob_decision_function_', None)
        self.oob_score_ = getattr(self._pipeline['rfc'], 'oob_score_', None)

        return self

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input time series is a vote by the trees
        in the forest, weighted by their probability estimates.
        That is, the predicted class is the one with highest mean
        probability estimate across the trees.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        Returns
        -------
        y : array, shape = (n_samples,)
            The predicted classes.

        """
        check_is_fitted(self)
        return self._pipeline.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        The predicted class probabilities of an input time series are computed
        as the mean predicted class probabilities of the trees in the forest.
        The class probability of a single tree is the fraction of samples
        of the same class in a leaf.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Univariate time series.

        Returns
        -------
        p : array, shape = (n_samples, n_classes)
            The class probabilities of the input time series.
            The order of the classes corresponds to that in the
            attribute `classes_`.

        """
        check_is_fitted(self)
        return self._pipeline.predict_proba(X)

    def score(self, X, y):
        """Return the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        y : array-like, shape = (n_samples,)
            True labels for X.

        Returns
        -------
        score : float
            Mean accuracy of self.predict(X) wrt. y.

        """
        check_is_fitted(self)
        return self._pipeline.score(X, y)
