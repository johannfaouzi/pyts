"""Code for Time Series Bag-of-Features."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from math import ceil
from numba import njit
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import (
    check_array, check_is_fitted, check_random_state, check_X_y
)
from ..base import UnivariateClassifierMixin, UnivariateTransformerMixin


@njit
def extract_features(X, interval_indices, n_samples, n_subseries, n_intervals):
    # There are 3 features for each interval (mean, std, slope) plus
    # the mean, std over the whole subsequence and indices of the subsequence
    n_features = n_intervals * 3 + 4
    X_features = np.zeros((n_samples * n_subseries, n_features))

    for j in range(n_subseries):
        for k in range(n_intervals):

            # Get the start and end indices of the interval
            start, end = interval_indices[:, j, k]
            arange = np.arange((start - end + 1) / 2, (end + 1 - start) / 2)
            if end - start == 1:
                var_arange = 1.
            else:
                var_arange = np.sum(arange ** 2)

            for i in range(n_samples):
                i_j = i * n_subseries + j
                mean = np.mean(X[i, start:end])

                # Compute the three statistics of the interval
                X_features[i_j, 3 * k] = mean
                X_features[i_j, 3 * k + 1] = np.std(X[i, start:end])
                X_features[i_j, 3 * k + 2] = (
                    np.sum((X[i, start:end] - mean) * arange) / var_arange
                )

        start, end = interval_indices[0, j, 0], interval_indices[1, j, -1]
        for i in range(n_samples):
            i_j = i * n_subseries + j

            # Compute the four statistics of the subseries
            X_features[i_j, -4] = np.mean(X[i, start:end])
            X_features[i_j, -3] = np.std(X[i, start:end])
            X_features[i_j, -2] = start
            X_features[i_j, -1] = end

    return X_features


@njit
def histogram(X, bins, n_bins, n_samples, n_classes):
    X_new = np.empty((n_samples, (n_bins + 1) * n_classes))
    for i in range(n_samples):
        for j in range(n_classes):
            X_new[i, j * n_bins: (j + 1) * n_bins] = np.histogram(
                X[i, :, j], bins, (0., 1.))[0]
            X_new[i, - n_classes + j] = np.mean(X[i, :, j])
    return X_new


class IntervalFeatureExtractor(BaseEstimator, UnivariateTransformerMixin):
    """Feature extractor over the intervals of a subsequence.

    This transformer extracts 3 features from each interval of each
    subsequence: the mean, the standard deviation and the slope.

    Parameters
    ----------
    min_subsequence_size : int or float (default = 0.5)
        The minimum length of the subsequences. If float, it represents a
        percentage of the size of each time series. Note that the actual
        minimum length of the subsequences may be slighty lower due to the use
        of the floor function.

    min_interval_size : int or float (default = 0.1)
        The minimum length of the intervals. If float, it represents a
        percentage of the size of each time series. Must be lower than
        `min_subsequence_size`.

    n_subsequences : 'auto', int or float (default = 'auto')
        The number of considered subsequences. If 'auto', it is automatically
        calculated given the size of the time series and the values of the
        parameters. If float, it represents a percentage of the size of each
        time series.

    bins : int or array-like (default = 10)
        If bins is an int, it defines the number of equal-width bins in range
        [0, 1]. If bins is array-like, it defines a monotonically increasing
        array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.

    random_state : None, int or RandomState instance (default = None)
        The seed of the pseudo random number generator to use when shuffling
        the data. If int, random_state is the seed used by the random number
        generator. If RandomState instance, random_state is the random number
        generator. If None, the random number generator is the RandomState
        instance used by `np.random`.

    Attributes
    ----------
    interval_indices_ : array, shape = (n_subsequences, n_intervals + 1)
        The indices for the intervals of each subsequence.

    min_subsequence_size_ : int
        The actual minimum length of the subsequences.

    """
    def __init__(self, min_subsequence_size=0.5, min_interval_size=0.1,
                 n_subsequences='auto', random_state=None):
        self.min_subsequence_size = min_subsequence_size
        self.min_interval_size = min_interval_size
        self.n_subsequences = n_subsequences
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
        X = check_array(X, dtype='float64')
        n_timestamps = X.shape[1]

        (min_subsequence_size, min_interval_size, n_subsequences,
         rng) = self._check_params(X, n_timestamps)

        # Define the parameters of the subseries and intervals
        n_intervals = int(min_subsequence_size / min_interval_size)
        min_subseries_size = min_interval_size * n_intervals
        if n_subsequences == 'auto':
            n_subseries = max(
                1, n_timestamps // min_interval_size - n_intervals)
        else:
            n_subseries = n_subsequences

        # Generate the indices of the subseries
        subseries_start = rng.randint(
            0, n_timestamps - min_subseries_size + 1, size=n_subseries
        )
        subseries_end = rng.randint(
            subseries_start + min_subseries_size, n_timestamps + 1,
            size=n_subseries
        )

        # Generate the indices of the intervals
        interval_indices = np.linspace(
            subseries_start, subseries_end, n_intervals + 1, axis=1
        ).astype('int64')

        # Save the relevant parameters as attributes
        self.interval_indices_ = interval_indices
        self.min_subsequence_size_ = min_subsequence_size

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
        X_new : array, shape = (n_samples * n_subseries, 3 * n_intervals + 4)
            Extracted features.

        """
        X = check_array(X, dtype='float64')
        n_samples = X.shape[0]

        # Extract the features from each interval
        interval_indices = np.array(
            [self.interval_indices_[:, :-1], self.interval_indices_[:, 1:]]
        )

        n_subseries, n_intervals = interval_indices.shape[1:]
        X_features = extract_features(X, interval_indices, n_samples,
                                      n_subseries, n_intervals)
        return X_features

    def _check_params(self, X, n_timestamps):

        if not isinstance(self.min_subsequence_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'min_subsequence_size' must be an integer or a "
                            "float.")
        if isinstance(self.min_subsequence_size, (int, np.integer)):
            if not 1 <= self.min_subsequence_size <= n_timestamps:
                raise ValueError(
                    "If 'min_subsequence_size' is an integer, it must be "
                    "greater than or equal to 1 and lower than or equal to "
                    "n_timestamps (got {0}).".format(self.min_subsequence_size)
                )
            min_subsequence_size = self.min_subsequence_size
        else:
            if not (0 < self.min_subsequence_size <= 1.):
                raise ValueError(
                    "If 'min_subsequence_size' is a float, it must be greater "
                    "than 0 and lower than or equal to 1 (got {0})."
                    .format(self.min_subsequence_size)
                )
            min_subsequence_size = ceil(
                self.min_subsequence_size * n_timestamps)

        if not isinstance(self.min_interval_size,
                          (int, np.integer, float, np.floating)):
            raise TypeError("'min_interval_size' must be an integer or a "
                            "float.")
        if isinstance(self.min_interval_size, (int, np.integer)):
            if self.min_interval_size < 1:
                raise ValueError(
                    "If 'min_interval_size' is an integer, it must be "
                    "positive (got {0}).".format(self.min_interval_size)
                )
            min_interval_size = self.min_interval_size
        else:
            if self.min_interval_size <= 0:
                raise ValueError(
                    "If 'min_interval_size' is a float, it must be greater "
                    "than 0 (got {0}).".format(self.min_interval_size)
                )
            min_interval_size = ceil(self.min_interval_size * n_timestamps)

        if min_interval_size > min_subsequence_size:
            raise ValueError("'min_interval_size' must be lower than or equal "
                             "to 'min_subsequence_size' ({} > {})."
                             .format(min_interval_size, min_subsequence_size))

        n_subsequences_auto = (isinstance(self.n_subsequences, str) and
                               self.n_subsequences == 'auto')
        if not (n_subsequences_auto or isinstance(
            self.n_subsequences, (int, np.integer, float, np.floating))
        ):
            raise TypeError("'n_subsequences' must be 'auto', an integer or a "
                            "float.")
        if n_subsequences_auto:
            n_subsequences = 'auto'
        elif isinstance(self.n_subsequences, (int, np.integer)):
            if self.n_subsequences < 1:
                raise ValueError(
                    "If 'n_subsequences' is an integer, it must be positive "
                    "(got {0}).".format(self.n_subsequences)
                )
            n_subsequences = self.n_subsequences
        else:
            if self.n_subsequences <= 0:
                raise ValueError(
                    "If 'n_subsequences' is a float, it must be greater "
                    "than 0 (got {0}).".format(self.n_subsequences)
                )
            n_subsequences = ceil(self.n_subsequences * n_timestamps)

        rng = check_random_state(self.random_state)

        return (min_subsequence_size, min_interval_size,
                n_subsequences, rng)


class TSBF(BaseEstimator, UnivariateClassifierMixin):
    """Time Series Bag-of-Features algorithm.

    Parameters
    ----------
    n_estimators : int (default = 500)
        The number of trees in the forest.

    min_subsequence_size : int or float (default = 0.5)
        The minimum length of the subsequences. If float, it represents a
        percentage of the size of each time series. Note that the actual
        minimum length of the subsequences may be slighty lower due to the use
        of the floor function.

    min_interval_size : int or float (default = 0.1)
        The minimum length of the intervals. If float, it represents a
        percentage of the size of each time series. Must be lower than
        `min_subsequence_size`.

    n_subsequences : 'auto', int or float (default = 'auto')
        The number of considered subsequences. If 'auto', it is automatically
        calculated given the size of the time series and the values of the
        parameters. If float, it represents a percentage of the size of each
        time series.

    bins : int or array-like (default = 10)
        If bins is an int, it defines the number of equal-width bins in range
        [0, 1]. If bins is array-like, it defines a monotonically increasing
        array of bin edges, including the rightmost edge, allowing for
        non-uniform bin widths.

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

    interval_indices_ : array, shape = (n_subsequences, n_intervals + 1)
        The indices for the intervals of each subsequence.

    min_subsequence_size_ : int
        The actual minimum length of the subsequences.

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
    >>> from pyts.classification import TSBF
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = TSBF(random_state=43)
    >>> clf.fit(X_train, y_train)
    TSBF(...)
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
    .. [1] M.G. Baydogan, G. Runger and E. Tuv, "A Bag-of-Features Framework
       to Classify Time Series". IEEE Transactions on Pattern Analysis
       and Machine Intelligence, 35(11), 2796-2802 (2013).

    .. [2] Leo Breiman, "Random Forests", Machine Learning, 45(1), 5-32, 2001.

    """  # noqa: E501
    def __init__(self,
                 n_estimators=500,
                 min_subsequence_size=0.5,
                 min_interval_size=0.1,
                 n_subsequences='auto',
                 bins=10,
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
        self.min_subsequence_size = min_subsequence_size
        self.min_interval_size = min_interval_size
        self.n_subsequences = n_subsequences
        self.bins = bins
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
        X_binned = self._transform(X)
        return self._clf.apply(X_binned)

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
        X_binned = self._transform(X)
        return self._clf.decision_path(X_binned)

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
        # Extract some information from the dataset
        X, y = check_X_y(X, y)
        n_samples, n_timestamps = X.shape
        n_classes = np.unique(y).size

        bins = self._check_params()
        self._bins = bins

        # Create the instances of the feature extractor anf classifiers
        feature_extractor = IntervalFeatureExtractor(
            min_subsequence_size=self.min_subsequence_size,
            min_interval_size=self.min_interval_size,
            n_subsequences=self.n_subsequences,
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
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            verbose=self.verbose,
            class_weight=self.class_weight,
            ccp_alpha=self.ccp_alpha,
            max_samples=self.max_samples,
            warm_start=False,
            oob_score=True
        )
        clf = RandomForestClassifier(
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
            warm_start=False,
        )

        # Fit the random forest and get out-of-bag probabilities
        X_features = feature_extractor.fit_transform(X)
        n_subsequences = feature_extractor.interval_indices_.shape[0]
        y_features = np.repeat(y, n_subsequences)
        rfc.fit(X_features, y_features)
        X_oob_proba = rfc.oob_decision_function_.reshape(
            n_samples, n_subsequences, n_classes)

        # Check for subsequences without OOB scores
        no_oob_scores = (
            (np.isnan(X_oob_proba).any() or
             np.all(X_oob_proba == 0., axis=2).any())
        )
        if no_oob_scores:
            raise ValueError(
                "At least one sample was never left out during the bootstrap. "
                "Increase the number of trees (n_estimators)."
            )

        # Bin the probabilities
        if isinstance(bins, (int, np.integer)):
            n_bins = bins
        else:
            n_bins = len(bins) - 1
        X_binned = histogram(X_oob_proba, bins, n_bins, n_samples, n_classes)

        # Fit a random forest on the binned probabilities
        clf.fit(X_binned, y)

        # Get attributes
        self.estimator_ = clf.estimator_
        self.classes_ = clf.classes_
        self.estimators_ = clf.estimators_
        self.feature_importances_ = clf.feature_importances_
        self.interval_indices_ = feature_extractor.interval_indices_
        self.min_subsequence_size_ = feature_extractor.min_subsequence_size_
        self.n_features_in_ = (
            clf.n_features_in_
            if hasattr(clf, 'n_features_in_')
            else clf.n_features_
        )
        self.oob_decision_function_ = getattr(
            clf, 'oob_decision_function_', None)
        self.oob_score_ = getattr(clf, 'oob_score_', None)

        self._feature_extractor = feature_extractor
        self._rfc = rfc
        self._clf = clf

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
        X_binned = self._transform(X)
        return self._clf.predict(X_binned)

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
        X_binned = self._transform(X)
        return self._clf.predict_proba(X_binned)

    def _check_params(self):
        if not isinstance(self.bins,
                          (int, np.integer, list, tuple, np.ndarray)):
            raise TypeError("'bins' must be an integer or array-like.")
        if isinstance(self.bins, (int, np.integer)):
            bins = self.bins
        else:
            bins = np.asarray(self.bins)
            if (np.diff(bins) < 0).any():
                raise ValueError("If 'bins' is array-like, the bin edges "
                                 "must increase monotonically.")
        return bins

    def _transform(self, X):
        check_is_fitted(self)
        X = check_array(X, dtype='float64')
        n_samples = X.shape[0]
        n_subsequences = self.interval_indices_.shape[0]
        n_classes = len(self.classes_)

        # Predict probabilities using the inner random forest
        X_features = self._feature_extractor.transform(X)
        X_proba = self._rfc.predict_proba(X_features).reshape(
            n_samples, n_subsequences, n_classes)

        # Bin the probabilities
        if isinstance(self._bins, (int, np.integer)):
            n_bins = self._bins
        else:
            n_bins = len(self._bins) - 1
        X_binned = histogram(X_proba, self._bins, n_bins, n_samples, n_classes)

        return X_binned
