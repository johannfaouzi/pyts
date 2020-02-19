"""Code for k-nearest-neighbors."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_is_fitted
from ..metrics import boss, dtw, sakoe_chiba_band, itakura_parallelogram
from ..metrics.dtw import (_dtw_classic, _dtw_region, _dtw_fast,
                           _dtw_multiscale)


class KNeighborsClassifier(BaseEstimator, ClassifierMixin):
    """k-nearest neighbors classifier.

    Parameters
    ----------
    n_neighbors : int, optional (default = 1)
        Number of neighbors to use.

    weights : str or callable, optional (default = 'uniform')
        weight function used in prediction.  Possible values:

        - 'uniform' : uniform weights.  All points in each neighborhood
          are weighted equally.
        - 'distance' : weight points by the inverse of their distance.
          in this case, closer neighbors of a query point will have a
          greater influence than neighbors which are further away.
        - [callable] : a user-defined function which accepts an
          array of distances, and returns an array of the same shape
          containing the weights.

    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
        Algorithm used to compute the nearest neighbors. Ignored ff ``metric``
        is either 'dtw', 'dtw_sakoechiba', 'dtw_itakura', 'dtw_multiscale',
        'dtw_fast' or 'boss' ('brute' will be used).

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or DistanceMetric object (default = 'minkowski')
        The distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class from
        scikit-learn for a list of available metrics.
        For Dynamic Time Warping, the available metrics are 'dtw',
        'dtw_sakoechiba', 'dtw_itakura', 'dtw_multiscale' and 'dtw_fast'.
        For BOSS metric, one can use 'boss'.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``n_jobs=-1``, then the number of jobs is set to the number of CPU
        cores. Doesn't affect :meth:`fit` method.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    Examples
    --------
    >>> from pyts.classification import KNeighborsClassifier
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = KNeighborsClassifier()
    >>> clf.fit(X_train, y_train)
    KNeighborsClassifier(...)
    >>> clf.score(X_test, y_test)
    0.91...

    """

    def __init__(self, n_neighbors=1, weights='uniform', algorithm='auto',
                 leaf_size=30, p=2, metric='minkowski', metric_params=None,
                 n_jobs=1, **kwargs):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.kwargs = kwargs

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Training vector.

        y : array-like, shape = (n_samples,)
            Class labels for each data sample.

        Returns
        -------
        self : object

        """
        X, y = check_X_y(X, y)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_

        if self.metric == 'dtw':
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=dtw,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'dtw_classic':
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=_dtw_classic,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'dtw_sakoechiba':
            n_timestamps = X.shape[1]
            if self.metric_params is None:
                region = sakoe_chiba_band(n_timestamps)
            else:
                if 'window_size' not in self.metric_params.keys():
                    window_size = 0.1
                else:
                    window_size = self.metric_params['window_size']
                region = sakoe_chiba_band(n_timestamps,
                                          window_size=window_size)
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=_dtw_region,
                metric_params={'region': region},
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'dtw_itakura':
            n_timestamps = X.shape[1]
            if self.metric_params is None:
                region = itakura_parallelogram(n_timestamps)
            else:
                if 'max_slope' not in self.metric_params.keys():
                    max_slope = 2.
                else:
                    max_slope = self.metric_params['max_slope']
                region = itakura_parallelogram(n_timestamps,
                                               max_slope=max_slope)
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=_dtw_region,
                metric_params={'region': region},
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'dtw_multiscale':
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=_dtw_multiscale,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'dtw_fast':
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=_dtw_fast,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        elif self.metric == 'boss':
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm='brute', metric=boss,
                metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        else:
            self._clf = SklearnKNN(
                n_neighbors=self.n_neighbors, weights=self.weights,
                algorithm=self.algorithm, leaf_size=self.leaf_size,
                p=self.p, metric=self.metric, metric_params=self.metric_params,
                n_jobs=self.n_jobs, **self.kwargs
            )

        self._clf.fit(X, y)
        return self

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Test samples.

        Returns
        -------
        p : array, shape = (n_samples, n_classes)
            Probability estimates.

        """
        check_is_fitted(self, '_clf')
        return self._clf.predict_proba(X)

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        y_pred : array-like, shape = (n_samples,)
            Class labels for each data sample.

        """
        check_is_fitted(self, '_clf')
        return self._clf.predict(X)
