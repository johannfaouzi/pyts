from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
standard_library.install_aliases()
from pyts.utils import idf_func, idf_smooth_func, dtw, fast_dtw
from sklearn.base import BaseEstimator
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import NotFittedError
import numpy as np


class SAXVSMClassifier(BaseEstimator):
    """
    Classifier based on SAX-VSM representation and tf-idf statistics.

    Parameters
    ----------
    tf : str
        the way to compute idf. Possibles values:

        - 'raw' : raw count
        - 'freq' : term frequency
        - 'log' : log normalization
        - 'binary' : binary

    idf : str.
        the way to compute idf. Possibles values:

        - 'idf' : inverse document frequency
        - 'idf_smooth' : inverse document frequency smooth
        - 'idf_max' : inverse document frequency max

    https://en.wikipedia.org/wiki/Tf%E2%80%93idf
    """

    def __init__(self, tf='log', idf='idf'):

        self.tf = tf
        self.idf = idf
        self.fitted = False

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples,]
            Training vector, where n_samples in the number of samples.

        y : np.array, shape = [n_samples]
            Target vector relative to X

        Returns
        -------
        self : object
            Returns self.
        """

        # Check Parameters
        if self.tf not in ['raw', 'log', 'binary', 'freq']:
            raise ValueError("'tf' must be 'raw', 'log', 'binary' or 'freq'.")
        if self.idf not in ['idf', 'idf_smooth', 'idf_max']:
            raise ValueError("'idf' must be 'idf', 'idf_smooth' or 'idf_max'.")

        # Sanity check for y
        y_unique = np.unique(y)
        if not np.all(y_unique == np.arange(y_unique.size)):
            raise ValueError("Each element of 'y' must belong to {0, ..., num_class - 1}")

        # Class parameters
        num_classes = y_unique.size
        self.num_classes_ = num_classes

        # Compute tf and idf
        all_words = np.unique(np.array([word for sublist in X for word in sublist]))
        n_all_words = all_words.size

        self.all_words_ = all_words
        self.n_all_words_ = n_all_words

        tf = np.zeros((num_classes, n_all_words))
        idf = [0 for _ in range(n_all_words)]

        for classe in range(num_classes):

            class_list = list([X[i] for i in np.where(y == classe)[0]])
            flat_class_list = np.array([word for sublist in class_list for word in sublist])

            class_unique, class_counts = np.unique(flat_class_list, return_counts=True)

            if self.tf == 'raw':
                tf_class = class_counts.copy()
            elif self.tf == 'log':
                tf_class = np.log(1 + class_counts)
            elif self.tf == 'binary':
                tf_class = np.ceil(np.clip(class_counts / class_counts.sum(), 0, 1))
            elif self.tf == 'freq':
                tf_class = np.clip(class_counts / class_counts.sum(), 0, 1)

            for i, word in enumerate(all_words):
                if word in class_unique:
                    tf[classe, i] = tf_class[np.where(class_unique == word)[0]]
                    idf[i] += 1

        if self.idf == 'idf':
            idf = np.array([idf_func(x, num_classes) for x in idf])
        elif self.idf == 'idf_smooth':
            idf = np.array([idf_smooth_func(x, num_classes) for x in idf])
        elif self.idf == 'idf_max':
            idf = np.asarray(idf)
            idf = np.log(idf.max() / (1 + idf))

        self.tf_array_ = tf.copy()
        self.idf_array_ = idf.copy()
        self.tf_idf_array_ = tf * idf

        self.fitted = True

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples]

        Returns
        -------
        y : np.array of shape [n_samples]
            Class labels for each data sample.
        """

        if not self.fitted:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")

        n_samples = len(X)
        frequencies = np.zeros((n_samples, self.n_all_words_))
        for i in range(n_samples):
            words_unique, words_counts = np.unique(X[i], return_counts=True)
            for j, word in enumerate(self.all_words_):
                if word in words_unique:
                    frequencies[i, j] = words_counts[np.where(words_unique == word)[0]]

        self.frequencies_ = frequencies

        y_pred = cosine_similarity(frequencies, self.tf_idf_array_).argmax(axis=1)

        return y_pred

    def score(self, X, y):
        """Predict the class labels for the provided data X and compute
        the accuracy with y.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples]

        y : np.array, shape = [n_samples]

        Returns
        -------
        score : float
            Score between y and the predicted classes.
        """

        return np.mean(self.predict(X) == y)


class KNNClassifier(BaseEstimator):
    """k nearest neighbors classifier

    Parameters
    ----------
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for :meth:`k_neighbors` queries.

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
        Algorithm used to compute the nearest neighbors:

        - 'ball_tree' will use :class:`BallTree`
        - 'kd_tree' will use :class:`KDTree`
        - 'brute' will use a brute-force search.
        - 'auto' will attempt to decide the most appropriate algorithm
          based on the values passed to :meth:`fit` method.

        Note: fitting on sparse input will override the setting of
        this parameter, using brute force.

    leaf_size : int, optional (default = 30)
        Leaf size passed to BallTree or KDTree.  This can affect the
        speed of the construction and query, as well as the memory
        required to store the tree.  The optimal value depends on the
        nature of the problem.

    metric : string or DistanceMetric object (default = 'minkowski')
        the distance metric to use for the tree.  The default metric is
        minkowski, and with p=2 is equivalent to the standard Euclidean
        metric. See the documentation of the DistanceMetric class for a
        list of available metrics. 'dtw' and 'fast_dtw' are also
        available.

    p : integer, optional (default = 2)
        Power parameter for the Minkowski metric. When p = 1, this is
        equivalent to using manhattan_distance (l1), and euclidean_distance
        (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.

    metric_params : dict, optional (default = None)
        Additional keyword arguments for the metric function.

    n_jobs : int, optional (default = 1)
        The number of parallel jobs to run for neighbors search.
        If ``-1``, then the number of jobs is set to the number of CPU cores.
        Doesn't affect :meth:`fit` method.

    """

    def __init__(self, n_neighbors=1, weights='uniform', algorithm='auto', leaf_size=30,
                 p=2, metric='minkowski', metric_params=None, n_jobs=1, **kwargs):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        self.kwargs = kwargs

        self.fitted = False

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : np.array, shape = [n_samples]
            Target vector relative to X

        Returns
        -------
        self : object
            Returns self.
        """

        if self.metric == 'dtw':
            self.clf = KNeighborsClassifier(self.n_neighbors, self.weights, self.algorithm,
                                            self.leaf_size, self.p, dtw, self.metric_params,
                                            self.n_jobs, **self.kwargs)

        elif self.metric == 'fast_dtw':
            self.clf = KNeighborsClassifier(self.n_neighbors, self.weights, self.algorithm,
                                            self.leaf_size, self.p, fast_dtw, self.metric_params,
                                            self.n_jobs, **self.kwargs)

        else:
            self.clf = KNeighborsClassifier(self.n_neighbors, self.weights, self.algorithm,
                                            self.leaf_size, self.p, self.metric, self.metric_params,
                                            self.n_jobs, **self.kwargs)

        self.clf.fit(X, y)
        self.fitted = True

    def predict(self, X):
        """Predict the class labels for the provided data

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        Returns
        -------
        y : np.array of shape [n_samples]
            Class labels for each data sample.
        """

        if not self.fitted:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")

        return self.clf.predict(X)

    def score(self, X, y):
        """Predict the class labels for the provided data X and compute
        the accuracy with y.

        Parameters
        ----------
        X : np.ndarray, shape = [n_samples, n_features]

        y : np.array, shape = [n_samples]

        Returns
        -------
        score : float
            Score between y and the predicted classes.
        """

        return np.mean(self.clf.predict(X) == y)
