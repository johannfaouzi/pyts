from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from itertools import chain
from future import standard_library
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from pyts.utils import dtw, fast_dtw
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.exceptions import NotFittedError
import numpy as np


standard_library.install_aliases()


class SAXVSMClassifier(BaseEstimator, ClassifierMixin):
    """Classifier based on SAX-VSM representation and tf-idf statistics.
    It uses the implementation from scikit-learn: TfidfVectorizer.

    Parameters
    ----------
    norm : 'l1', 'l2' or None, optional
        Norm used to normalize term vectors. None for no normalization.

    use_idf : boolean, default=True
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, default=False
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of terms to feature indices.

    idf_ : array, shape = [n_features], or None
        The learned idf vector (global term weights)
        when ``use_idf`` is set to True, None otherwise.

    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.
    """

    def __init__(self, norm='l2', use_idf=True, smooth_idf=True, sublinear_tf=False):
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

        self.fitted = False

    def fit(self, X, y):

        check_classification_targets(y)

        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = classes = le.classes_
        n_classes = classes.size

        X_clas = []

        for cur_class in range(n_classes):
            center_mask = y_ind == cur_class
            sentence = ' '.join(list(chain(*X[center_mask])))
            X_clas.append(sentence)

        tfidf = TfidfVectorizer(norm=self.norm,
                                use_idf=self.use_idf,
                                smooth_idf=self.smooth_idf,
                                sublinear_tf=self.sublinear_tf)
        self.tfidf_array_ = tfidf.fit_transform(X_clas)
        self.tfidf_ = tfidf

        self.fitted = True

        return self

    def predict(self, X):

        if not self.fitted:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")
        check_is_fitted(self.tfidf_, ['vocabulary_', 'idf_', 'stop_words_'])
        check_is_fitted(self, ['classes_', 'tfidf_array_', 'tfidf_'])

        X_test = [' '.join(x) for x in X]
        X_transformed = self.tfidf_.transform(X_test)
        y_pred = cosine_similarity(X_transformed, self.tfidf_array_).argmax(axis=1)

        return self.classes_[y_pred]


class KNNClassifier(BaseEstimator, ClassifierMixin):
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

        X, y = check_X_y(X, y)

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

        return self

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

        X = check_array(X)
        if X.ndim == 1:
            X_ = X.reshape((1, -1))
        else:
            X_ = X.copy()

        if not self.fitted:
            raise NotFittedError("Estimator not fitted, call `fit` before exploiting the model.")

        return self.clf.predict(X_)
