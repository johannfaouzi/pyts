"""The :mod:`pyts.classification` module includes classification algorithms.

Implemented algorithms are:
- k nearest neighbors
- SAX-VSM
- Bag-of-SFA in Vector Space
"""

from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from future import standard_library
import numpy as np
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from ..bow import BOW
from ..quantization import SAX, SFA
from ..utils import dtw, fast_dtw, numerosity_reduction


standard_library.install_aliases()


class KNNClassifier(BaseEstimator, ClassifierMixin):
    """k nearest neighbors classifier.

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
        Algorithm used to compute the nearest neighbors.

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
        If ``n_jobs=-1``, then the number of jobs is set to the number of CPU
        cores. Doesn't affect :meth:`fit` method.

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
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Class labels for each data sample.

        Returns
        -------
        self : object
            Returns self.

        """
        X, y = check_X_y(X, y)

        if self.metric == 'dtw':
            self._clf = KNeighborsClassifier(self.n_neighbors, self.weights,
                                             self.algorithm, self.leaf_size,
                                             self.p, dtw, self.metric_params,
                                             self.n_jobs, **self.kwargs)

        elif self.metric == 'fast_dtw':
            self._clf = KNeighborsClassifier(self.n_neighbors, self.weights,
                                             self.algorithm, self.leaf_size,
                                             self.p, fast_dtw,
                                             self.metric_params,
                                             self.n_jobs, **self.kwargs)

        else:
            self._clf = KNeighborsClassifier(self.n_neighbors, self.weights,
                                             self.algorithm, self.leaf_size,
                                             self.p, self.metric,
                                             self.metric_params,
                                             self.n_jobs, **self.kwargs)

        self._clf.fit(X, y)
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y : array-like, shape [n_samples]
            Class labels for each data sample.

        """
        check_is_fitted(self, '_clf')
        X = check_array(X)
        return self._clf.predict(X)


class SAXVSMClassifier(BaseEstimator, ClassifierMixin):
    """Classifier based on SAX-VSM representation and tf-idf statistics.

    Parameters
    ----------
    n_bins : int (default = 4)
        Number of bins (also known as the size of the alphabet).

    quantiles : {'gaussian', 'empirical'} (default = 'empirical')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used.

    window_size : int (default = 4)
        Size of the window (i.e. the size of each word).

    numerosity_reduction : bool (default = True)
        If True, delete all but one occurence of back to back
        identical occurences of the same words.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = True)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = False)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of feature indices to terms.

    tfidf_ : sparse matrix, shape = [n_classes, n_words]
        Term-document matrix

    idf_ : array, shape = [n_features], or None
        The learned idf vector (global term weights) when ``use_idf=True``,
        None otherwise.

    stop_words_ : set
        Terms that were ignored because they either:
          - occurred in too many documents (`max_df`)
          - occurred in too few documents (`min_df`)
          - were cut off by feature selection (`max_features`).
        This is only available if no vocabulary was given.

    """

    def __init__(self, n_bins=4, quantiles='empirical', window_size=4,
                 numerosity_reduction=True, use_idf=True, smooth_idf=True,
                 sublinear_tf=False):
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.window_size = window_size
        self.numerosity_reduction = numerosity_reduction
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples]
            Training vector, where n_samples is the number of samples.

        y : array-like, shape = [n_samples]
            Class labels for each data sample.

        Returns
        -------
        self : object
            Returns self.

        """
        # Check parameters
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.n_bins > 26:
            raise ValueError("'n_bins' must be lower than or equal to 26.")
        if self.quantiles not in ['gaussian', 'empirical']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")

        if not isinstance(self.use_idf, (int, float)):
            raise TypeError("'use_idf' must be a boolean.")
        if not isinstance(self.smooth_idf, (int, float)):
            raise TypeError("'smooth_idf' must be a boolean.")
        if not isinstance(self.sublinear_tf, (int, float)):
            raise TypeError("'sublinear_tf' must be a boolean.")

        X, y = check_X_y(X, y)
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self._classes = le.classes_
        n_classes = self._classes.size

        # SAX and BOW transformations
        sax = SAX(self.n_bins, self.quantiles)
        X_sax = sax.fit_transform(X)
        bow = BOW(self.window_size, self.numerosity_reduction)
        X_bow = bow.fit_transform(X_sax)

        X_class = [' '.join(X_bow[y_ind == classe])
                   for classe in range(n_classes)]

        tfidf = TfidfVectorizer(norm=None,
                                use_idf=self.use_idf,
                                smooth_idf=self.smooth_idf,
                                sublinear_tf=self.sublinear_tf)
        if self.window_size == 1:
            tfidf.set_params(tokenizer=self._tok)
        self.tfidf_ = tfidf.fit_transform(X_class)
        self.vocabulary_ = {value: key for key, value in
                            tfidf.vocabulary_.items()}
        self.stop_words_ = tfidf.stop_words
        if self.use_idf:
            self.idf_ = tfidf.idf_
        else:
            self.idf_ = None
        self._tfidf = tfidf
        self._sax = sax
        self._bow = bow
        return self

    def predict(self, X):
        """Predict the class labels for the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        y : array-like, shape [n_samples]
            Class labels for each data sample.

        """
        check_is_fitted(self, ['vocabulary_', 'tfidf_', 'idf_',
                               'stop_words_', '_tfidf'])

        # SAX and BOW transformations
        X_sax = self._sax.transform(X)
        X_bow = self._bow.transform(X_sax)
        X_transformed = self._tfidf.transform(X_bow)
        if self.use_idf:
            X_transformed /= self._tfidf.idf_
        y_pred = cosine_similarity(X_transformed,
                                   self.tfidf_).argmax(axis=1)
        return self._classes[y_pred]

    def _tok(self, x):
        return x.split(' ')


class BOSSVSClassifier(BaseEstimator, ClassifierMixin):
    """Bag-of-SFA Symbols in Vector Space.

    Parameters
    ----------
    n_coefs : None or int (default = None)
        The number of Fourier coefficients to keep. If ``n_coefs=None``,
        all Fourier coefficients are returned. If ``n_coefs`` is an integer,
        the ``n_coefs`` most significant Fourier coefficients are returned if
        ``anova=True``, otherwise the first ``n_coefs`` Fourier coefficients
        are returned. A even number is required (for real and imaginary values)
        if ``anova=False``.

    window_size : int
        Window length to use to extracte sub time series.

    norm_mean : bool (default = True)
        If True, center the data before scaling. If ``norm_mean=True`` and
        ``anova=False``, the first Fourier coefficient will be dropped.

    norm_std : bool (default = True)
        If True, scale the data to unit variance.

    n_bins : int (default = 4)
        The number of bins. Ignored if ``quantiles='entropy'``.

    quantiles : {'gaussian', 'empirical'} (default = 'gaussian')
        The way to compute quantiles. If 'gaussian', quantiles from a
        gaussian distribution N(0,1) are used. If 'empirical', empirical
        quantiles are used.

    variance_selection : bool (default = False)
        If True, the Fourier coefficients with low variance are removed.

    variance_threshold : float (default = 0.)
        Fourier coefficients with a training-set variance lower than this
        threshold will be removed. Ignored if ``variance_selection=False``.

    numerosity_reduction : boolean (default = True)
        whether or not numerosity reduction is applied. When the same word
        occurs several times in a row, only one instance of this word is kept
        if ``numerosity_reduction=True``, otherwise all instances are kept.

    smooth_idf : boolean, default=True
        smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : boolean, default=False
        apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    Attributes
    ----------
    vocabulary_ : dict
        A mapping of features indices to terms.

    """

    def __init__(self, n_coefs, window_size, norm_mean=True, norm_std=True,
                 n_bins=4, quantiles='empirical', variance_selection=False,
                 variance_threshold=0., numerosity_reduction=True,
                 smooth_idf=True, sublinear_tf=True):
        self.n_coefs = n_coefs
        self.window_size = window_size
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.n_bins = n_bins
        self.quantiles = quantiles
        self.variance_selection = variance_selection
        self.variance_threshold = variance_threshold
        self.numerosity_reduction = numerosity_reduction
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y, overlapping=True):
        """Fit the model according to the given training data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training vector, where n_samples in the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Class labels for each data sample.

        overlapping : bool (default = False)
            If True, overlapping windows are used for the training phase.

        Returns
        -------
        self : object

        """
        # Check input data
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self._classes = le.classes_
        n_classes = self._classes.size
        n_samples, n_features = X.shape

        # Check parameters
        if (not isinstance(self.n_coefs, int)) and (self.n_coefs is not None):
            raise TypeError("'n_coefs' must be None or an integer.")
        if isinstance(self.n_coefs, int) and self.n_coefs < 2:
            raise ValueError("'n_coefs' must be greater than or equal to 2.")
        if isinstance(self.n_coefs, int) and self.n_coefs % 2 != 0:
            raise ValueError("'n_coefs' must be an even integer.")
        if not isinstance(self.window_size, int):
            raise TypeError("'window_size' must be an integer.")
        if self.window_size > n_features:
            raise ValueError("'window_size' must be lower than or equal to "
                             "the size of each time series.")
        if isinstance(self.n_coefs, int) and self.n_coefs > self.window_size:
            raise TypeError("'n_coefs' must be lower than or equal to "
                            "'window_size'.")
        if not isinstance(self.norm_mean, (int, float)):
            raise TypeError("'norm_mean' must be a boolean.")
        if not isinstance(self.norm_std, (int, float)):
            raise TypeError("'norm_std' must be a boolean.")
        if not isinstance(self.n_bins, int):
            raise TypeError("'n_bins' must be an integer.")
        if self.n_bins < 2:
            raise ValueError("'n_bins' must be greater than or equal to 2.")
        if self.quantiles not in ['empirical', 'gaussian']:
            raise ValueError("'quantiles' must be either 'gaussian' or "
                             "'empirical'.")
        if not isinstance(self.variance_selection, (int, float)):
            raise TypeError("'variance_selection' must be a boolean.")
        if not isinstance(self.variance_threshold, (int, float)):
            raise TypeError("'variance_threshold' must be a float.")
        if not isinstance(self.numerosity_reduction, (int, float)):
            raise TypeError("'numerosity_reduction' must be a boolean.")
        if not isinstance(self.smooth_idf, (int, float)):
            raise TypeError("'smooth_idf' must be a boolean.")
        if not isinstance(self.sublinear_tf, (int, float)):
            raise TypeError("'sublinear_tf' must be a boolean.")
        if not isinstance(overlapping, (int, float)):
            raise TypeError("'overlapping' must be a boolean.")

        self.vocabulary_ = {}

        if overlapping:
            n_windows = n_features - self.window_size + 1
            X_window = np.asarray([X[:, i: i + self.window_size]
                                   for i in range(n_windows)])
            X_window = X_window.reshape(n_samples * n_windows, -1, order='F')
        else:
            n_windows = n_features // self.window_size
            remainder = n_features % self. window_size
            if remainder == 0:
                window_idx = np.array_split(np.arange(0, n_features),
                                            n_windows)
            else:
                split_idx = np.arange(self.window_size,
                                      n_windows * (self.window_size + 1),
                                      self.window_size)
                window_idx = np.split(np.arange(0, n_features), split_idx)[:-1]
            X_window = X[:, window_idx].reshape(n_samples * n_windows, -1)

        sfa = SFA(self.n_coefs, False, self.norm_mean,
                  self.norm_std, self.n_bins, self.quantiles,
                  self.variance_selection, self.variance_threshold)
        tfidf = TfidfVectorizer(ngram_range=(1, 1), smooth_idf=self.smooth_idf,
                                sublinear_tf=self.sublinear_tf)

        X_sfa = sfa.fit_transform(X_window)
        X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_sfa).reshape(n_samples, -1)
        word_size = len(X_sfa[0, 0])
        if word_size == 1:
            tfidf.set_params(tokenizer=self._tok)
        if self.numerosity_reduction:
            X_sfa = np.apply_along_axis(numerosity_reduction, 1, X_sfa)
        else:
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)

        X_class = np.array([' '.join(X_sfa[y_ind == i])
                            for i in range(n_classes)])

        X_tfidf = tfidf.fit_transform(X_class)
        for key, value in tfidf.vocabulary_.items():
            self.vocabulary_[value] = key
        self._sfa = sfa
        self._tfidf = tfidf
        self.tfidf_ = X_tfidf
        return self

    def predict(self, X):
        """Transform the provided data.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]

        Returns
        -------
        X_new : sparse matrix, shape [n_samples, n_words]
            Document-term matrix.

        """
        # Check fitted
        check_is_fitted(self, ['tfidf_', '_sfa', '_tfidf', 'vocabulary_'])

        # Check X
        X = check_array(X)
        n_samples, n_features = X.shape

        n_windows = n_features - self.window_size + 1
        X_window = np.asarray([X[:, i: i + self.window_size]
                               for i in range(n_windows)])
        X_window = X_window.reshape(n_samples * n_windows, -1, order='F')

        X_sfa = self._sfa.transform(X_window)
        X_sfa = np.apply_along_axis(lambda x: ''.join(x),
                                    1,
                                    X_sfa).reshape(n_samples, -1)
        if self.numerosity_reduction:
            X_sfa = np.apply_along_axis(numerosity_reduction, 1, X_sfa)
        else:
            X_sfa = np.apply_along_axis(lambda x: ' '.join(x), 1, X_sfa)
        tf = self._tfidf.transform(X_sfa) / self._tfidf.idf_
        y_pred = cosine_similarity(tf, self.tfidf_).argmax(axis=1)
        return self._classes[y_pred]

    def _tok(self, x):
        return x.split(' ')
