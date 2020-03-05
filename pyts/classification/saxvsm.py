"""Code for SAX-VSM."""

# Author: Johann Faouzi <johann.faouzi@gmail.com>
# License: BSD-3-Clause

from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from ..bag_of_words import BagOfWords


class SAXVSM(BaseEstimator, ClassifierMixin):
    """Classifier based on SAX-VSM representation and tf-idf statistics.

    Time series are first transformed into bag of words using Symbolic
    Aggregate approXimation (SAX) algorithm followed by a bag-of-words
    model. Then the classes are transformed into a Vector Space Model
    (VSM) using term frequencies (tf) and inverse document frequencies
    (idf).

    Parameters
    ----------
    window_size : int or float (default = 0.5)
        Length of the sliding window. If float, it represents
        a percentage of the size of each time series and must be
        between 0 and 1.

    word_size : int or float (default = 0.5)
        Length of the words. If float, it represents
        a percentage of the length of the sliding window and must be
        between 0. and 1.

    n_bins : int (default = 4)
        The number of bins to produce. It must be between 2 and
        ``min(window_size, 26)``.

    strategy : 'uniform', 'quantile' or 'normal' (default = 'normal')
        Strategy used to define the widths of the bins:

        - 'uniform': All bins in each sample have identical widths
        - 'quantile': All bins in each sample have the same number of points
        - 'normal': Bin edges are quantiles from a standard normal distribution

    numerosity_reduction : bool (default = True)
        If True, delete sample-wise all but one occurence of back to back
        identical occurences of the same words.

    window_step : int or float (default = 1)
        Step of the sliding window. If float, it represents the percentage of
        the size of each time series and must be between 0 and 1. The step of
        sliding window will be computed as
        ``ceil(window_step * n_timestamps)``.

    norm_mean : bool (default = True)
        If True, center each subseries before scaling.

    norm_std : bool (default = True)
        If True, scale each subseries to unit variance.

    use_idf : bool (default = True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : bool (default = False)
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    sublinear_tf : bool (default = True)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    overlapping : bool (default = True)
        If True, time points may belong to two bins when decreasing the size
        of the subsequence with the Piecewise Aggregate Approximation
        algorithm. If False, each time point belong to one single bin, but
        the size of the bins may vary.

    alphabet : None or array-like, shape = (n_bins,)
        Alphabet to use. If None, the first `n_bins` letters of the Latin
        alphabet are used.

    Attributes
    ----------
    classes_ : array, shape = (n_classes,)
        An array of class labels known to the classifier.

    idf_ : array, shape = (n_features,) , or None
        The learned idf vector (global term weights) when ``use_idf=True``,
        None otherwise.

    tfidf_ : array, shape = (n_classes, n_words)
        Term-document matrix.

    vocabulary_ : dict
        A mapping of feature indices to terms.

    References
    ----------
    .. [1] P. Senin, and S. Malinchik, "SAX-VSM: Interpretable Time Series
           Classification Using SAX and Vector Space Model". International
           Conference on Data Mining, 13, 1175-1180 (2013).

    Examples
    --------
    >>> from pyts.classification import SAXVSM
    >>> from pyts.datasets import load_gunpoint
    >>> X_train, X_test, y_train, y_test = load_gunpoint(return_X_y=True)
    >>> clf = SAXVSM(window_size=64, word_size=12, n_bins=5, strategy='normal')
    >>> clf.fit(X_train, y_train)
    SAXVSM(...)
    >>> clf.score(X_test, y_test)
    0.9933...

    """

    def __init__(self, window_size=0.5, word_size=0.5, n_bins=4,
                 strategy='normal', numerosity_reduction=True, window_step=1,
                 norm_mean=True, norm_std=True, use_idf=True, smooth_idf=False,
                 sublinear_tf=True, overlapping=True, alphabet=None):
        self.window_size = window_size
        self.word_size = word_size
        self.n_bins = n_bins
        self.strategy = strategy
        self.numerosity_reduction = numerosity_reduction
        self.window_step = window_step
        self.norm_mean = norm_mean
        self.norm_std = norm_std
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.overlapping = overlapping
        self.alphabet = alphabet

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
        check_classification_targets(y)
        le = LabelEncoder()
        y_ind = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = self.classes_.size

        # Transform each time series into a bag of words
        bow = BagOfWords(
            window_size=self.window_size, word_size=self.word_size,
            n_bins=self.n_bins, strategy=self.strategy,
            numerosity_reduction=self.numerosity_reduction,
            window_step=self.window_step, norm_mean=self.norm_mean,
            norm_std=self.norm_std, overlapping=self.overlapping,
            alphabet=self.alphabet
        )
        X_bow = bow.fit_transform(X)

        X_class = [' '.join(X_bow[y_ind == classe])
                   for classe in range(n_classes)]

        tfidf = TfidfVectorizer(
            norm=None, use_idf=self.use_idf, smooth_idf=self.smooth_idf,
            sublinear_tf=self.sublinear_tf
        )
        self.tfidf_ = tfidf.fit_transform(X_class).A
        self.vocabulary_ = {value: key for key, value in
                            tfidf.vocabulary_.items()}
        if self.use_idf:
            self.idf_ = tfidf.idf_
        else:
            self.idf_ = None
        self._tfidf = tfidf
        self._bow = bow
        return self

    def decision_function(self, X):
        """Evaluate the cosine similarity between document-term matrix and X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_timestamps)
            Test samples.

        Returns
        -------
        X : array-like, shape (n_samples, n_classes)
            osine similarity between the document-term matrix and X.

        """
        check_is_fitted(self, ['vocabulary_', 'tfidf_', 'idf_',
                               '_tfidf', 'classes_'])
        X = check_array(X)
        X_bow = self._bow.transform(X)
        vectorizer = CountVectorizer(vocabulary=self._tfidf.vocabulary_)
        X_transformed = vectorizer.transform(X_bow).toarray()
        return cosine_similarity(X_transformed, self.tfidf_)

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
        return self.classes_[self.decision_function(X).argmax(axis=1)]
