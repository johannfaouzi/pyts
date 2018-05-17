User guide
==========

Introduction
------------

Time series are very common data and classifying them can be of interest in a
lot of fields. However tradional machine learning algorithms for classification,
like Logistic Regression, Support Vector Machine or K-Nearest Neighbors with
usual metrics, don't work very well. To be more precise, these algorithms
don't work well on **raw time series of real numbers**. Most algorithms
developed recently have been focusing on *transforming* the raw time series
before applying a standard machine learning classification algorithm.

In the following sections we'll present the algorithms implemented in pyts.

Preprocessing
-------------

It is standard in machine learning to perform some preprocessing on raw data.
Likewise it is standard to perform some preprocessing on time series. Implemented
algorithms can be found in the :mod:`pyts.preprocessing` module.

Currently the only preprocessing tool implemented is **StandardScaler**
It performs standardization (z-normalization) for each time series: the preprocessed
time series all have zero mean and unit variance.
It is implemented as :class:`pyts.preprocessing.StandardScaler`.

Approximation
-------------

Time series can be of huge size or be very noisy. It can be useful to sum up
the most important information of each time series. Implemented algorithms
to approximate a time series can be found in the :mod:`pyts.approximation` module.

The first algorithm implemented is **Piecewise Aggregate Approximation (PAA)**. The
main idea of this algorithm is to apply windows along a time series and to
take the mean value in each window. It is implemented as :class:`pyts.approximation.PAA`
and you can find an illustration of the algorithm in the ..auto_examples section.

The second algorithm implemented is **Discrete Fourier Transform (DFT)**. The idea
is to approximate a time series with a subsample of its Fourier coefficients.
The selected Fourier coefficients are either the first ones (as they represent
the trend of the time series) or the ones that separe the different classes
the most if a vector of class labels is provided.
It is implemented as :class:`pyts.approximation.DFT`.

Quantization
------------

One of the most interesting parts in time series classification is that several
state-of-the-art algorithms use text mining techniques for classification
and thus transform time series into bag of words. But first a time series
of real numbers needs to be transformed into a sequence of letters. Implemented
algorithms that quantize time series can be found in the :mod:`pyts.quantization` module.

The first algorithm implemented is **Symbolic Aggregate approXimation (SAX)**. For
eah time series, bins are computed using gaussian or empirical quantiles. Then
each datapoint is replaced by the bin it is in. It is implemented as
:class:`pyts.quantization.SAX`.

The second algorithm implemented is **Multiple Coefficient Binning (MCB)**. The idea
is very similar to SAX and the difference is that the quantization is applied
at each timestamp. It is implemented as :class:`pyts.quantization.MCB`.

The third algorithm implemented is **Symbolic Fourier Approximation (SFA)**.
It performs DFT then MCB, i.e. MCB is applied to the selected Fourier coefficients
of each time series. It is implemented as :class:`pyts.quantization.SFA`.

Bag of Words
------------

Now that you know how you can transform a time series of real numbers into
a sequence of letters, it's time to create bag of words. These algorithms are
can be found in the :mod:`pyts.bow` module.

The only algorithm implemented for the moment is **Bag of Words (BOW)**. It
applies a sliding window of fixed length along the sequence of letters to create
words. It is implemented as :class:`pyts.bow.BOW`.

Transformation
--------------

The :mod:`pyts.transformation` module consists of more complex algorithms that
transform a dataset of raw time series with shape [n_samples, n_timestamps] into
a more standard dataset of features with shape [n_samples, n_features] that
can be used as input data for a standard machine learning classification
algorithm.

The first algorithm implemented is **Bag-of-SFA Symbols (BOSS)**. Each time
series is first transformed into a bag of words using SFA and BOW. After this
transformation the features that are created are the frequencies of each word.
It is implemented as :class:`pyts.transformation.BOSS`.

The second algorithm implemented is **Word ExtrAction for time SEries cLassification (WEASEL)**.
The idea is similar to BOSS: first transform each time series into a bag of words
then compute the frequencies of each word. WEASEL is more sophisticated in the sense
that the selected Fourier coefficients are the most discrimative ones (based on the
one-way ANOVA test), several lengths for the sliding window are used and the most
discrimative features (i.e. words) are kept (based on the chi-2 test).
It is implemented as :class:`pyts.transformation.WEASEL`.

CLassification
--------------

The :mod:`pyts.classification` module consists of several classification
algorithms.

The first algorithm implemented is **K-Nearest Neighbors (KNN)**. For time
series classification it is the go-to algorithm for a good baseline. The most
common metrics used for time series classification are the Euclidean distance
and the Dynamic Time Warping distance.
It is implemented as :class:`pyts.classification.KNNClassifier`.

The second algorithm implemented is **SAX-VSM**. The outline of this algorithm is
to first transform raw time series into bags of words using SAX and BOW, then
merge, for each class label, all bags of words for this class label into only
one bag of words, and finally compute tf-idf for each bag of words. This leads
to a tf-idf vector for each class label. To predict an unlabeled time series,
this time series if first transformed into a term frequency vector, then the
predicted label is the one giving the highest cosine similarity among the tf-idf
vectors learned in the training phase.
It is implemented as :class:`pyts.classification.SAXVSMClassifier`.

The third algorithm implemented is **Bag-of-SFA Symbols in Vector Space (BOSSVS)**.
The outline of this algorithm is quite similar to the one of SAX-VSM but words
are created using SFA instead of SAX.
It is implemented as :class:`pyts.classification.BOSSVSClassifier`.

Image
-----

Instead of transforming a time series into a bag of words, it is also possible
to transform it into an image ! The :mod:`pyts.image` module consists of
several algorithms that perform that kind of transformation.

The first algorithm implemented is **Recurrence Plot**. It transforms a time series
into a matrix where each value corresponds to the distance between two trajectories
(a trajectory is a sub time series, i.e. a subsequence of back-to-back values
of a time series). The matrix can be binarized using a threshold.
It is implemented as :class:`pyts.image.RecurrencePlots`.

The second algorithm implemented is **Gramian Angular Field (GAF)**. First a
time series is represented as polar coordinates. Then the time series can be
transformed into a **Gramian Angular Summation Field (GASF)** when the cosine
of the sum of the angular coordinates is computed or a **Gramian Angular Difference
Field (GADF)** when the sine of the difference of the angular coordinates is computed.
It is implemented as :class:`pyts.image.GASF` and :class:`pyts.image.GADF`.

The third algorithm implemented is **Markov Transition Field (MTF)**. The outline
of the algorithm is to first quantize a time series using SAX, then to compute
the Markov transition matrix (the quantized time series is seen as a Markov chain)
and finally to compute the Markov transition field from the transition matrix.
It is implemented as :class:`pyts.image.MTF`.

Decomposition
-------------

The :mod:`pyts.decomposition` module consists of algorithms that decompose a
time series into several time series. The idea is to distinguish the different parts
of time series, such as the trend, the noise, etc.

The only algorithm implemented currently is **Singular Spectrum Analysis (SSA)**.
The outline of the algorithm is to first compute a matrix from a time series using lagged
vectors, then compute the eigenvalues and eigenvectors of this matrix multiplied by its
transpose, after compute the eigenmatrices and finally compute the time series for each
eigenmatrice.
It is implemented as :class:`pyts.decomposition.SSA`.
