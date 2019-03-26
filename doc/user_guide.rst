User guide
==========

Introduction
------------

Time series are very common data and classifying them can be of interest in a
lot of fields. However standard machine learning algorithms for classification,
like Logistic Regression, Support Vector Machine or K-Nearest Neighbors with
usual metrics, don't work very well. To be more precise, these algorithms
don't work well on **raw time series of real numbers**. Most algorithms
developed recently have been focusing on *transforming* the raw time series
before applying a standard machine learning classification algorithm.

In the following sections we will present the algorithms implemented in pyts. If
you want more information about the algorithms, you can have a look at the references
and the :ref:`general_examples` section. If you want more information about
their implementations in pyts, you can have a look at the :ref:`api` section.


Preprocessing
-------------

It is standard in machine learning to perform some preprocessing on raw data.
Likewise it is standard to perform some preprocessing on time series. Implemented
algorithms can be found in the :mod:`pyts.preprocessing` module.

pyts provides most of the preprocessing tools from scikit-learn, but instead of
applying them column-wise (i.e. independently to each feature), they are
applied row-wise (i.e. independently to each sample). Available tools are
:class:`pyts.preprocessing.StandardScaler`,
:class:`pyts.preprocessing.MinMaxScaler`,
:class:`pyts.preprocessing.MaxAbsScaler`,
:class:`pyts.preprocessing.RobustScaler`,
:class:`pyts.preprocessing.PowerTransformer`,
:class:`pyts.preprocessing.QuantileTransformer` and
:class:`pyts.preprocessing.KBinsDiscretizer`.

pyts also provides a tool to impute missing values using interpolation:
:class:`pyts.preprocessing.InterpolationImputer`.


Approximation
-------------

Time series can be of huge size or be very noisy. It can be useful to sum up
the most important information of each time series. Implemented algorithms
to approximate a time series can be found in the :mod:`pyts.approximation` module.

The first algorithm is **Piecewise Aggregate Approximation (PAA)**. The
main idea of this algorithm is to apply windows along a time series and to
take the mean value in each window. It is implemented as
:class:`pyts.approximation.PiecewiseAggregateApproximation`.

The second algorithm is **Symbolic Aggregate approXimation (SAX)**. For
each time series, bins are computed using a given strategy. Then
each datapoint is replaced by the index of the bin it belongs to. It is
implemented as :class:`pyts.approximation.SymbolicAggregateApproximation`.
It is similar to :class:`pyts.preprocessing.KBinsDiscretizer`.

The third algorithm is **Discrete Fourier Transform (DFT)**. The idea
is to approximate a time series with a subsample of its Fourier coefficients.
The selected Fourier coefficients are either the first ones (as they represent
the trend of the time series) or the ones that discriminate the different classes
the most if a vector of class labels is provided.
It is implemented as :class:`pyts.approximation.DiscreteFourierTransform`.

The fourth algorithm is **Multiple Coefficient Binning (MCB)**. The idea
is very similar to SAX and the difference is that the discretization is done
column-wise instead of row-wise. It is implemented as
:class:`pyts.approximation.MultipleCoefficientBinning`.

The fifth algorithm is **Symbolic Fourier Approximation (SFA)**.
It performs DFT followed MCB, i.e. the selected Fourier coefficients
of each time series are discretized. It is implemented as
:class:`pyts.approximation.SymbolicFourierApproximation`.

References
^^^^^^^^^^

- Eamonn J. Keogh and Michael J. Pazzani.
  A simple dimensionality reduction technique for fast similarity search in
  large time series databases. *Knowledge Discovery and Data Mining* ,2000.

- Christos Faloutsos, M. Ranganathan and Yannis Manolopoulos.
  Fast Subsequence Matching in Time-Series Databases. *ACM SIGMOD Record*, 2000.

- Jessica Lin, Eamonn Keogh, Li Wei, and Stefano Lonardi. Experiencing SAX: a Novel
  Symbolic Representation of Time Series. *Data Mining and Knowledge Discovery*, 2007.

- Patrick Schäfer and Mikael Högqvist. SFA: A Symbolic Fourier Approximation
  and Index for Similarity Search in High Dimensional Datasets.
  *ACM International Conference Proceeding Series*, 2012.


Bag of Words
------------

Now that you know how you can transform a time series of real numbers into
a sequence of letters, it's time to create bag of words. These algorithms are
can be found in the :mod:`pyts.bag_of_words` module.

The only algorithm is **Bag of Words (BOW)**. It applies a sliding window of
fixed length along the sequence of letters to create words. It is implemented
as :class:`pyts.bag_of_words.BagOfWords`.


Metrics
-------

It is often of interest to be able to compare time series. However, standard
metrics like the Euclidean distance are not always well-suited for time series.
To tackle this issue, metrics specific to time series have been developed.
pyts provides implementations for some of them in the :mod:`pyts.metrics` module.

The most famous metric is **Dynamic Time Warping (DTW)**. It computes the Euclidean
distance on the optimal path between two time series. This metric is
computationally expensive, thus several variants of DTW have been developed.
The ones available in pyts are DTW with a region constraint (Sakoe-Chiba band,
Itakura parallelogram), MultiscaleDTW and FastDTW, as well as the classic DTW.
Classic DTW and its variants can all be used with a single function:
:func:`pyts.metrics.dtw`.

Another metric available in this package is the **BOSS** metric. This metric
has been introduced with the **BOSS** algorithm (see below) and computes the
Euclidean distance between two time series, but only using the indices where
the first time series is not equal to zero. This metric is usually not applied
on time series directly, but after the transformation from the BOSS algorithm,
where each time series is replaced with its histogram of words.

References
^^^^^^^^^^

- Meinard Müller. Dynamic Time Warping (DTW).
  *Information Retrieval for Music and Motion*, 2007.

- Patrick Schäfer. The BOSS is concerned with time series classification in
  the presence of noise. *Data Mining and Knowledge Discovery*, 2015.


Transformation
--------------

The :mod:`pyts.transformation` module consists of more complex algorithms that
transform a dataset of raw time series with shape `(n_samples, n_timestamps)`
into a more standard dataset of features with shape `(n_samples, n_features)`
that can be used as input data for a standard machine learning classification
algorithm.

The first algorithm is **Bag-of-SFA Symbols (BOSS)**. Each time
series is first transformed into a bag of words using SFA and BOW. Then the
frequencies of each word are computed.
It is implemented as :class:`pyts.transformation.BOSS`.

The second algorithm is **Word ExtrAction for time SEries cLassification (WEASEL)**.
The idea is similar to BOSS: first transform each time series into a bag of words
then compute the frequencies of each word. WEASEL is more sophisticated in the sense
that the selected Fourier coefficients are the most discrimative ones (based on the
one-way ANOVA test), several lengths for the sliding window are used and the most
discriminative features (i.e. words) are kept (based on the chi-2 test).
It is implemented as :class:`pyts.transformation.WEASEL`.

References
^^^^^^^^^^

- Patrick Schäfer. The BOSS is concerned with time series classification in
  the presence of noise. *Data Mining and Knowledge Discovery*, 2015.

- Patrick Schäfer and Ulf Leser. Fast and Accurate Time Series Classification with WEASEL.
  *CoRR*, 2017.

Classification
--------------

The :mod:`pyts.classification` module consists of several classification
algorithms.

The first algorithm implemented is **K-Nearest Neighbors (KNN)**. For time
series classification it is the go-to algorithm for a good baseline. The most
common metrics used for time series classification are the Euclidean distance
and the Dynamic Time Warping distance. It extends the implementation from
scikit-learn with more metrics available.
It is implemented as :class:`pyts.classification.KNeighborsClassifier`.

The second algorithm implemented is **SAX-VSM**. The outline of this algorithm is
to first transform raw time series into bags of words using SAX and BOW, then
merge, for each class label, all bags of words for this class label into only
one bag of words, and finally compute tf-idf statistics for each bag of words. This leads
to a tf-idf vector for each class label. To predict an unlabeled time series,
this time series if first transformed into a term frequency vector, then the
predicted label is the one giving the highest cosine similarity among the tf-idf
vectors learned in the training phase.
It is implemented as :class:`pyts.classification.SAXVSM`.

The third algorithm implemented is **Bag-of-SFA Symbols in Vector Space (BOSSVS)**.
The outline of this algorithm is quite similar to the one of SAX-VSM but words
are created using SFA instead of SAX.
It is implemented as :class:`pyts.classification.BOSSVS`.

References
^^^^^^^^^^

- Senin Pavel and Malinchik Sergey. SAX-VSM: Interpretable Time Series
  Classification Using SAX and Vector Space Model. *Data Mining (ICDM),
  2013 IEEE 13th International Conference on, pp.1175,1180*, 2013.

- Patrick Schäfer. Scalable Time Series Classification. *DMKD* and *ECML/PKDD*, 2016.

Image
-----

Instead of transforming a time series into a bag of words, it is also possible
to transform it into an image ! The :mod:`pyts.image` module consists of
several algorithms that perform that kind of transformation.

The first algorithm implemented is **Recurrence Plot**. It transforms a time series
into a matrix where each value corresponds to the distance between two trajectories
(a trajectory is a sub time series, i.e. a subsequence of back-to-back values
of a time series). The matrix can be binarized using a threshold.
It is implemented as :class:`pyts.image.RecurrencePlot`.

The second algorithm implemented is **Gramian Angular Field (GAF)**. First a
time series is represented as polar coordinates. Then the time series can be
transformed into a **Gramian Angular Summation Field (GASF)** when the cosine
of the sum of the angular coordinates is computed or a **Gramian Angular Difference
Field (GADF)** when the sine of the difference of the angular coordinates is computed.
It is implemented as :class:`pyts.image.GramianAngularField`

The third algorithm implemented is **Markov Transition Field (MTF)**. The outline
of the algorithm is to first quantize a time series using SAX, then to compute
the Markov transition matrix (the quantized time series is seen as a Markov chain)
and finally to compute the Markov transition field from the transition matrix.
It is implemented as :class:`pyts.image.MarkovTransitionField`.

References
^^^^^^^^^^

- Jean-Pierre Eckmann, Sylvie Oliffson Kamphorst and David Ruelle.
  Recurrence Plots of Dynamical Systems. *Europhysics Letters*, 1987.

- Nima Hatami, Yann Gavet and Johan Debayle. Classification of Time-Series
  Images Using Deep Convolutional Neural Networks. *arXiv:1710.00886 [cs]*, 2017.

- Zhiguang Wang and Tim Oates. Imaging time-series to improve classification and imputation.
  *Proceedings of the 24th International Conference on Artificial Intelligence*, 2015.

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
It is implemented as :class:`pyts.decomposition.SingularSpectrumAnalysis`.

References
^^^^^^^^^^^

- Nina Golyandina and Anatoly Zhigljavsky.
  Singular Spectrum Analysis for Time Series. 2013
