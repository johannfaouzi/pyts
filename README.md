Welcome to **pyts**, a Python package for time series transformation and classification !
===================

### **pyts** provides an implementation of:
- Transformation :
	- StandardScaler : zero mean and unit variance for each time series
	- PAA : Piecewise Aggregate Approximation
	- SAX : Symbolic Aggregate approXimation
	- VSM : Vector Space Model
	- GAF : Grammian Angular Field
	- MTF : Markov Transition Field
- Classification :
	- SAX-VSM : algorithm based on SAX-VSM transformation that uses tf-idf statistics to perform classification
	- kNN : k nearest neighbors (scikit-learn implementation) with two more metrics : DTW and FastDTW

### **pyts** provides three modules:
- transformation
- classification
- visualization

#### Notes:
- This package follows the spirit of scikit-learn : transformers and classifiers can be used with Pipeline and GridSearchCV. Thus, every class from transformers or classifiers expect all the time series to have the same length.
- The module **visualization** allows you to see what most transformers do (PAA, SAX, PAA-SAX, GAF, MTF) as well as DTW and FastDTW.

### 0. Installation
#### 0.1 Installation with pip
**pyts** is available on PyPI and can be easily installed with the command **pip**:

	pip install pyts

#### 0.2 Package dependencies
**pyts** requires the following packages to be installed:

- numpy
- scipy
- scikit-learn
- matplotlib
- math

### 1. Notations

We consider our dataset to have the following requirements:

- `X` is a `numpy.ndarray` with shape `(n_samples, n_features)`.
- `y` is a `numpy.array` of length `n_samples`.

You can simulate a toy dataset with the following code:

```python
import numpy as np
from scipy.stats import norm
	
n_samples = 10
n_features = 48
n_classes = 2
	
delta = 0.5
dt = 1
x = 0.
	
X = np.zeros((n_samples, n_features))
X[:, 0] = x
	
for i in range(n_samples):
    start = x
    for k in range(1, n_features):
        start += norm.rvs(scale=delta**2 * dt)
        X[i][k] = start
	
y = np.random.randint(n_classes, size=n_samples)
```

You can easily plot a time series with the function `plot_ts` from the module `visualization`:

```python
from pyts.visualization import plot_ts

plot_ts(X[0])
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts.png">
</p>


### 2. StandardScaler

Standard normalization (also known as z-normalization) is a common preprocessing step : it applies an affine transformation to a time series such that the new time series has zero mean and (almost) unit variance: we add a small term *epsilon* to the standard deviation before dividing (in order to prevent a division by zero and to avoid over-amplification of an eventual background noise).

It is implemented as a class named `StandardScaler(epsilon)` from the module `transformation`.

```python
from pyts.transformation import StandardScaler

standardscaler = StandardScaler(epsilon=1e-2)
X_standardized = standardscaler.transform(X)
```
	
The function `plot_standardscaler` from the module `visualization` allows you to see the time series before and after the transformation.

```python
from pyts.visualization import plot_standardscaler

plot_standardscaler(X[0])
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_ss.png">
</p>


### 3. Piecewise Aggregation Approximation (PAA)

Piecewise Aggregation Approximation is a dimension reduction technique. It splits the time series into bins and computes the mean inside each bin. It returns the sample of means. It is implemented as a class named `PAA(window_size=None, output_size=None, overlapping=True)`, from the module `transformation`. 

You can specify either `window_size` or `output_size`. If you specify `output_size`, the returned time series will be of length `output_size`. You can also specify `overlapping`, which will determine if each bin has the same number of elements `(overlapping=True)` or if each datapoint belong to only one bin `(overlapping=False)`. If you specify `window_size`, the size of each bin will be equal to `window_size` and `overlapping` will be ignored.

Here is the code to perform the transformation:

```python
from pyts.transformation import PAA

paa = PAA(window_size=None, output_size=8, overlapping=True)
X_paa = paa.transform(X_standardized)
```
	
The function `plot_paa` from the module `visualization` allows you to see the time series before and after the transformation.

```python
from pyts.visualization import plot_paa

plot_paa(X_standardized[0], window_size=None, output_size=8, overlapping=True, marker='o')
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_paa.png">
</p>


### 4. Symbolic Aggregation approXimation (SAX)

Symbolic Aggregation approXimation transforms a time series into a string. More precisely, given a number of bins, it transforms each datapoint (real number) into a letter (from an alphabet) depending on the bin the datapoint belongs to. For instance, If your alphabet is "abcde", then the state space changes from R to {a,b,c,d,e}. It is a dimension reduction technique for the output space.

Here is the code to perform the transformation:

```python
from pyts.transformation import SAX

sax = SAX(n_bins=5, quantiles='gaussian')
X_sax = sax.transform(X_paa)
```
	
The function `plot_sax` from the module `visualization` allows you to see the time series before and after the transformation. It may help you understand what SAX does.

```python
from pyts.visualization import plot_sax

plot_sax(X_paa[0], n_bins=5, quantiles='gaussian')
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_sax.png">
</p>

The function `plot_paa_sax` from the module `visualization` allows you to see both PAA and SAX transformations on the same figure.

```python
from pyts.visualization import plot_paa_sax

plot_paa_sax(X_standardized[0], window_size=None, output_size=8, overlapping=True, n_bins=5, quantiles='gaussian')
```


<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_paa_sax.png">
</p>


### 5. Vector Space Model (VSM)
Vector Space Model transforms a long string into a set of small words. It slides a window with a step of size 1 across the long string and extract each words. It is implemented as a class named `VSM(window_size=4, numerosity_reduction=True)`. It is possible that the same word occurs several times in a row. If `numerosity_reduction=True`, only one will be kept. If `numerosity_reduction=False`, all words are kept.

Here is the code to perform the transformation:

```python
from pyts.transformation import VSM

vsm = VSM(window_size=4, numerosity_reduction=True)
X_vsm = vsm.transform(X_sax)
```

### 6. Gramian Angular Field (GAF)
Gramian Angular Field transforms a time series into an image. First the time series is represented in a polar coordinate system. Then two matrices can be computed : the cosine of the summation of angles (Gramian Angular Summation Field - GASF) or the sine of the difference of angles (Gramian Angular Difference Field - GADF). One possible issue is the size of these matrices: if the size of the time series is n, then the size of the matrix is (n, n). To avoid this issue, PAA can be applied to reduce the size of the time series first.

Both methods are implemented in two classes : `GASF(image_size=32, overlapping=False, scale='-1')` and `GADF(image_size=32, overlapping=False, scale='-1')` from the module `transformation`. `image_size` is an integer (since the output will be a square matrix, only one dimension needs to be specified) and corresponds to `output_size` for PAA. `overlapping` is the parameter for PAA too. In order to use the polar coordinate system, the time series should be normalized. If `scale='-1'`, the time series is normalized into [-1,1]. If `scale='0'`, the time series is normalized into [0,1].

Here is the code to perform the transformation:

```python
from pyts.transformation import GASF, GADF

gasf = GASF(image_size=24, overlapping=False, scale='-1')
X_gasf = gasf.transform(X_standardized)

gadf = GADF(image_size=24, overlapping=False, scale='-1')
X_gadf = gadf.transform(X_standardized)
```

The functions `plot_gasf` and `plot_gadf` from the module `visualization` allows you to see the image.

```python
from pyts.visualization import plot_gasf

plot_gasf(X_standardized[0], image_size=48, overlapping=False, scale='-1')
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_gasf.png">
</p>


```python
from pyts.visualization import plot_gadf

plot_gadf(X_standardized[0], image_size=48, overlapping=False, scale='-1')
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_gadf.png">
</p>


### 7. Markov Transition Field (MTF)
Markov Transition Field also transforms a time series into an image. First the time series is transformed into a string using SAX. This string is interpreted as a sequence of observations from a Markov Chain. Then, the matrix transition probability is computed. The issue is that temporal dependices are lost. To avoid this issue, the transition matrix is turned into a transition field. Here again, if the size of the time series is n, then the size of the matrix is (n, n). To avoid this issue, the size of the image is reduced by averaging pixels with a blurring kernel (PAA for 2D arrays).

This method is implemented as a class named `MTF(image_size=32, n_bins=8, quantiles='empirical', overlapping=False)` from the module `transformation`. `image_size` and `overlapping` are the parameters for the dimension reduction of the image (see PAA), while `n_bins` and `quantiles` are the parameters for SAX.

Here is the code to perform the transformation:
```python
from pyts.transformation import MTF

mtf = MTF(image_size=48, n_bins=4, quantiles='empirical', overlapping=False)
X_mtf = mtf.transform(X_standardized)
```

The function `plot_mtf` from the module `visualization` allows you to see the image.
```python
from pyts.visualization import plot_mtf

plot_mtf(X_standardized[0], image_size=48, n_bins=4, quantiles='empirical', overlapping=False)
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/ts_mtf.png">
</p>

### 8. SAX-VSM Classification
SAX-VSM is a classification method based on the SAX-VSM representation of the time series. Once this transformation is done, tf-idf statistics are computed, where a document is a time series and each corpus corresponds to a class. To predict the class with an unlabeled time series, tf is computed for this time series and the predicted label corresponds to the class that gives the highest cosine similarity between tf and its tf-idf.

Here is to code to perform a classification:

```python
from pyts.classification import SAXVSMClassifier

clf = SAXVSMClassifier()
clf.fit(X_vsm[:80], y[:80])
y_pred = clf.predict(X_vsm[80:])
clf.score(X_vsm[80:], y[80:])
```

### 9. kNN with DTW or FastDTW
k nearest neighbors is an often used classifier when dealing with time series. Typical metrics are Minkowski distance and dynamic time warping. Unfortunately, dynamic time warping is not available with scikit-learn implementation of kNN. The class `KNNClassifier` from the module `classification` uses scikit-learn implementation of kNN but allows you to use this metric.

Here is the code to perform a classification:

```python
from pyts.classifier import KNNClassifier

clf = KNNClassifier(metric='minkowski', p=2)
clf.fit(X_standardized[:80], y[:80])
y_pred = clf.predict(X_standardized[80:])
clf.score(X_standardized[80:], y[80:])
```
<!-- tsk -->

```python
clf.set_params(metric='fast_dtw', metric_params={"approximation": True, "window_size": 8})
clf.fit(X_standardized[:80], y[:80])
y_pred = clf.predict(X_standardized[80:])
clf.score(X_standardized[80:], y[80:])
```

#### More about DTW and FastDTW

We refer to [4] for more information about DTW and FastDTW.

The function `plot_dtw(X, Y)` from the module `visualization` allows you to see the optimal warping path between two time series with same length.

```python
from pyts.visualization import plot_dtw

plot_dtw(X[0], X[1])
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/dtw.png">
</p>

The function `plot_fastdtw(X, Y, window_size)` from the module `visualization` allows you to see the optimal warping path between two time series with same length, as well as the region constraint.

```python
from pyts.visualization import plot_fastdtw

plot_fastdtw(X[0], X[1], window_size=12)
```

<p align="center">
<img src="https://raw.githubusercontent.com/johannfaouzi/pyts/master/pictures/fastdtw.png">
</p>

### 10. Working with Pipeline and GridSearchCV
All the classes implemented in the modules `` and `` can be used with Pipeline and GridSearchCV from scikit-learn package.

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([("sc", StandardScaler()),
                     ("paa", PAA(output_size=12, overlapping=True)),
                     ("sax", SAX(n_bins=5, quantiles='gaussian')),
                     ("vsm", VSM(window_size=4, numerosity_reduction=True)),
                     ("clf", SAXVSMClassifier())
                    ])

pipeline.fit(X[:80], y[:80])
y_pred = pipeline.predict(X[80:])
pipeline.score(X[80:], y[80:])
```

<!-- tsk -->

```python
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([("sc", StandardScaler()),
                     ("paa", PAA(output_size=13)),
                     ("sax", SAX(n_bins=5, quantiles='gaussian')),
                     ("vsm", VSM(numerosity_reduction=True)),
                     ("clf", SAXVSMClassifier())
                    ])
                    
parameters = {"paa__overlapping": [True, False], "vsm__window_size": [3, 4, 5]}

clf = GridSearchCV(pipeline, parameters)
clf.fit(X[:80], y[:80])
y_pred = clf.predict(X[80:])
clf.score(X[80:], y[80:])
```

#### References :
[1] Eamonn Keogh, Kaushik Chakrabarti, Michael Pazzani, and Sharad Mehrotra. Dimensionality reduction for fast similarity search in large time series databases. *Knowledge and information Systems, 3(3), 263-286*, 2001.

[2] Jessica Lin, Eamonn Keogh, Li Wei, and Stefano Lonardi. Experiencing SAX: a Novel Symbolic Representation of Time Series. *Data Mining and Knowledge Discovery*, 2007.

[3] Senin Pavel, and Malinchik Sergey. SAX-VSM: Interpretable Time Series Classification Using SAX and Vector Space Model. *Data Mining (ICDM), 2013 IEEE 13th International Conference on, pp.1175,1180*, 2013.

[4] Dynamic Time Warping (DTW). *SpringerReference*.

[5] Zhiguang Wang, and Tim Oates. Imaging time-series to improve classification and imputation. *Proceedings of the 24th International Conference on Artificial Intelligence*, 2015.