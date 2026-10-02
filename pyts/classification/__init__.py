"""The :mod:`pyts.classification` module includes classification algorithms."""

from pyts.classification._bossvs import BOSSVS
from pyts.classification._knn import KNeighborsClassifier
from pyts.classification._learning_shapelets import LearningShapelets
from pyts.classification._saxvsm import SAXVSM
from pyts.classification._time_series_forest import TimeSeriesForest
from pyts.classification._tsbf import TSBF

__all__ = [
    'BOSSVS',
    'SAXVSM',
    'TSBF',
    'KNeighborsClassifier',
    'LearningShapelets',
    'TimeSeriesForest',
]
