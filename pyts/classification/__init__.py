"""The :mod:`pyts.classification` module includes classification algorithms."""

from .bosssp import BOSSSP
from .bossvs import BOSSVS
from .learning_shapelets import LearningShapelets
from .knn import KNeighborsClassifier
from .saxvsm import SAXVSM
from .time_series_forest import TimeSeriesForest
from .tsbf import TSBF

__all__ = ['BOSSSP', 'BOSSVS', 'KNeighborsClassifier', 'LearningShapelets', 'SAXVSM',
           'TimeSeriesForest', 'TSBF']
