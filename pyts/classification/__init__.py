"""The :mod:`pyts.classification` module includes classification algorithms."""

from .bossvs import BOSSVS
from .knn import KNeighborsClassifier
from .saxvsm import SAXVSM

__all__ = ['BOSSVS', 'KNeighborsClassifier', 'SAXVSM']
