"""Time series transformation and classification module for Python.

pyts is a Python module integrating classical and recently published
algorithms used for time series transformation and classification.

It aims to provide implementations for state-of-the-art as well as
recently published algorithms for time series transformation and
classification.
"""

from . import (preprocessing, approximation, quantization, bow, decomposition,
               image, transformation, classification, utils)

__version__ = '0.7.0'

__all__ = ["preprocessing",
           "approximation",
           "quantization",
           "bow",
           "decomposition",
           "image",
           "transformation",
           "classification",
           "utils"]
