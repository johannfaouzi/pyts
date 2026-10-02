"""The :mod:`pyts.approximation` module includes approximation algorithms."""

from pyts.approximation._dft import DiscreteFourierTransform
from pyts.approximation._mcb import MultipleCoefficientBinning
from pyts.approximation._paa import PiecewiseAggregateApproximation
from pyts.approximation._sax import SymbolicAggregateApproximation
from pyts.approximation._sfa import SymbolicFourierApproximation

__all__ = [
    'DiscreteFourierTransform',
    'MultipleCoefficientBinning',
    'PiecewiseAggregateApproximation',
    'SymbolicAggregateApproximation',
    'SymbolicFourierApproximation',
]
