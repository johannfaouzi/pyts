"""The :mod:`pyts.approximation` module includes approximation algorithms."""

from .paa import PiecewiseAggregateApproximation
from .sax import SymbolicAggregateApproximation
from .dft import DiscreteFourierTransform
from .mcb import MultipleCoefficientBinning
from .sfa import SymbolicFourierApproximation


__all__ = ['PiecewiseAggregateApproximation',
           'SymbolicAggregateApproximation',
           'DiscreteFourierTransform',
           'MultipleCoefficientBinning',
           'SymbolicFourierApproximation']
