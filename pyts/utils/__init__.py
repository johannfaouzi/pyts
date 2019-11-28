"""The :mod:`pyts.utils` module includes utility tools."""

from .utils import segmentation, windowed_view
from .deprecation import deprecated


__all__ = ['segmentation', 'windowed_view', 'deprecated']
