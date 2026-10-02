"""The :mod:`pyts.utils` module includes utility tools."""

from pyts.utils._deprecation import deprecated
from pyts.utils._utils import segmentation, windowed_view

__all__ = ['deprecated', 'segmentation', 'windowed_view']
