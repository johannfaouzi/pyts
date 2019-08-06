"""
The :mod:`pyts.datasets` module tools for making, loading and fetching time
series datasets.
"""

from .ucr import ucr_dataset_info, ucr_dataset_list, fetch_ucr_dataset
from .uea import uea_dataset_info, uea_dataset_list, fetch_uea_dataset
from .make import make_cylinder_bell_funnel

__all__ = ['ucr_dataset_info', 'ucr_dataset_list', 'fetch_ucr_dataset',
           'uea_dataset_info', 'uea_dataset_list', 'fetch_uea_dataset',
           'make_cylinder_bell_funnel']
