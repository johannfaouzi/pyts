"""
The :mod:`pyts.datasets` module tools for making, loading and fetching time
series datasets.
"""

from pyts.datasets._load import (
    load_basic_motions,
    load_coffee,
    load_gunpoint,
    load_pig_central_venous_pressure,
)
from pyts.datasets._make import make_cylinder_bell_funnel
from pyts.datasets._ucr import (
    fetch_ucr_dataset,
    ucr_dataset_info,
    ucr_dataset_list,
)
from pyts.datasets._uea import (
    fetch_uea_dataset,
    uea_dataset_info,
    uea_dataset_list,
)

__all__ = [
    'fetch_ucr_dataset',
    'fetch_uea_dataset',
    'load_basic_motions',
    'load_coffee',
    'load_gunpoint',
    'load_pig_central_venous_pressure',
    'make_cylinder_bell_funnel',
    'ucr_dataset_info',
    'ucr_dataset_list',
    'uea_dataset_info',
    'uea_dataset_list',
]
