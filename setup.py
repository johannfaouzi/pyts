"""A python package for time series classification."""

import pyts
from setuptools import find_packages, setup


DISTNAME = 'pyts'
DESCRIPTION = 'A python package for time series classification'
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
MAINTAINER = 'Johann Faouzi'
MAINTAINER_EMAIL = 'johann.faouzi@gmail.com'
URL = 'https://github.com/johannfaouzi/pyts'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/johannfaouzi/pyts'
VERSION = pyts.__version__
INSTALL_REQUIRES = ['numpy>=1.17.5',
                    'scipy>=1.3.0',
                    'scikit-learn>=0.22.1',
                    'joblib>=0.12',
                    'numba>=0.48.0']
CLASSIFIERS = ['Development Status :: 3 - Alpha',
               'Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7',
               'Programming Language :: Python :: 3.8']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx==1.8.5',
        'sphinx-gallery',
        'numpydoc',
        'matplotlib'
    ]
}
PACKAGE_DATA = {
    'pyts': ['datasets/cached_datasets/UCR/Coffee/*.txt',
             'datasets/cached_datasets/UCR/GunPoint/*.txt',
             'datasets/cached_datasets/UCR/PigCVP/*.txt',
             'datasets/cached_datasets/UEA/BasicMotions/*.arff',
             'datasets/cached_datasets/UEA/BasicMotions/*.txt',
             'datasets/info/*.pickle']
}

setup(name=DISTNAME,
      maintainer=MAINTAINER,
      maintainer_email=MAINTAINER_EMAIL,
      description=DESCRIPTION,
      license=LICENSE,
      url=URL,
      version=VERSION,
      download_url=DOWNLOAD_URL,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
      zip_safe=False,
      classifiers=CLASSIFIERS,
      packages=find_packages(),
      package_data=PACKAGE_DATA,
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
