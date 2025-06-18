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
INSTALL_REQUIRES = ['numpy>=1.22.4',
                    'scipy>=1.8.1',
                    'scikit-learn>=1.2.0',
                    'joblib>=1.1.1',
                    'numba>=0.55.2']
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
               'Programming Language :: Python :: 3.8',
               'Programming Language :: Python :: 3.9',
               'Programming Language :: Python :: 3.10',
               'Programming Language :: Python :: 3.11']
EXTRAS_REQUIRE = {
    'linting': [
        'flake8'
    ],
    'tests': [
        'pytest',
        'pytest-cov'
    ],
    'docs': [
        'docutils==0.14',
        'sphinx==1.8.5',
        'alabaster==0.7.12',
        'sphinx-gallery',
        'numpydoc',
        'matplotlib',
        'packaging',
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
