"""A python package for time series transformation and classification."""

import pyts
from setuptools import find_packages, setup


DISTNAME = 'pyts'
DESCRIPTION = ('A python package for time series transformation '
               'and classification')
with open('README.md') as f:
    LONG_DESCRIPTION = f.read()
LONG_DESCRIPTION_CONTENT_TYPE = 'text/markdown'
MAINTAINER = 'Johann Faouzi'
MAINTAINER_EMAIL = 'johann.faouzi@gmail.com'
URL = 'https://github.com/johannfaouzi/pyts'
LICENSE = 'new BSD'
DOWNLOAD_URL = 'https://github.com/johannfaouzi/pyts'
VERSION = pyts.__version__
INSTALL_REQUIRES = ['numpy>=1.15.4'
                    'scipy>=1.2.1'
                    'scikit-learn>=0.20.1'
                    'numba>=0.41.0']
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
               'Programming Language :: Python :: 3.5',
               'Programming Language :: Python :: 3.6',
               'Programming Language :: Python :: 3.7']
EXTRAS_REQUIRE = {
    'tests': [
        'pytest',
        'pytest-cov'],
    'docs': [
        'sphinx == 1.8.2',
        'sphinx-gallery',
        'sphinx_rtd_theme',
        'numpydoc',
        'matplotlib'
    ]
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
      install_requires=INSTALL_REQUIRES,
      extras_require=EXTRAS_REQUIRE)
