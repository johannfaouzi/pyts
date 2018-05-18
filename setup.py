from setuptools import setup, find_packages
from distutils.core import setup


setup(name='pyts',
    version='0.7.0',
    description='A package for transformation and classification of time series',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
      ],
    keywords='time series machine learning transformation classification',
    project_urls={
    'Documentation': 'https://johannfaouzi.github.io/pyts/',
    'Source': 'https://github.com/johannfaouzi/pyts',
    'Tracker': 'https://github.com/johannfaouzi/pyts/issues',
    },
    author='Johann Faouzi',
    author_email='johann.faouzi@gmail.com',
    license='new BSD',
    packages=find_packages(),
    tests_require=['pytest'],
    install_requires=[
        'numpy>=1.8.2'
        'scipy>=0.13.3'
        'scikit-learn>=0.17.0'
        'future>=0.13.1'
    ],
    zip_safe=False)
