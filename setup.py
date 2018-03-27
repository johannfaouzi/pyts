from setuptools import setup, find_packages
from distutils.core import setup


setup(name='pyts',
    version='0.6.1',
    description='A package for transformation and classification of time series',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
        'Topic :: Scientific/Engineering',
      ],
    keywords='time series machine learning transformation classification',
    url='https://github.com/johannfaouzi/pyts',
    author='Johann Faouzi',
    author_email='johann.faouzi@gmail.com',
    license='MIT',
    packages=find_packages(),
    tests_require=['pytest'],
    install_requires=[
        'numpy',
        'scipy',
        'scikit-learn',
        'matplotlib',
        'future'
    ],
    zip_safe=False)
