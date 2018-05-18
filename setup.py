from setuptools import setup, find_packages
from distutils.core import setup


setup(name='pyts',
    version='0.7.0',
    description='A package for transformation and classification of time series',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: BSD-3 License',
        'Programming Language :: Python :: 2.7',
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
        'numpy>=1.8.2'
        'scipy>=0.13.3'
        'scikit-learn>=0.17.0'
        'future>=0.13.1'
    ],
    zip_safe=False)
