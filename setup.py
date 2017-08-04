from setuptools import setup, find_packages
from distutils.core import setup


setup(name='pyts',
    version='0.4',
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
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib'
    ],
    zip_safe=False)