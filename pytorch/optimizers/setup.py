#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='pytorch_optimizers',
    version='0.0.1',

    description='Numba accelerated PyTorch Optimizers',

    # The project's main homepage.
    url='https://github.com/madsbk/pytorch-optimizers',

    # Author details
    author='Mads R. B. Kristensen',
    author_email='madsbk@gmail.com',

    # Choose your license
    license='Apache 2.0',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache License 2.0',
        'Programming Language :: Python :: 3',
    ],

    # What does your project relate to?
    keywords='PyTorch',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    install_requires=[
          'torch',
    ],
)

