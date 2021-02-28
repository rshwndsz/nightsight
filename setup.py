#!/usr/bin/env python
"""NightSight

TODO
Here be dragons
"""
import os
import sys
import subprocess
from setuptools import setup, find_packages

DOCLINES = __doc__.split("\n")
MAJOR = 0
MINOR = 1
MICRO = 0
VERSION = '%d.%d.%d' % (MAJOR, MINOR, MICRO)

if sys.version_info[0:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

setup(
    name='nightsight',
    version=VERSION,
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    author="Russel",
    url="https://github.com/rshwndsz/nightsight",
    packages=find_packages(include=['nightsight', 'nightsight.*']),
    install_requires=[
        'torch>=1.5',
        'torchvision',
        'pytorch_lightning',
        'numpy',
        'matplotlib',
        'pillow',
    ],
    extras_require={
        # Use pip install -e .[training]
        'training': ['albumentations', 'matplotlib', 'tqdm'],
    }
    tests_require=['pytest'],
)
