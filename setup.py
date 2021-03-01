#!/usr/bin/env python
import os
import sys
import subprocess
from setuptools import setup, find_packages

if sys.version_info[0:2] < (3, 6):
    raise RuntimeError("Python version >= 3.6 required.")

setup(
    name='nightsight',
    version='0.1.0',
    description="Low light image enhancement on low power devices",
    long_description=open("README.md").read(),
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
        'pytest'
    ],
    extras_require={
        # Use pip install -e .[training]
        # Use pip install -e .\[training\] on zsh
        'training': ['albumentations', 'matplotlib', 'tqdm'],
    },
    tests_require=['pytest'],
)
