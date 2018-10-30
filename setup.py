"""Config for PyPI."""

from setuptools import find_packages
from setuptools import setup

setup(
    author='John Haiducek',
    author_email='jhaiduce@umich.edu',
    version='0.0.1',
    zip_safe=True,
    packages=find_packages(),
    install_requires=[
          'scipy',
      ],
    name='swmf_validation'
)
