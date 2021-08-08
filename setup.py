from distutils.core import setup

from setuptools import find_packages

# requires = ["gym", "python-dateutil", "pytorch", "plaidml", "gtimer"]

setup(
    name="plaidrl",
    version="0.0.1",
    packages=find_packages(),
    license="MIT License",
    long_description=open("README.md").read(),
)
