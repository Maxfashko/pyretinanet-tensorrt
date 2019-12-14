#!/usr/bin/env python3

from setuptools import setup, find_packages
from os.path import join, dirname

setup(name='pyretinanet',
      version='0.1',
      packages=find_packages(),
      long_description=open(join(dirname(__file__), 'README.md')).read(),
      install_requires=['pika==0.12.0',]
      )
