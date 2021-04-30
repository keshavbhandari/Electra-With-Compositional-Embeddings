# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 20:03:46 2021

@author: kesha
"""

from setuptools import setup

install_requires = [
    'tensorflow-gpu>=2.4.0',
    'numpy==1.19.2',
    'pandas==1.1.3',
    'glob2>=0.7',
    'scikit-learn>=0.23.2',
    're',
    'pickle',
    'os',
    'gc'
]

# tests_require = ['pytest>=4.0.2']

setup(name='src',
      version='0.0.1',
      description='installing packages for Electra with compositional embeddings using complementary partitions',
      author='Keshav Bhandari',
      author_email='keshavbhandari@gmail.com',
      install_requires=install_requires,
      packages=['src'])