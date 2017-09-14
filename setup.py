from __future__ import print_function
import sys
from setuptools import setup, find_packages

with open('requirements.txt') as f:
    INSTALL_REQUIRES = [l.strip() for l in f.readlines() if l]


try:
    import numpy
except ImportError:
    print('numpy is required during installation')
    sys.exit(1)

try:
    import scipy
except ImportError:
    print('scipy is required during installation')
    sys.exit(1)


setup(name='metaheuristic',
      version='1.3',
      description='A package with some metaheuristics to feature selection',
      long_description=" Take a look into https://github.com/gonzalesMK/MetaHeuristic",
      author='Juliano D. Negri',
      packages=find_packages(),
      install_requires=INSTALL_REQUIRES,
      author_email='julianonegri@gmail.com',
      url='https://github.com/gonzalesMK/MetaHeuristic',
      classifiers=[
              'Programming Language :: Python :: 3.6',
              'Development Status :: 3 - Alpha',
              'Intended Audience :: Science/Research',
              'Topic :: Scientific/Engineering :: Information Analysis',
              'Topic :: Software Development :: Libraries'
              ]
      )
