.. -*- mode: rst -*-

|Coverage|_ |Pypi|_

.. |Travis| image:: https://travis-ci.org/gonzalesMK/MetaHeuristic.svg?branch=master
.. _Travis: https://travis-ci.org/gonzalesMK/MetaHeuristic

.. |Coverage| image:: https://coveralls.io/repos/github/gonzalesMK/MetaHeuristic/badge.svg?branch=master
.. _Coverage: https://coveralls.io/github/gonzalesMK/MetaHeuristic?branch=master

.. |CircleCI| image:: https://circleci.com/gh/gonzalesMK/MetaHeuristic/tree/master.svg?style=shield&circle-token=:circle-token
.. _CircleCI: https://circleci.com/gh/gonzalesMK/MetaHeuristic/tree/master

.. |Pypi| image:: https://badge.fury.io/py/metaheuristic.svg
.. _Pypi: https://badge.fury.io/py/metaheuristic
MetaHeuristic
=============
A repository with Meta Heuristics (ie: ga, pso) implementations for feature selection.

Important Links
---------------

HTML Documentation - http://metaheuristic.readthedocs.io/en/latest/

Installation and Usage
----------------------
The package by itself comes with a single module and an estimator. Before
installing the module you will need 
- Numpy
- Scipy
- DEAP 
- Matplotli
- Scikit-learn

To install the module execute, ``python`` ::

  python setup.py install

or: 

  pip install metaheuristic
If the installation is successful, and `MetaHeuristic` is correctly installed,
you should be able to execute the following in Python:

  >>> from feature_selection import HarmonicSearch
  >>> estimator = HarmonicSearch()

DEAP
--------
DEAP is a novel evolutionary computation framework for rapid prototyping and testing of 
ideas. It seeks to make algorithms explicit and data structures transparent. It works in perfect harmony with parallelisation mechanism such as multiprocessing and [SCOOP](http://pyscoop.org).

See the [DEAP User's Guide](http://deap.readthedocs.org/) for DEAP documentation.

