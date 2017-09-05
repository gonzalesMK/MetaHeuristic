.. -*- mode: rst -*-

|Travis|_ |Coverage|_ |CircleCI|_ 

.. |Travis| image:: https://travis-ci.org/gonzalesMK/MetaHeuristic.svg?branch=master
.. _Travis: https://travis-ci.org/gonzalesMK/MetaHeuristic

.. |Coverage| image:: https://coveralls.io/repos/scikit-learn-contrib/project-template/badge.svg?branch=master&service=github
.. _Coverage: https://coveralls.io/r/scikit-learn-contrib/project-template

.. |CircleCI| image:: https://circleci.com/gh/gonzalesMK/MetaHeuristic/tree/master.svg?style=svg
.. _CircleCI: https://circleci.com/gh/gonzalesMK/MetaHeuristic/tree/master


# MetaHeuristic
A repository with some simple implementation of Meta Heuristics (ie: ga, pso) , using sklearn and other common libraries

## Important Links
HTML Documentation - http://metaheuristic.readthedocs.io/en/latest/

## Installation and Usage
The package by itself comes with a single module and an estimator. Before
installing the module you will need `numpy`,`scipy`,`DEAP`,`Matplotlib`, `Scikit-learn`.

To install the module execute:
```shell
$ python setup.py install
```
If the installation is successful, and `MetaHeuristic` is correctly installed,
you should be able to execute the following in Python:
```python
>>> from feature_selection import HarmonicSearch
>>> estimator = HarmonicSearch()
```

## DEAP
DEAP is a novel evolutionary computation framework for rapid prototyping and testing of 
ideas. It seeks to make algorithms explicit and data structures transparent. It works in perfect harmony with parallelisation mechanism such as multiprocessing and [SCOOP](http://pyscoop.org).

See the [DEAP User's Guide](http://deap.readthedocs.org/) for DEAP documentation.

