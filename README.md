[![Coverage Status](https://coveralls.io/repos/github/gonzalesMK/MetaHeuristic/badge.svg?branch=master)](https://coveralls.io/github/gonzalesMK/MetaHeuristic?branch=master)
[![Build Status](https://travis-ci.org/gonzalesMK/MetaHeuristic.svg?branch=master)](https://travis-ci.org/gonzalesMK/MetaHeuristic)
# MetaHeuristic
A repository with some simple implementation of Meta Heuristics (ie: ga, pso) , using sklearn and other common libraries

Most of the code is based on:

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

