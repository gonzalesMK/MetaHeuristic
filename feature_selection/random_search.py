from __future__ import print_function
import random
from itertools import compress
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask
from .base import *

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state


class RandomSearch(_BaseMetaHeuristic):
    """Implementation of a Random Search Algorithm for Feature Selection.
    It is useful as the worst case

    Parameters
    ----------
    number_gen : positive integer, (default=5)
            Number of generations

    size_pop : positive integer, (default=40)
            Size of random samples in each iteration

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process

    parallel : boolean, (default=False)
            Set to True if you want to use multiprocessors

    make_logbook: boolean, (default=False)
            If True, a logbook from DEAP will be made

    cv_metric_function : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None, number_gen=5, size_pop=40, verbose=0,
                 repeat=1,
                 parallel=False, make_logbook=False, random_state=None,
                 cv_metric_function=None, features_metric_function=None,
                 print_fnc=None, name="RandomSearch"):
        
        self.name = name
        self.estimator = estimator
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_function
        self.features_metric_function = features_metric_function
        self.print_fnc = print_fnc

        self.size_pop = size_pop
        self.parallel = parallel

    def _setup(self, X, y, normalize):

        X, y= super()._setup(X,y,normalize)

        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        
        return X, y

        
    def _do_generation(self, pop, hof, paretoFront):
        pop = self._toolbox.population(n=self.size_pop)

        # Evaluate the entire population
        fitnesses = self._toolbox.map(self._toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # Log statistic
        hof.update(pop)
        paretoFront.update(pop)

        return pop, hof, paretoFront
