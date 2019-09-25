from __future__ import print_function
import random
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *


from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state


class BinaryBlackHole(_BaseMetaHeuristic):
    """
    Binary Black Hole for Feature Selection.

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    number_gen : positive integer, (default=10)
            Number of generations to iterate the algorithm. The number of generations
            can also be limited by ``time_limit`` in the ``fit`` method

    size_pop : positive integer, (default=40)
            Number of stars in the population.

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process. This is important if one wants to 
            repeat the fitting process in the pipeline, given the chance of getting a poor 
            solution at first

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made. Check the implementation of ``_make_stats`` for more info

    parallel : boolean, (default=False)
            Set to True if you want to use python's multiprocessor library for evaluating each solution in parallel

    cv_metric_function : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    References
    ----------
    .. [1] Elnaz Pashaei and Nizamettin Aydin. 2017. Binary black hole algorithm for feature 
           selection and classification on biological data.Appl. Soft Comput. 56, C (July 2017), 
           94-106. DOI: https://doi.org/10.1016/j.asoc.2017.03.002

    """

    def __init__(self, 
                 estimator=None, 
                 number_gen=10, 
                 size_pop=40, 
                 repeat=1,
                 verbose=False, 
                 make_logbook=False, 
                 random_state=None, 
                 parallel=False,
                 cv_metric_function=None, 
                 features_metric_function=None):

        self.estimator = estimator
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.size_pop = size_pop
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_function
        self.features_metric_function = features_metric_function

        np.random.seed(self.random_state)

    def _do_generation(self, galaxy, hof, paretoFront):

        # Evaluate the entire population
        fitnesses = self._toolbox.map(self._toolbox.evaluate, galaxy)
        for ind, fit in zip(galaxy, fitnesses):
            ind.fitness.values = fit

        # Log statistic
        hof.update(galaxy)
        paretoFront.update(galaxy)

        # Update Global Information
        hof[0].radius = sum(hof[0].fitness.wvalues) / \
            sum([sum(i.fitness.wvalues) for i in galaxy])

        # Update particles
        for part in galaxy:
            self._toolbox.update(part, hof[0])

        return galaxy, hof, paretoFront

    @staticmethod
    def _dist(star, blackhole):
        return np.linalg.norm([blackhole[i] - star[i] for i in range(0, len(star))])

    def _updateStar(self, star, blackhole):
        star[:] = [1 if abs(np.tanh(star[x] + self._random_object.uniform(0, 1) *
                                    (blackhole[x] - star[x]))) > self._random_object.uniform(0, 1) else 0 for x in range(0, self.n_features_)]

        if self._dist(star, blackhole) < blackhole.radius:
            star[:] = self._toolbox.population(n=1)[0]

    def _setup(self, X, y, normalize):

        X, y = super()._setup(X, y, normalize)
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("star", tools.initIterate,
                               BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                               list, self._toolbox.star)
        self._toolbox.register("update", self._updateStar)

        return X, y
