from __future__ import print_function
import random
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


class BinaryBlackHole(_BaseMetaHeuristic):
    """Implementation of Binary Black Hole for Feature Selection

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    number_gen : positive integer, (default=10)
            Number of generations

    size_pop : positive integer, (default=40)
            Number of individuals in the population

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made

    parallel : boolean, (default=False)
            Set to True if you want to use multiprocessors

    cv_metric_function : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None, number_gen=10, size_pop=40, verbose=False,
                 repeat=1, make_logbook=False, random_state=None, parallel=False,
                 cv_metric_function=None, features_metric_function=None,
                 print_fnc=None, name="BinaryBlackHole"):

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

        random.seed(self.random_state)

    def fit(self, X=None, y=None, normalize=False, **arg):
        """ Fit method

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
                The input samples

        y : array of shape [n_samples, 1]
                The input of labels

        normalize : boolean, (default=False)
                If true, StandardScaler will be applied to X

        **arg : parameters
                Set parameters
        """
        initial_time = time.clock()

        self._setup()

        self.set_params(**arg)

        X, y = self._set_dataset(X=X, y=y, normalize=normalize)
        

        for i in range(self.repeat):
            galaxy = self._toolbox.galaxy(n=self.size_pop)
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()

            for g in range(self.number_gen):

                # Evaluate the entire population
                fitnesses = self._toolbox.map(self._toolbox.evaluate, galaxy)
                for ind, fit in zip(galaxy, fitnesses):
                    ind.fitness.values = fit

                # Update Global Information
                hof.update(galaxy)
                pareto_front.update(galaxy)
                hof[0].radius = sum(hof[0].fitness.wvalues) / \
                    sum([sum(i.fitness.wvalues) for i in galaxy])

                # Update particles
                for part in galaxy:
                    self._toolbox.update(part, hof[0])

                # Log statistic
                hof.update(galaxy)
                pareto_front.update(galaxy)
                if self.make_logbook:
                    record = self.stats.compile(galaxy)
                    print("Record: {}".format(record))
                    self.logbook[i].record(gen=g,
                                           best_fit=hof[0].fitness.values[0],
                                           **record)
                    self._make_generation_log(hof, pareto_front)

                if self.verbose:
                    self._print(g, i, initial_time, time.clock())

            self._make_repetition_log(hof, pareto_front)
            # print(len(self.gen_hof_))

        self._estimator.fit(X=self.transform(X), y=y)

        return self

    @staticmethod
    def _dist(star, blackhole):
        return np.linalg.norm([blackhole[i] - star[i] for i in range(0, len(star))])

    def _updateStar(self, star, blackhole):
        if self._dist(star, blackhole) < blackhole.radius:
            star[:] = self._toolbox.galaxy(n=1)[0]
        else:
            star[:] = [1 if abs(np.tanh(star[x] + self._random_object.uniform(0, 1) *
                                        (blackhole[x] - star[x]))) > self._random_object.uniform(0, 1) else 0 for x in range(0, self.n_features_)]


    def _setup(self):

        super(BinaryBlackHole, self)._setup()
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("star", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("galaxy", tools.initRepeat,
                              list, self._toolbox.star)
        self._toolbox.register("update", self._updateStar)
        self._toolbox.register("evaluate", self._evaluate)
