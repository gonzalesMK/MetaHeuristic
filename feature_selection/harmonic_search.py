from __future__ import print_function
import random
from itertools import compress
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask

from sklearn.svm import  SVC
from sklearn.base import clone
from sklearn.utils import check_random_state

from multiprocessing import Pool

class HarmonicSearch(_BaseMetaHeuristic):
    """Implementation of a Harmonic Search Algorithm for Feature Selection

    Parameters
    ----------
    HMCR : float in [0,1], (default=0.95)
            Is the Harmonic Memory Considering Rate

    indpb : float in [0,1], (default=0.05)
            Is the mutation rate of each new harmony

    pitch : float in [0,1], (default=0.05)
            Is the Pitch Adjustament factor

    number_gen : positive integer, (default=100)
            Number of generations

    size_pop : positive integer, (default=50)
            Size of the Harmonic Memory

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process

    parallel : boolean, (default=False)
            Set to True if you want to use multiprocessors

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made

    cv_metric_fuction : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, classifier=None, HMCR=0.95, indpb=0.05, pitch=0.05,
                 number_gen=100, size_pop=50, verbose=0, repeat=1,
                 make_logbook=False, random_state=None, parallel = False,
                 cv_metric_fuction=None, features_metric_function=None):

        super(HarmonicSearch, self).__init__(
                name = "HarmonicSearch",
                classifier=classifier,
                number_gen=number_gen,
                verbose=verbose,
                repeat=repeat,
                parallel=parallel,
                make_logbook=make_logbook,
                random_state=random_state,
                cv_metric_fuction=cv_metric_fuction,
                features_metric_function=features_metric_function)

        self.HMCR = HMCR
        self.indpb = indpb
        self.pitch = pitch
        self.estimator = SVC(kernel='linear', verbose=False, max_iter=10000) if classifier is None else clone(classifier)
        self.size_pop = size_pop

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              BaseMask, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("get_worst", tools.selWorst, k=1)
        self.toolbox.register("evaluate", self._evaluate, X=None, y=None)

        if parallel:
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)

        self.toolbox.register("improvise", self._improvise, HMCR=self.HMCR)
        self.toolbox.register("mutate", tools.mutUniformInt,low=0, up=1,
                              indpb=self.indpb)
        self.toolbox.register("pitch_adjustament", tools.mutFlipBit,
                              indpb=self.pitch)

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

        self.set_params(**arg)
        X,y = self._set_dataset(X=X, y=y, normalize=normalize)

        self._set_fit()

        for i in range(self.repeat):
            harmony_mem = self.toolbox.population(n=self.size_pop)
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()

            # Evaluate the entire population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, harmony_mem)

            for ind, fit in zip(harmony_mem, fitnesses):
                ind.fitness.values = fit

            for g in range(self.number_gen):

                # Improvise a New Harmony
                new_harmony = self.toolbox.improvise(harmony_mem)
                new_harmony.fitness.values = self.toolbox.evaluate(new_harmony)

                # Select the Worst Harmony
                worst = self.toolbox.get_worst(harmony_mem)[0]

                # Check and Update Harmony Memory
                if worst.fitness < new_harmony.fitness:
                    worst[:] = new_harmony[:]
                    worst.fitness.values = new_harmony.fitness.values

                # Log statistic
                hof.update(harmony_mem)
                pareto_front.update(harmony_mem)
                if self.make_logbook:
                    self.logbook[i].record(gen=g,
                                           best_fit=hof[0].fitness.values[0],
                                           **self.stats.compile(harmony_mem))
                if self.verbose:
                    print("Repetition:", i+1 ,"Generation: ", g + 1, "/", self.number_gen,
                          "Elapsed time: ", time.clock() - initial_time, end="\r")

            self._make_repetition(hof,pareto_front)

        self.estimator.fit(X= self.transform(X), y=y)

        return self

    def _improvise(self, pop, HMCR):
        """ Function that improvise a new harmony"""
        # HMCR = Harmonic Memory Considering Rate
        # pylint: disable=E1101
        new_harmony = self.toolbox.individual()

        rand_list = self._random_object.randint(low=0, high=len(pop),
                                               size=len(new_harmony))

        for i in range(len(new_harmony)):
            new_harmony[i] = pop[rand_list[i]][i]
        self.toolbox.mutate(new_harmony)
        self.toolbox.pitch_adjustament(new_harmony)
        return new_harmony

    def set_params(self, **params):

        super(HarmonicSearch, self).set_params(**params)

        self.toolbox.register("improvise", self._improvise, HMCR=self.HMCR)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.indpb)
        self.toolbox.register("pitch_adjustament", tools.mutFlipBit,
                              indpb=self.pitch)
        if self.parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)

        return self
