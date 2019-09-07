# -*- coding: utf-8 -*-
from __future__ import print_function
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask
from .base import *

import random


class SPEA2(_BaseMetaHeuristic):
    """Implementation of Strenght Pareto Front Envolutionary Algorithm 2

    https://pdfs.semanticscholar.org/6672/8d01f9ebd0446ab346a855a44d2b138fd82d.pdf

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    elite_size : positive integer, (default=10)
            Number of individuals in the Elite population

    mutant_size : positive integer, (default=10)
            Number of new individuals in each iteration

    number_gen : positive integer, (default=10)
            Number of generations

    cxUniform_indpb : float in [0,1], (default=0.2)
             A uniform crossover modify in place the two sequence individuals.
             Inherits from the allele of the elite chromossome with indpb.

    size_pop : positive integer, (default=40)
            Number of individuals (choromosome ) in the population

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

    features_metric_function : { "log", "poly" }
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None,
                 archive_size=1, cxUniform_indpb=0.2,
                 number_gen=10, size_pop=3, verbose=0, repeat=1,
                 individual_mut_prob=0.5, gene_mutation_prob=0.01,
                 make_logbook=False, random_state=None, parallel=False,
                 cv_metric_function=None, features_metric_function=None,
                 print_fnc=None, name="SPEA2"):

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
        self.archive_size = archive_size
        self.cxUniform_indpb = cxUniform_indpb
        self.individual_mut_prob = individual_mut_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.parallel = parallel

        random.seed(self.random_state)

    def _setup(self):
        super()._setup()
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        self._toolbox.register("mate", tools.cxUniform,
                              indpb=self.cxUniform_indpb)
        self._toolbox.register("select", tools.selTournament, tournsize=2)
        self._toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.gene_mutation_prob)

    def _do_generation(self, archive, hof, paretoFront):

        # Mating Selection
        pop = self._toolbox.select(archive, self.size_pop)

        # Clone the selected individuals
        pop = list(map(self._toolbox.clone, pop))

        # Apply variation
        pop = self._variation(pop)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]
        fitnesses = self._toolbox.map(
            self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Environmental Selection
        archive = tools.selSPEA2(archive + pop, self.archive_size)

        # Log Statistics
        hof.update(archive)
        paretoFront.update(archive)

        return archive, hof, paretoFront

    def _variation(self, offspring):
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            self._toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

        for mutant in offspring:
            if random.random() < self.individual_mut_prob:
                self._toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

