# -*- coding: utf-8 -*-
from __future__ import print_function
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *

import random


class SPEA2(_BaseMetaHeuristic):
    """Implementation of Strenght Pareto Front Envolutionary Algorithm 2

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    archive : positive integer, (default=10)
            Number of individuals in the Elite population

    number_gen : positive integer, (default=10)
            Number of generations

    cxUniform_indpb : float in [0,1], (default=0.2)
             A uniform crossover modify in place the two sequence individuals.
             Inherits from the allele of the elite chromossome with indpb.

    individual_mut_prob : float in [0,1], (default=0.2)
             The chances of selecting a solution for mutation

    gene_mutation_prob : float in [0,1], (default=0.2)
             The chances of mutation of each gene ( intensity of the mutation itself )

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

    References
    ----------
    .. [1]  "Spea2: Improving the strength pareto evolutionary algorithm". ITZLER M. LAUMANNS. 
            Eidgenössische Technische Hochschule Zürich (ETH),Institut für Technische Informatik 
            und Kommunikationsnetze (TIK), 2001.  
          
    """

    def __init__(self, estimator=None,
                 archive_size=1, 
                 cxUniform_indpb=0.2,
                 number_gen=10, 
                 size_pop=3, 
                 verbose=0, 
                 repeat=1,
                 individual_mut_prob=0.5, 
                 gene_mutation_prob=0.01,
                 make_logbook=False, 
                 random_state=None, 
                 parallel=False,
                 cv_metric_function=None):

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

        np.random.seed(self.random_state)

    def _setup(self, X, y, normalize):
        X, y = super()._setup(X,y,normalize)
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

        return X, y
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

