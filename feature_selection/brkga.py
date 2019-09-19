# -*- coding: utf-8 -*-
from __future__ import print_function
from timeit import time

import numpy as np

from deap import base
from deap import tools


from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *

class BRKGA(_BaseMetaHeuristic):
    """Implementation of a Biased Random Key Genetic Algorithm as the papers:

    Biased random-key genetic algorithms for combinatorial optimization

    Introdução aos algoritmos genéticos de chaves aleatórias viciadas

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

    features_metric_function :
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None, sorting_method='simple',
                 elite_size=1, mutant_size=1, cxUniform_indpb=0.2,
                 number_gen=10, size_pop=3, verbose=0, repeat=1,
                 make_logbook=False, random_state=None, parallel=False,
                 cv_metric_function=None, features_metric_function=None,
                 print_fnc=None, name="BRKGA2"):

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
        self.sorting_method = sorting_method
        self.size_pop = size_pop

        self.cxUniform_indpb = cxUniform_indpb
        self.elite_size = elite_size
        self.mutant_size = mutant_size
        
        random.seed(self.random_state)
     

    def _setup(self, X, y, normalize):

        if(self.elite_size + self.mutant_size > self.size_pop):
            raise ValueError(" Elite size({}) + Mutant_size({}) is bigger than population"
                             " size({})\n The algorithm may not work properly".format(
                                 self.elite_size, self.mutant_size, self.size_pop))
        self._n_cross_over = self.size_pop - (self.elite_size + self.mutant_size)
        self._non_elite_size = self.size_pop - self.elite_size

        X, y= super()._setup(X,y,normalize)
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        self._toolbox.register("mate", tools.cxUniform,
                              indpb=self.cxUniform_indpb)
        self._toolbox.register("select", tools.selTournament, tournsize=3)
        
        if(self.sorting_method == 'simple'):
            self._toolbox.register( "sort", sorted, key=lambda ind: ind.fitness.values[0], reverse=True)
        elif(self.sorting_method == 'NSGA2'):
            self._toolbox.register( "sort", tools.selNSGA2, k=self.size_pop)
        elif(self.sorting_method == 'NSGA3'):
            self._toolbox.register( "sort", tools.selNSGA3, k=self.size_pop)
        else:
            raise ValueError("The {} sorting method is not valid".format(self.sorting_method))

        return X, y

    def _do_generation(self, pop, hof, paretoFront):

        # Ordering
        ordered = self._toolbox.sort(pop)  
        
        # Partition Elite and Non Elite -> We can repeat elites index, but no repeating non-elite!
        father_indexes = np.random.randint( 0, self.elite_size, self._n_cross_over)
        mother_indexes = np.random.permutation(np.arange(self.elite_size, self.elite_size + self._non_elite_size))[0:self._n_cross_over]

        children = [self._toolbox.clone(ordered[ind]) for ind in father_indexes ]

        # Cross-Over
        for ind in range(0, len(children)):
            mother_index = mother_indexes[ind]
            self._toolbox.mate( children[ind], ordered[mother_index])
            del children[ind].fitness.values

        # Evaluate the individuals with an invalid fitness ( new individuals)
        fitnesses = self._toolbox.map(self._toolbox.evaluate, children)
        for ind, fit in zip(children, fitnesses): 
            ind.fitness.values = fit

        # The botton is replaced by mutant individuals
        mutant = self._toolbox.population(self.mutant_size)
        fitnesses = self._toolbox.map(self._toolbox.evaluate, mutant)
        
        for ind, fit in zip(mutant, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = ordered[0:self.elite_size] + children + mutant

        # Log Statistics
        hof.update(pop)
        paretoFront.update(pop)

        return pop, hof, paretoFront
