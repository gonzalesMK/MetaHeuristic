# -*- coding: utf-8 -*-
from __future__ import print_function
from timeit import time

import numpy as np

from deap import base
from deap import tools


from .base import _BaseMetaHeuristic
from .base import BaseMask
from .base import *

class BRKGA2(_BaseMetaHeuristic):
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

    def __init__(self, estimator=None,
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

        self.size_pop = size_pop

        self.cxUniform_indpb = cxUniform_indpb
        self.elite_size = elite_size
        self.mutant_size = mutant_size
        
        random.seed(self.random_state)
     

    def _setup(self):

        if(self.elite_size + self.mutant_size > self.size_pop):
            raise ValueError(" Elite size({}) + Mutant_size({}) is bigger than population"
                             " size({})\n The algorithm may not work properly".format(
                                 self.elite_size, self.mutant_size, self.size_pop))
        self._n_cross_over = self.size_pop - (self.elite_size + self.mutant_size)
        self._non_elite_size = self.size_pop - self.elite_size

        super()._setup()
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        self._toolbox.register("mate", tools.cxUniform,
                              indpb=self.cxUniform_indpb)
        self._toolbox.register("select", tools.selTournament, tournsize=3)

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
            # Generate Population
            pop = self._toolbox.population(self.size_pop)
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()

            # Evaluate the entire population
            fitnesses = self._toolbox.map(self._toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            del fit, ind

            pareto_front.update(pop)
            hof.update(pop)
            for g in range(self.number_gen):

                ordered = tools.selNSGA2(pop, self.size_pop)  # Ordering
                # Partitioning
                elite = ordered[0:self.elite_size]
                non_elite = ordered[self.elite_size:self._non_elite_size+self.elite_size]

                # Cross_over between Elite and Non Elite
                father_ind = np.random.randint(
                    0, self.elite_size, self._n_cross_over)
                mother_ind = np.random.permutation(np.arange(0, self._non_elite_size))[
                    0:self._n_cross_over]

                child1 = self._toolbox.clone([elite[ind] for ind in father_ind])
                child2 = [non_elite[ind] for ind in mother_ind]

                for ind in range(0, len(child1)):
                    child1[ind], child2[ind] = self._toolbox.mate(
                        child1[ind], child2[ind])

                for ind1 in child1:
                    del ind1.fitness.values

                # Evaluate the individuals with an invalid fitness ( new individuals)
                fitnesses = self._toolbox.map(self._toolbox.evaluate, child1)
                for ind, fit in zip(child1, fitnesses):
                    ind.fitness.values = fit

                # The botton is replaced by mutant individuals
                mutant = self._toolbox.population(self.mutant_size)
                fitnesses = self._toolbox.map(self._toolbox.evaluate, mutant)
                for ind, fit in zip(mutant, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                pop[:] = elite + child1 + mutant

                # Log Statistics
                hof.update(pop)
                pareto_front.update(pop)
                if self.make_logbook:
                    self.logbook[i].record(gen=g,
                                           best_fit=hof[0].fitness.values[0],
                                           **self.stats.compile(pop))
                    self._make_generation_log(hof, pareto_front)

                if self.verbose:
                    self._print(g, i, initial_time, time.clock())

            self._make_repetition_log(hof, pareto_front)

        self._estimator.fit(X=self.transform(X), y=y)

        return self
