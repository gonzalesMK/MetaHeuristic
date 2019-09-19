from __future__ import print_function
import random
from timeit import time

from deap import base
from deap import tools

from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *


class GeneticAlgorithm(_BaseMetaHeuristic):
    """Implementation of a Genetic Algorithm for Feature Selection

    Parameters
    ----------
    classifier : sklearn classifier , (default=SVM)
            Any classifier that adheres to the scikit-learn API

    cross_over_prob :  float in [0,1], (default=0.5)
            Probability of happening a cross-over in a individual (chromosome)

    individual_mutation_probability : float in [0,1], (default=0.05)
            Probability of happening mutation in a individual ( chromosome )

    gene_mutation_prob : float in [0,1], (default=0.05)
            For each gene in the individual (chromosome) chosen for mutation,
            is the probability of it being mutate

    number_gen : positive integer, (default=10)
            Number of generations

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

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None, cross_over_prob=0.5, cxUniform_indpb=0.9,
                 individual_mut_prob=0.05, gene_mutation_prob=0.05,
                 number_gen=10, size_pop=40, verbose=0, repeat=1,
                 make_logbook=False, random_state=None, parallel=False,
            
                 cv_metric_function=None, features_metric_function=None,
                 print_fnc=None, name="GeneticAlgorithm"):

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

        self.individual_mut_prob = individual_mut_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.cross_over_prob = cross_over_prob
        self.size_pop = size_pop
        self.cxUniform_indpb = cxUniform_indpb
        self.parallel = parallel

    def _setup(self, X, y, normalize):
        X, y = super()._setup(X,y,normalize)

        self._toolbox.register("attribute", self._gen_in)
        
        self._toolbox.register("individual", tools.initIterate,
                               BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                               list, self._toolbox.individual)
        self._toolbox.register("mate", tools.cxUniform,
                               indpb=self.cxUniform_indpb)
        self._toolbox.register("select", tools.selTournament, tournsize=3)

        self._toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                               indpb=self.gene_mutation_prob)

        return X, y

    def _do_generation(self, pop, hof, paretoFront):

        # Select the next generation individuals through tournament
        offspring = self._toolbox.select(pop, len(pop))
        offspring = list(map(self._toolbox.clone, offspring))

        # Apply crossover 
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < self.cross_over_prob:
                self._toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        # Apply Mutation
        for mutant in offspring:
            if random.random() < self.individual_mut_prob:
                self._toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the individuals with an invalid fitness ( new individuals)
        invalid_ind = [ ind for ind in offspring if not ind.fitness.valid]
        fitnesses = self._toolbox.map( self._toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # The population is entirely replaced by the offspring
        pop[:] = offspring

        # Log statistic
        hof.update(pop)
        paretoFront.update(pop)

        return pop, hof, paretoFront
