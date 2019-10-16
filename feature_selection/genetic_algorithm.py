from __future__ import print_function
import random
from timeit import time

from deap import base
from deap import tools

from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *

def mutBinaryUniform(individual, indpb, prob):
    """Mutate an individual by replacing attributes, with probability *indpb*,
    by 1  with ``prob`` chance, otherwise 0.
    :param individual: :term:`Sequence <sequence>` individual to be mutated.
    :param low: The lower bound or a :term:`python:sequence` of
                of lower bounds of the range from wich to draw the new
                integer.
    :param up: The upper bound or a :term:`python:sequence` of
               of upper bounds of the range from wich to draw the new
               integer.
    :param indpb: Independent probability for each attribute to be mutated.
    :returns: A tuple of one individual.
    """
    size = len(individual)
    if not isinstance(low, Sequence):
        low = repeat(low, size)
    elif len(low) < size:
        raise IndexError("low must be at least the size of individual: %d < %d" % (len(low), size))
    if not isinstance(up, Sequence):
        up = repeat(up, size)
    elif len(up) < size:
        raise IndexError("up must be at least the size of individual: %d < %d" % (len(up), size))

    for i, xl, xu in zip(xrange(size), low, up):
        if random.random() < indpb:
            individual[i] = int(random.uniform() < prob)

    return individual,


class GeneticAlgorithm(_BaseMetaHeuristic):
    """Implementation of a Genetic Algorithm for Feature Selection

    Parameters
    ----------
    classifier : sklearn classifier , (default=SVM)
            Any classifier that adheres to the scikit-learn API

    number_gen : positive integer, (default=10)
            Number of generations

    size_pop : positive integer, (default=40)
            Number of individuals (choromosome ) in the population

    cross_over_type : one of {'uniform','onePoint', 'twoPoint'}, (default='uniform')

    cross_over_prob :  float in [0,1], (default=0.5)
            Probability of happening cross-over in a individual (chromosome)

    cxUniform_indpb : float in [0,1], (default=0.9)
            If ``cross_over_type_`` is 'uniform', this set the intesity of the gene exchange between the solutions. Otherwise, has no use.

    individual_mutation_probability : float in [0,1], (default=0.05)
            Probability of happening mutation in a individual ( chromosome )

    gene_mutation_prob : float in [0,1], (default=0.05)
            For each gene in the individual (chromosome) chosen for mutation,
            it is the probability mutation

    mutation_skewed_prob: float in [0, 1], (default=0.5)
            To increase the pressure to reduce the number of features, one can skew the chances of getting 0 (or 1) in the mutation procedure.
            
            The chances of a mutated gene to become a 1 is ``mutation_skewed_prob``. 0.5 is the uniform distribuition between {0,1}

    selection_method : one of {'tournament','roulette', 'NSGA2', 'SPEA2', 'best'}
            This is the selection method to create the offspring

            Check the explanation for each option here: https://deap.readthedocs.io/en/master/api/tools.html

    tournament_size: positive integer, (default=3)
            If ``selection_method`` is equal to 'tournament , then one can set the size of the tournament. Otherwise has no use.

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
    """

    def __init__(self,
                 estimator=None,
                 number_gen=10, 
                 size_pop=40,
                 cross_over_type='uniform',
                 cross_over_prob=0.5,
                 cxUniform_indpb=0.5,
                 individual_mut_prob=0.05, 
                 gene_mutation_prob=0.05,
                 selection_method='tournament',
                 tournament_size=3,
                 verbose=0,
                 repeat=1,
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

        self.individual_mut_prob = individual_mut_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.cross_over_type = cross_over_type
        self.cross_over_prob = cross_over_prob
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.size_pop = size_pop
        self.cxUniform_indpb = cxUniform_indpb
        self.mutation_skewed_prob = mutation_skewed_prob
        self.parallel = parallel

    def _setup(self, X, y, normalize):
        X, y = super()._setup(X,y,normalize)

        self._toolbox.register("attribute", self._gen_in)
        
        self._toolbox.register("individual", tools.initIterate,
                               BaseMask, self._toolbox.attribute)
        
        self._toolbox.register("population", tools.initRepeat,
                               list, self._toolbox.individual)
        
        if self.cross_over_type == 'uniform':
                self._toolbox.register("mate", tools.cxUniform,
                               indpb=self.cxUniform_indpb)
        elif self.cross_over_type == 'onePoint':
                self._toolbox.register("mate", tools.cxOnePoint)
        elif self.cross_over_type == 'twoPoint':
                self._toolbox.register("mate", tools.cxTwoPoint)
        else:
                raise ValueError("Unkown cross_over_type: {}".format(self.cross_over_type))
        
        if self.selection_method == "tournament":
            self._toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)
        elif self.selection_method == 'roullete':
            self._toolbox.register("select", tools.selRoulette)
        elif self.selection_method == 'NSGA2':
            self._toolbox.register("select", tools.selNSGA2)
        elif self.selection_method == 'SPEA2':
            self._toolbox.register("select", tools.selSPEA2)
        elif self.selection_method == 'best':
            self._toolbox.register("select", tools.selBest)
        else:
            raise ValueError("Unkown selection_method: {}".format(self.selection_method))

        self._toolbox.register("mutate", mutBinaryUniform,prob=self.mutation_skewed_prob,
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
