from __future__ import print_function
import random
from timeit import time

import numpy as np

from deap import base
from deap import tools

import copy

from .meta_base import _BaseMetaHeuristic
from .meta_base import BaseMask
from .meta_base import *
from sklearn.utils import check_random_state


class SimulatedAnneling(_BaseMetaHeuristic):
    """
    
    Implementation of Simulated Anneling Algorithm for Feature Selection

    The temperature range is [initial_temp, 0]. So, the decay of the temperature 
    each generation is given by ``temp_init/number_gen``.
     
    The total number of evaluated solutions are: ``number_gen * repetition_schedule`` 

    INCLUDE A GRAPH W/ THE METROPOLIS PROBABILITY:

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    number_gen : positive integer, (default=10)
            Number of generations

    mutation_prob : float in [0,1], (default=0.05)
            Is the the probability for each value in the solution to be mutated
            when searching for some neighbor solution.

    initial_temp : positive integer, (default=10)
            The initial temperature

    repetition_schedule: positive integer, (default=10)
            The number of times to repeat the searching process before changing temperature 

    skip: positive integer, (default=10)
            This parameter is important when ``make_logbook`` is True. Usually, a lot
            of generations are needed by HS to achieve convergence, given that 
            one solution is evaluated at a time. Consequently, the amount of data 
            saved by the ``logbook`` can be bigger than necessary. So, one can
            choose to log data only every ``skip``iteration

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process

    parallel : boolean, (default=False)
            Set to True if you want to use multiprocessors

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made

    cv_metric_function : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter


    References  
    ----------
    .. [1] Fred W. Glover - Handbook of Metaheuristics. 2003.
    """

    def __init__(self,
                 estimator=None, 
                 number_gen=10, 
                 mutation_prob=0.05, 
                 initial_temp=1,
                 repetition_schedule=10, 
                 skip=10,
                 verbose=0,
                 repeat=1, 
                 parallel=False, 
                 make_logbook=False, 
                 random_state=None,
                 cv_metric_function=None):

        self.estimator = estimator
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_function

        self.mutation_prob = mutation_prob
        self.initial_temp = initial_temp
        self.repetition_schedule = repetition_schedule
        self.skip = skip
        self.parallel = parallel
        
    def _setup(self, X, y, normalize):

        X, y = super()._setup(X,y,normalize)
        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              BaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        
        self._toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.mutation_prob)

        return X, y
        
    def fit(self, X, y, time_limit = None, normalize=False, **arg):
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
        
        X, y = self._setup(X, y, normalize)
        self._initial_time = time.clock()
        self.set_params(**arg)
        

        for i in range(self.repeat):
            solution = self._toolbox.individual()
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()

            # Evaluate the solution
            solution.fitness.values = self._toolbox.evaluate(solution)

            g = 0
            for temp in np.arange(self.initial_temp, 0,
                                  - self.initial_temp/self.number_gen):

                for _ in range(self.repetition_schedule):

                    prev_solution = copy.deepcopy(solution)
                    self._toolbox.mutate(solution)
                    solution.fitness.values = self._toolbox.evaluate(solution)

                    if prev_solution.fitness > solution.fitness:
                        solution = self._metropolis_criterion(
                            solution, prev_solution, temp)

                    # Log statistic
                    hof.update([solution])
                    pareto_front.update([solution])

                    g = g+1

                    if self.skip == 0 or g % self.skip == 0:
                        if self.make_logbook:
                            self._make_generation_log(g, i, [solution], hof, pareto_front)

                        if self.verbose and not self.make_logbook:
                            self._print(temp, _, i, self._initial_time, time.clock())
                
                    if time_limit is not None and time.clock() - self._initial_time > time_limit:
                        break

            self._make_repetition_log(hof, pareto_front)

        self._estimator.fit(X=self.transform(X), y=y)

        return self

    def _print(self, temp, schedule, rep, initial_time, final_time):
        self._toolbox.print("""Repetition: {:d}\tTemperature: {:.4f}/{:.4f}\tSchedule: {:d}/{:d}\tElapsed time:{:.4f} \r""".format(
            rep+1, temp, self.initial_temp, schedule, self.repetition_schedule,
            final_time - initial_time))

    @staticmethod
    def _metropolis_criterion(solution, prev_solution, temp):
        prob = np.exp(100 * (sum(solution.fitness.wvalues) -
                       sum(prev_solution.fitness.wvalues))/(temp))

        if random.random() < prob:
            return solution
        else:
            return prev_solution


def _test_metropolis( now, before, temp):
    return np.exp( (now - before)/ temp * 100)
  

