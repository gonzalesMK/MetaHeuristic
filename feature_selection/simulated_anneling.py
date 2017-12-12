from __future__ import print_function
import random
from timeit import time

import numpy as np

from deap import base
from deap import tools

import copy

from .base import _BaseMetaHeuristic
from .base import BaseMask
from sklearn.utils import check_random_state

class SimulatedAnneling(_BaseMetaHeuristic):
    """Implementation of a Simulated Anneling Algorithm for Feature Selection as
    stated in the book : Fred W. Glover - Handbook of Metaheuristics.

    the decay of the temperature is given by temp_init/number_gen

    Parameters
    ----------
    classifier : sklearn classifier , (default=SVM)
            Any classifier that adheres to the scikit-learn API

    mutation_prob : float in [0,1], (default=0.05)
            Is the the probability for each value in the solution to be mutated
            when searching for some neighbor solution.

    number_gen : positive integer, (default=10)
            Number of generations

    initial_temp : positive integer, (default=10)
            The initial temperature

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

    size_pop: None
            It is needed to
    """
    def __init__(self, classifier=None, mutation_prob=0.05, initial_temp=10,
                 repetition_schedule=10, number_gen=10, repeat=1, verbose=0,
                 parallel=False, make_logbook=False, random_state=None,
                 cv_metric_fuction=None, features_metric_function=None,
                 **arg):

        super(SimulatedAnneling, self).__init__(
                name = "SimulatedAnneling",
                classifier=classifier,
                number_gen=number_gen,
                verbose=verbose,
                repeat=repeat,
                parallel=parallel,
                make_logbook=make_logbook,
                random_state=random_state,
                cv_metric_fuction=cv_metric_fuction,
                features_metric_function=features_metric_function)

        self.mutation_prob = mutation_prob
        self.initial_temp = initial_temp
        self.repetition_schedule = repetition_schedule

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              BaseMask, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate, X= None, y=None)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.mutation_prob)
        if parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)

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
            solution = self.toolbox.individual()
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()
            # Evaluate the solution
            solution.fitness.values = self.toolbox.evaluate(solution)

            for temp in np.arange(self.initial_temp, 0,
                                  - self.initial_temp/self.number_gen):

                for _ in range(self.repetition_schedule):

                    prev_solution = copy.deepcopy(solution)
                    self.toolbox.mutate(solution)
                    solution.fitness.values = self.toolbox.evaluate(solution)

                    if prev_solution.fitness > solution.fitness:
                        solution = self._metropolis_criterion(
                                solution, prev_solution, temp)

                    # Log statistic
                    hof.update([solution])
                    pareto_front.update([solution])
                if self.make_logbook:
                    self.logbook[i].record(gen=temp,
                                best_fit=hof[0].fitness.values[0],
                                **self.stats.compile([solution]))
                    self._make_generation( hof, pareto_front)
                if self.verbose:
                    #if (int)(temp * self.number_gen/ self.initial_temp) % self.verbose == 0:
                    print("Repetition:", i+1, "Temperature: ", temp ,
                          "Elapsed time: ", time.clock() - initial_time,
                          end="\r")

            self._make_repetition(hof, pareto_front)

        self.estimator.fit(X= self.transform(X), y=y)

        return self

    @staticmethod
    def _metropolis_criterion(solution, prev_solution, temp):
        prob = np.exp(
    (sum(solution.fitness.wvalues) - sum(prev_solution.fitness.wvalues))/temp )

        if random.random() < prob:
            return solution
        else:
            return prev_solution

    def set_params(self, **params):
        super(SimulatedAnneling, self).set_params(**params)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.mutation_prob)

        if self.parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)

        return self