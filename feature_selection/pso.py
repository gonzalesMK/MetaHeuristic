""" It is hard to predict the mutation rate, because it varies depending on the
length of the solution... I though about creating an inverse_sigmoid function to
adapt the coefficient of the sigmoid function to ensure that particles
(list of [-1,1]) will have less than 5 mutations when  within 0.1 from the bounds,
otherwise, no convergence is expected"""

from __future__ import print_function

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask
from .base import *

import operator
import math
import numpy as np
import random
from timeit import time


class PSOBaseMask(BaseMask):
    def __init__(self, mask, speed=None):
        super(PSOBaseMask, self).__init__(mask)
        self.best = None
        self.speed = speed if speed != None else mask.speed


class PSO(_BaseMetaHeuristic):
    """Implementation of a Genetic Algorithm for Feature Selection

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

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

    cv_metric_fuction : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, estimator=None, phi1=0.2, phi2=0.2,
                 number_gen=10, size_pop=40, verbose=0, repeat=1, slim=1,
                 make_logbook=False, random_state=None, parallel=False,
                 cv_metric_fuction=None, features_metric_function=None,
                 print_fnc=None, name="PSO"):

        self.name = name
        self.estimator = estimator
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_fuction
        self.features_metric_function = features_metric_function
        self.print_fnc = print_fnc

        self.size_pop=size_pop
        self.phi1=phi1
        self.phi2=phi2
        self.slim=slim
        self.parallel = parallel

    def _make_toolbox(self):

        super()._make_toolbox()

        self._toolbox.register("attribute", self._gen_in)
        self._toolbox.register("individual", tools.initIterate,
                              PSOBaseMask, self._toolbox.attribute)
        self._toolbox.register("population", tools.initRepeat,
                              list, self._toolbox.individual)
        self._toolbox.register("evaluate", self._evaluate, X = None, y = None)


    def _gen_in(self):
        ind=super(PSO, self)._gen_in()
        speed=[random.uniform(-self.slim, self.slim) for _ in range(len(ind))]

        return PSOBaseMask(ind, speed)

    def updateParticle(self, part, best, phi1, phi2):

        # Update Personal Best
        if not part.best or part.best.fitness < part.fitness:
            part.best=self._toolbox.clone(part)

        # Personal Influence
        u1=(random.uniform(0, phi1) for _ in range(len(part)))
        v_u1=list(map(operator.mul, u1, map(operator.sub, part.best, part)))

        # Global Influence
        u2=(random.uniform(0, phi2) for _ in range(len(part)))
        v_u2=list(map(operator.mul, u2, map(operator.sub, best, part)))

        # Computate Speed
        part.speed[:]=list(map(operator.add, part.speed,
                              map(operator.add, v_u1, v_u2)))

        # Verifiy speed bounds
        for i, speed in enumerate(part.speed):
            if speed < - self.slim:
                part.speed[i]=- self.slim
            elif speed > self.slim:
                part.speed[i]=self.slim

        # Update Position
        rand=[random.random() for _ in range(len(part))]
        sig=map(self.sigmoid, part.speed)
        dif=map(operator.add, sig, rand)
        part[:]=np.round([x*0.5 for x in dif])

        # Delete Fitness
        del part.fitness.values

    def fit(self, X = None, y = None, normalize = False, **arg):
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
        initial_time=time.clock()

        self.set_params(**arg)
        self._make_toolbox()

        X, y=self._set_dataset(X = X, y = y, normalize = normalize)

        if self.n_features_ > 4:
            self._sigmoid_coeff=self.inverse_sigmoid(1-2/self.n_features_)
        else:
            self._sigmoid_coeff=self.inverse_sigmoid(0.9)

        self._set_fit()
        for i in range(self.repeat):
            pop=self._toolbox.population(self.size_pop)
            hof=tools.HallOfFame(1)
            pareto_front=tools.ParetoFront()

            # Evaluate the entire population
            fitnesses=self._toolbox.map(self._toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values=fit

            hof.update(pop)
            pareto_front.update(pop)
            for g in range(self.number_gen):
                # Update Particles
                for part in pop:
                    self.updateParticle(part, hof[0], self.phi1, self.phi2)

                # Evaluate the entire population
                fitnesses=self._toolbox.map(self._toolbox.evaluate, pop)
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values=fit

                # Log statistic
                hof.update(pop)
                pareto_front.update(pop)
                if self.make_logbook:
                        self.logbook[i].record(gen = g,
                                               best_fit = hof[0].fitness.values[0],
                                               **self.stats.compile(pop))
                        self._make_generation_log(hof, pareto_front)

                if self.verbose:
                    self._print(g, i, initial_time, time.clock())

            self._make_repetition_log(hof, pareto_front)

        self._estimator.fit(X = self.transform(X), y = y)

        return self


    @staticmethod
    def inverse_sigmoid(x):
        return -np.log(1/x - 1)

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x*self._sigmoid_coeff))
