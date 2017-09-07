from __future__ import print_function
import random
from itertools import compress
from timeit import time

import numpy as np

from deap import base, creator
from deap import tools

from .base import _BaseMetaHeuristic
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state

   
class RandomSearch(_BaseMetaHeuristic):    
    """Implementation of a Random Search Algorithm for Feature Selection. 
    It is useful as the worst case
    
    Parameters
    ----------
    number_gen : positive integer, (default=5)
            Number of generations

    size_pop : positive integer, (default=40)
            Size of random samples in each iteration

    verbose : int, (default=0)
            Print information in every generation% verbose == 0

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process
            
    predict_with : one of { 'all' , 'best' }, (default='best')
        'all' - will predict the X with all masks and return the mean

        'best' - will predict the X using the mask with the best fitness

        obs: If you are going to make grid search in hyperparameters, use 'all'

    make_logbook: boolean, (default=False)
            If True, a logbook from DEAP will be made
    """

    def __init__(self, classifier=None, number_gen=5, size_pop=40,verbose=0, 
                 repeat=1, predict_with='best', make_logbook=False,
                 random_state=None):

        super(RandomSearch, self).__init__(
            classifier=classifier, number_gen=number_gen, size_pop=size_pop, 
            verbose=verbose, repeat=repeat, predict_with=predict_with, 
            make_logbook=make_logbook, random_state=random_state)

        self._name = "RandomSearch"
        creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                         self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate)
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
        self.set_params(**arg)
        initial_time = time.clock()
        
        if normalize:
            self._sc_X = StandardScaler()
            X = self._sc_X.fit_transform(X)
            
        self.normalize_ = normalize
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        self.X_ = X
        self.y_ = y

        self.n_features_ = X.shape[1]
        if self.predict_with == 'all':
            self.mask_ = []
            self.fitnesses_ = []
        random.seed(self.random_state)        
        self._random_object = check_random_state(self.random_state)
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)
        self.toolbox.register("map", map)

        if self.make_logbook:
            self.stats = tools.Statistics(self._get_accuracy)
            self.stats.register("avg", np.mean)
            self.stats.register("std", np.std)
            self.stats.register("min", np.min)
            self.stats.register("max", np.max)
            self.logbook = [tools.Logbook() for i in range(self.repeat)]
            for i in range(self.repeat):
                self.logbook[i].header = ["gen"] + self.stats.fields

        best = tools.HallOfFame(1)
        for i in range(self.repeat):
            hof = tools.HallOfFame(1)

            for g in range(self.number_gen):
                pop = self.toolbox.population(n=self.size_pop) 
        
                # Evaluate the entire population
                fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
                for ind, fit in zip(pop, fitnesses):
                    ind.fitness.values = fit
        
                # Log statistic
                hof.update(pop)
                if self.make_logbook:
                    self.logbook[i].record(gen=g,
                                           best_fit=hof[0].fitness.values[0],
                                           **self.stats.compile(pop))
                if self.verbose:
                    print("Repetition:", i+1 ,"Generation: ", g + 1, "/", self.number_gen,
                          "Elapsed time: ", time.clock() - initial_time, end="\r")

            best.update(hof)
            if self.predict_with == 'all':
                self.mask_.append(hof[0][:])
                self.fitnesses_.append(hof[0].fitness.values)

        if self.predict_with == 'all':
            self.mask_ = np.array(self.mask_)
        self.best_mask_ = np.asarray(best[0][:], dtype=bool)
        self.fitness_ = best[0].fitness.values

        features = list(compress(range(len(self.best_mask_)), self.best_mask_))
        train = np.reshape([X[:, i] for i in features], [len(features), len(X)]).T

        self.estimator.fit(X=train, y=y)

        return self

