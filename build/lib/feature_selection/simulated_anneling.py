import random
from itertools import compress
from datetime import datetime

import numpy as np

from deap import base, creator
from deap import tools

from .base import _BaseMetaHeuristic
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state
from sklearn.base import clone
from sklearn.svm import  SVC

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

    verbose : int, (default=0)
            Print information in every generation% verbose == 0

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process
    
    predict_with : one of { 'all' , 'best' }, (default='best')
        'all' - will predict the X with all masks and return the mean

        'best' - will predict the X using the mask with the best fitness

        obs: If you are going to make grid search in hyperparameters, use 'all'

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made
    """
    def __init__(self, classifier=None, mutation_prob=0.05, initial_temp=10,
                 repetition_schedule=10, number_gen=10, repeat=1, verbose=0,
                 predict_with='best', make_logbook=False, random_state=None):
    
        self._name = "SimulatedAnneling"
        self.estimator = SVC(kernel='linear', verbose=False, max_iter=10000) if classifier is None else clone(classifier)
        self.mutation_prob = mutation_prob
        self.initial_temp = initial_temp
        self.repetition_schedule = repetition_schedule
        self.number_gen = number_gen
        self.repeat = repeat
        self.verbose = verbose
        self.predict_with = predict_with
        self.make_logbook = make_logbook
        self.random_state = random_state
        self._random_object = check_random_state(self.random_state)
        random.seed(self.random_state)        
        
        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.Fitness)

        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("map", map)
        self.toolbox.register("evaluate", self._evaluate, X= None, y=None)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.mutation_prob)


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
        
        if normalize:
            self._sc_X = StandardScaler()
            X = self._sc_X.fit_transform(X)
            
        self.normalize_ = normalize
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        self.X_ = X
        self.y_ = y

        self.n_features_ = X.shape[1]
        self.mask_ = []
        self.fitnesses_ = []
        # pylint: disable=E1101
        random.seed(self.random_state)        
        self._random_object = check_random_state(self.random_state)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("map", map)
        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.mutation_prob)
        if self.make_logbook:
            self.stats = tools.Statistics(lambda ind: ind.fitness.wvalues[0])
            self.stats.register("avg", np.mean)
            self.stats.register("std", np.std)
            self.stats.register("min", np.min)
            self.stats.register("max", np.max)
            self.logbook = [tools.Logbook() for i in range(self.repeat)]
            for i in range(self.repeat):
                self.logbook[i].header = ["gen"] + self.stats.fields

        best = tools.HallOfFame(1)
        for i in range(self.repeat):
            solution = self.toolbox.individual()
            hof = tools.HallOfFame(1)
            # Evaluate the solution
            solution.fitness.values = self.toolbox.evaluate(solution)
            
            for temp in np.arange(self.initial_temp, 0, - self.initial_temp/self.number_gen):
                
                for m in range(self.repetition_schedule): 
                    
                    prev_solution = solution
                    self.toolbox.mutate(solution)
                    solution.fitness.values = self.toolbox.evaluate(solution)
                    
                    if prev_solution.fitness > solution.fitness:
                        solution = self._metropolis_criterion(solution, prev_solution, temp)
                    
                    # Log statistic
                    hof.update([solution])
                
                if self.make_logbook:
                    self.logbook[i].record(gen=temp,
                                best_fit=hof[0].fitness.values[0],
                                **self.stats.compile([solution]))
                if self.verbose:
                    if (int)(temp * self.number_gen/ self.initial_temp) % self.verbose == 0:
                        print("Temperature: ", temp ,  " TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)

            best.update(hof)
            if self.predict_with == 'all':
                self.mask_.append(hof[0][:])
                self.fitnesses_.append(hof[0].fitness.values)

        self.mask_ = np.array(self.mask_)
        self.best_mask_ = np.asarray(best[0][:], dtype=bool)
        self.fitness_ = best[0].fitness.values

        features = list(compress(range(len(self.best_mask_)), self.best_mask_))
        train = np.reshape([X[:, i] for i in features], [len(features), len(X)]).T

        self.estimator.fit(X=train, y=y)

        return self

    def _metropolis_criterion(solution, prev_solution, temp):
        prob = np.exp( (sum(solution.fitness.wvalues) - sum(prev_solution.fitness.wvalues))/temp )
      
        if random.random() < prob:
            return solution
        else:
            return prev_solution
        
