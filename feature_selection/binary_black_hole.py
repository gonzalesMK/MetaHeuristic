from __future__ import print_function
import random
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.utils import check_random_state


class BinaryBlackHole(_BaseMetaHeuristic):
    """Implementation of Binary Black Hole for Feature Selection

    Parameters
    ----------
    classifier : sklearn classifier , (default=SVM)
            Any classifier that adheres to the scikit-learn API
    
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

    def __init__(self, classifier=None, number_gen=10, size_pop=40, verbose=False, 
                 repeat=1, make_logbook=False, random_state=None, parallel=False,
                 cv_metric_fuction=None, features_metric_function=None):
    
        super(BinaryBlackHole, self).__init__(
                name = "BinaryBlackHole",
                classifier=classifier, 
                number_gen=number_gen,  
                verbose=verbose,
                repeat=repeat,
                parallel=parallel, 
                make_logbook=make_logbook,
                random_state=random_state,
                cv_metric_fuction=cv_metric_fuction,
                features_metric_function=features_metric_function)

        self.size_pop = size_pop        
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("star", tools.initIterate, BaseMask, self.toolbox.attribute)
        self.toolbox.register("galaxy", tools.initRepeat, list, self.toolbox.star)
        self.toolbox.register("update", self._updateStar)
        self.toolbox.register("evaluate", self._evaluate)
        self.parallel = parallel
        
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
        self.set_params(**arg)
        initial_time = time.clock()
        
        if normalize:
            self._sc_X = StandardScaler()
            X = self._sc_X.fit_transform(X)
        
        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        
        self.normalize_ = normalize
        self.n_features_ = X.shape[1]
        self.mask_ = []
        self.fitnesses_ = []
        
        # pylint: disable=E1101
        random.seed(self.random_state)
        self._random_object = check_random_state(self.random_state)
        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)

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
            galaxy = self.toolbox.galaxy(n=self.size_pop)
            hof = tools.HallOfFame(1)

            for g in range(self.number_gen):

                # Evaluate the entire population
                fitnesses = self.toolbox.map(self.toolbox.evaluate, galaxy)
                for ind, fit in zip(galaxy, fitnesses):
                    ind.fitness.values = fit

                # Update Global Information
                hof.update(galaxy)    
                hof[0].radius = sum(hof[0].fitness.wvalues) / sum( [sum(i.fitness.wvalues) for i in galaxy] )
                 
                # Update particles
                for part in galaxy:
                    self.toolbox.update(part, hof[0])

                # Log statistic
                hof.update(galaxy)
                if self.make_logbook:
                        self.logbook[i].record(gen=g,
                                               best_fit=hof[0].fitness.values[0],
                                               **self.stats.compile(galaxy))
                if self.verbose:
                    print("Repetition:", i+1 ,"Generation: ", g + 1, "/", self.number_gen,
                          "Elapsed time: ", time.clock() - initial_time, end="\r")


            best.update(hof)
            if self.make_logbook :
                self.mask_.append(hof[0][:])
                self.fitnesses_.append(hof[0].fitness.values)

        self.mask_ = np.array(self.mask_)
        self.best_mask_ = np.asarray(best[0][:], dtype=bool)
        self.fitness_ = best[0].fitness.values

        self.estimator.fit(X= self.transform(X), y=y)

        return self

    @staticmethod
    def _dist(star, blackhole):
        return np.linalg.norm([blackhole[i] - star[i] for i in range(0, len(star))])

    def _updateStar(self, star, blackhole):
        if self._dist(star, blackhole) < blackhole.radius :
            star[:] = self.toolbox.galaxy(n=1)[0]            
        else:
            star[:] = [ 1 if  abs(np.tanh(star[x] + self._random_object.uniform(0,1) * (blackhole[x] - star[x]))) > self._random_object.uniform(0,1) else 0 for x in range(0,self.n_features_)]
    
    
