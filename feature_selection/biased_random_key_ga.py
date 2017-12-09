# -*- coding: utf-8 -*-

from __future__ import print_function
import random
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base import _BaseMetaHeuristic
from .base import BaseMask

from sklearn.utils import check_random_state

class BRKGA(_BaseMetaHeuristic):
    """Implementation of a Biased Random Key Genetic Algorithm as the papers:
    Biased random-key genetic algorithms for combinatorial optimization
    
    Introdução aos algoritmos genéticos de chaves aleatórias viciadas

    Parameters
    ----------
    classifier : sklearn classifier , (default=SVM)
            Any classifier that adheres to the scikit-learn API
    
    elite_size : positive integer, (default=10)
            Number of individuals in the Elite population            
            
    mutant_size : positive integer, (default=10)
            Number of new individuals in each iteration
            
    number_gen : positive integer, (default=10)
            Number of generations

    cxUniform_indpb : float in [0,1], (default=0.5)
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
            
    cv_metric_fuction : callable, (default=matthews_corrcoef)            
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    
    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    """

    def __init__(self, classifier=None,
                 elite_size = 1, mutant_size = 1, cxUniform_indpb = 0.7,
                 number_gen=1, size_pop=3, verbose=0, repeat=1,
                 make_logbook=False, random_state=None, parallel=False,
                 cv_metric_fuction=None, features_metric_function=None):
    
        super(BRKGA, self).__init__(
                name = "BRKGA",
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
        if( elite_size + mutant_size > size_pop ):
            print(" Elite size({}) + Mutant_size({}) is bigger than population"
                   " size({})\n The algorithm may not work properly".format( 
                  elite_size, mutant_size, size_pop))
        
        self.cxUniform_indpb = cxUniform_indpb    
        self.elite_size = elite_size
        self.non_elite_size = size_pop - elite_size
        self.mutant_size = mutant_size
        self.n_cross_over = size_pop - (elite_size + mutant_size)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              BaseMask, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxUniform, indpb = self.cxUniform_indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("map", map)
        self.toolbox.register("evaluate", self._evaluate, X= None, y=None)

        if parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool(processes=4).map)
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
        
        if self.make_logbook:
            self._make_stats()

        self._random_object = check_random_state(self.random_state)
        random.seed(self.random_state)

        best = tools.HallOfFame(1)
        for i in range(self.repeat):
            # Generate Population
            pop = self.toolbox.population(self.size_pop)
            hof = tools.HallOfFame(1)
            
            # Evaluate the entire population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for g in range(self.number_gen):
                # Partitionate elite members
                elite = tools.selBest( pop, self.elite_size)
                non_elite = tools.selWorst( pop, self.non_elite_size)
                
                # Cross_over between Elite and Non Elite 
                father_ind = np.random.randint(0, self.elite_size, self.n_cross_over)
                mother_ind = np.random.randint(0, self.non_elite_size, self.n_cross_over)
                
                child1 = [self.toolbox.clone(elite[ind]) for ind in father_ind]
                child2 = [self.toolbox.clone(non_elite[ind]) for ind in mother_ind]
                
                self.toolbox.mate(child1, child2)
                for ind1 in child1:
                    del ind1.fitness.values
                    
                # Evaluate the individuals with an invalid fitness ( new individuals)
                fitnesses = self.toolbox.map(self.toolbox.evaluate, child1)
                for ind, fit in zip(child1, fitnesses):
                    ind.fitness.values = fit

                # The botton is replaced by mutant individuals
                mutant = self.toolbox.population(self.mutant_size)
                fitnesses = self.toolbox.map(self.toolbox.evaluate, mutant)
                for ind, fit in zip(mutant, fitnesses):
                    ind.fitness.values = fit
                    
                # The population is entirely replaced by the offspring
                pop[:] = elite + child1 + mutant

                # Log Statistics 
                hof.update(pop)
                if self.make_logbook:
                        self.logbook[i].record(gen=g,
                                               best_fit=hof[0].fitness.values[0],
                                               **self.stats.compile(pop))
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
    
    def set_params(self, **params):
        super(BRKGA, self).set_params(**params)

        if( self.elite_size + self.mutant_size > self.size_pop ):
            print(" Elite size({}) + Mutant_size({}) is bigger than population"
                   " size({})\n The algorithm may not work properly".format( 
                  self.elite_size, self.mutant_size, self.size_pop))

        if self.parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)    
            
        return self