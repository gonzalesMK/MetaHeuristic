# -*- coding: utf-8 -*-
from __future__ import print_function
from timeit import time

import numpy as np

from deap import base
from deap import tools

from .base_pareto import _BaseMetaHeuristicPareto
from .base_pareto import BaseMask

import random

class SPEA2(_BaseMetaHeuristicPareto):
    """Implementation of Strenght Pareto Front Envolutionary Algorithm 2
    
    https://pdfs.semanticscholar.org/6672/8d01f9ebd0446ab346a855a44d2b138fd82d.pdf

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
            
    cv_metric_fuction : callable, (default=matthews_corrcoef)            
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
    
    features_metric_function : { "log", "poly" }
            A function that return a float from the binary mask of features
    """

    def __init__(self, classifier=None,
                 archive_size = 1, cxUniform_indpb = 0.2,
                 number_gen=10, size_pop=3, verbose=0, repeat=1,
                 individual_mut_prob=0.5, gene_mutation_prob=0.01,
                 make_logbook=False, random_state=None, parallel=False,
                 cv_metric_fuction=None, features_metric_function="log",
                 print_fnc = None):
    
        super(SPEA2, self).__init__(
                name = "SPEA2",
                classifier=classifier, 
                number_gen=number_gen,  
                verbose=verbose,
                repeat=repeat,
                parallel=parallel, 
                make_logbook=make_logbook,
                random_state=random_state,
                cv_metric_fuction=cv_metric_fuction,
                features_metric_function=features_metric_function,
                print_fnc=print_fnc)
        
        self.size_pop = size_pop        
        self.archive_size = archive_size
        self.cxUniform_indpb = cxUniform_indpb    
        self.individual_mut_prob=individual_mut_prob
        self.gene_mutation_prob = gene_mutation_prob
        self.parallel = parallel
        
    def _make_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate,
                              BaseMask, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxUniform, indpb = self.cxUniform_indpb)
        self.toolbox.register("select", tools.selTournament, tournsize=2)
        self.toolbox.register("mutate", tools.mutUniformInt, low=0, up=1,
                              indpb=self.gene_mutation_prob)
        
        self.toolbox.register("map", map)
        self.toolbox.register("evaluate", self._evaluate, X= None, y=None)
        
        if self.parallel:
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
        self._make_toolbox()
        self.set_params(**arg)
        
        X,y = self._set_dataset(X=X, y=y, normalize=normalize)
        
        self._set_fit()
        
        for i in range(self.repeat):
            # Generate Population
            pop = self.toolbox.population(self.size_pop)
            hof = tools.HallOfFame(1)
            pareto_front = tools.ParetoFront()

            # Evaluate the entire population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            pareto_front.update(pop)   
            archive = tools.selSPEA2(pop, self.archive_size)
            
            for g in range(self.number_gen):
                
                # Mating Selection
                pop = self.toolbox.select(archive, self.size_pop)
                
                # Clone the selected individuals
                pop = list(map(self.toolbox.clone, pop))

                # Apply variation
                pop = self._variation(pop)

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in pop if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
                
                # Environmental Selection
                archive = tools.selSPEA2( archive + pop, self.archive_size)
                
                # Log Statistics 
                hof.update(archive)
                pareto_front.update(archive)
                if self.make_logbook:
                        self.logbook[i].record(gen=g,
                                               best_fit=hof[0].fitness.values[0],
                                               **self.stats.compile(archive))
                        self._make_generation( hof, pareto_front)
                        
                if self.verbose:
                    self._print(g, i, initial_time, time.clock())

            self._make_repetition(hof,pareto_front)

        self.estimator.fit(X= self.transform(X), y=y)

        return self

    def _variation(self, offspring):
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
                self.toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < self.individual_mut_prob:
                self.toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    def set_params(self, **params):
        super(SPEA2, self).set_params(**params)

        if self.parallel:
            from multiprocessing import Pool
            self.toolbox.register("map", Pool().map)
        else:
            self.toolbox.register("map", map)    
            
        return self