import numpy as np

from random import randint
import random
from itertools import compress
from datetime import datetime

from deap import base, creator
from deap import tools

from base import _BaseMetaHeuristic
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
from sklearn.svm import  SVC

from sklearn.utils import check_random_state

class GeneticAlgorithm(_BaseMetaHeuristic):
    """Implementation of a Genetic Algorithm for Feature Selection 
    
    Parameters
    ----------
    predict_with : one of { 'all' , 'best' }
        'all' - will predict the X with all masks and return the mean
        
        'best' - will predict the X using the mask with the best fitness
        
        obs: If you are going to make grid search in hyperparameters, use 'all'
    
    X : array of shape [n_samples, n_features]
            The input samples
    
    y : array of shape [n_samples, 1]            
            The input of labels 
    
    Cross_over_prob :  float in [0,1]
            Probability of happening a cross-over in a individual (chromosome) 
    
    individual_mutation_probability : float in [0,1]
            Probability of happening mutation in a individual ( chromosome )
            
    gene_mutation_prob : float in [0,1]
            For each gene in the individual (chromosome) chosen for mutation, 
            is the probability of it being mutate
            
    number_gen : positive integer
            Number of generations
            
    size_pop : positive integer
            Number of individuals (choromosome ) in the population
            
    verbose : boolean
            Print information
            
    repeat_ : positive int
            Number of times to repeat the fitting process
        
    make_logbook: boolean
            If True, a logbook from DEAP will be made
    """

    def __init__(self, estimator = None, X=None, y=None, cross_over_prob = 0.2, 
                 individual_mut_prob = 0.05, gene_mutation_prob = 0.05, 
                 number_gen = 20, size_pop = 40, verbose = 0, repeat_ = 1, 
                 predict_with = 'best', make_logbook = False, random_state = None):
        
        self.individual_mut_prob = individual_mut_prob
        self.number_gen = number_gen
        self.cross_over_prob = cross_over_prob
        self.size_pop = size_pop
        self.score_func = None
        self.estimator = estimator
        self.repeat_ = repeat_
        self.fitness = []
        self.mask = []
        self.predict_with =  predict_with
        self.gene_mutation_prob = gene_mutation_prob
        self.make_logbook = make_logbook
        self.verbose = verbose
        self.random_state = check_random_state(random_state)

        creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                         self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("map", map)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

          
        if( make_logbook ):
            self.logbook = tools.Logbook()
            self.logbook.header = ["gen"] + self.stats.fields        
        
        if(( type(X) != type(None) and type(y) == type(None) ) or (type(X) == type(None) and type(y) != type(None))):
                raise ValueError("It's necessary to input both X and y datasets")
        elif(type(X) != type(None) and type(y) != type(None)):
            self.toolbox.register("evaluate", self._evaluate, X = X, y = y)
            self.X = X
            self.y = y
        else:   
            self.X = None
            self.y = None
        
    def fit(self,X = None, y = None, normalize = True, **arg):
        """ Fit method
    
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
                The input samples
    
        y : array of shape [n_samples, 1]            
                The input of labels 
                
        normalize : boolean
                If true, StandardScaler will be applied to X
                
        **arg : parameters
                Set parameters
        """
        self.set_params(**arg)    
        
        if( type(X) == type(None) ):
            if(type(self.X) == type(None)):
                raise ValueError("You need to input X data")
            else:
                X = self.X
        else:
            self.X = X
        
        if( type(y) == type(None)):
            if(type(self.y) == type(None)):
                raise ValueError("You need to input y data")
            else:
                y = self.y
        else:
            self.y = y
                        
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')            
        self.n_features = len(X)   
        self.toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = self.gene_mutation_prob)        
        
        if( normalize ):
            sc_X = StandardScaler()
            X = sc_X.fit_transform(X)

        self.toolbox.register("evaluate", self._evaluate, X = X, y = y)        
        
        best = tools.HallOfFame(1)
        for i in range(self.repeat_):
            pop = self.toolbox.population(self.size_pop) 
            hof = tools.HallOfFame(1)
            # Evaluate the entire population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit
            
            for g in range(self.number_gen):
                # Select the next generation individuals
                offspring = self.toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = list(map(self.toolbox.clone, offspring)) 
            
                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cross_over_prob:
                        self.toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values
            
                for mutant in offspring:
                    if random.random() < self.individual_mut_prob:
                        self.toolbox.mutate(mutant)
                        del mutant.fitness.values
            
                # Evaluate the individuals with an invalid fitness ( new individuals)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit
            
                # The population is entirely replaced by the offspring
                pop[:] = offspring
                
                # Log statistic
                hof.update(pop)
                if(self.make_logbook):
                    self.logbook.record(gen=g, **self.stats.compile(pop))
                if(self.verbose):
                    if( g % self.verbose == 0):
                        print("Generation: ", g + 1 , "/", self.number_gen, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
                    
            best.update(hof)
        
        self.best_mask = best[0][:]
        self.best_fitness = best[0].fitness.values
        features = list( compress( range(len(self.best_mask)), self.best_mask))
        train =  np.reshape([self.X[:, i] for i in features], [ len(features),  len(self.X)]).T
        self.estimator.fit(X = train, y = self.y)
        
        return self
    
class HarmonicSearch(_BaseMetaHeuristic):    
    """Implementation of a Harmonic Search Algorithm for Feature Selection 
    
    Parameters
    ----------
    predict_with : one of { 'all' , 'best' }
        'all' - will predict the X with all masks and return the mean
        
        'best' - will predict the X using the mask with the best fitness
        
        obs: If you are going to make grid search in hyperparameters, use 'all'
    
    HMCR : float in [0,1]
            Is the Harmonic Memory Considering Rate

    indpb : float in [0,1]
            Is the mutation rate of each new harmony
            
    pitch : float in [0,1]
            Is the Pitch Adjustament factor
    
    X : array of shape [n_samples, n_features]
            The input samples
    
    y : array of shape [n_samples, 1]            
            The input of labels 
    
    number_gen : positive integer
            Number of generations
            
    size_pop : positive integer
            Size of the Harmonic Memory
            
    verbose : boolean
            Print information
            
    repeat_ : positive int
            Number of times to repeat the fitting process
        
    make_logbook: boolean
            If True, a logbook from DEAP will be made
    """

    def __init__(self, classifier = None, X=None, y=None, HMCR = 0.95, indpb = 0.05, 
                 pitch = 0.05, number_gen = 100, size_pop = 50, verbose = 0,
                 repeat_ = 1, predict_with = 'best', make_logbook = False, random_state = None):

        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.Fitness)
        
        self.HMCR = HMCR
        self.indpb = indpb
        self.pitch = pitch
        self.number_gen = number_gen
        self.size_pop = size_pop
        self.score_func = None
        self.estimator = SVC(kernel = 'linear', verbose=  False, max_iter = 10000) if classifier is None else classifier
        self.X = None
        self.y = None
          
        self.repeat_ = repeat_
        self.fitness = []
        self.mask = []
        self.predict_with =  predict_with
        self.make_logbook = make_logbook
        self.verbose = verbose
        self.random_state = random_state
        self.random_object = check_random_state(random_state)
        
        if( make_logbook ):
            self.stats = tools.Statistics(lambda ind: ind.fitness.values)
            self.stats.register("avg", np.mean)
            self.stats.register("std", np.std)
            self.stats.register("min", np.min)
            self.stats.register("max", np.max)
            self.logbook = tools.Logbook()
            self.logbook.header = ["gen"] + ["best_fit"] + self.stats.fields

    def fit(self,X = None, y = None):
        
        y = y if y is not None else self.y
        X = X if X is not None else self.X
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')
        self._validate_targets(y)
        self.n_features = X.shape[1]
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("get_worst", tools.selWorst, k = 1)
        self.toolbox.register("evaluate", self._evaluate, X = X, y = y )
        self.toolbox.register("map", map)
        self.toolbox.register("improvise", self._improvise, HMCR = self.HMCR)
        self.toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = self.indpb)
        self.toolbox.register("pitch_adjustament", tools.mutFlipBit, indpb = self.pitch)
        #toolbox.register("map", futures.map)
        
        best = tools.HallOfFame(1)
        for i in range(self.repeat_):
            harmony_mem = self.toolbox.population(n=self.size_pop) 
            hof = tools.HallOfFame(1)
                
            # Evaluate the entire population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, harmony_mem)
            
            for ind, fit in zip(harmony_mem, fitnesses):
                ind.fitness.values = fit
            
            for g in range(self.number_gen):
    
                # Improvise a New Harmony 
                new_harmony = self.toolbox.improvise(harmony_mem)
                new_harmony.fitness.values = self.toolbox.evaluate(new_harmony)
    
                # Select the Worst Harmony
                worst = self.toolbox.get_worst(harmony_mem)[0]
    
                # Check and Update Harmony Memory
                if( worst.fitness.values < new_harmony.fitness.values):
                    worst[:] = new_harmony[:]
                    worst.fitness.values = new_harmony.fitness.values
                
                # Log statistic
                hof.update(harmony_mem)
                if( self.make_logbook):
                    self.logbook.record(gen=g, best_fit= hof[0].fitness.values[0], **self.stats.compile(harmony_mem))
                if(self.verbose):
                    if( g % self.verbose == 0):
                        print("Generation: ", g + 1 , "/", self.number_gen, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
                #scoop.logger.info("Generation: %d", g)
            best.update(hof)
        

        self.best_mask =best[0][:]
        self.best_fitness = best[0].fitness.values
        
        features = list( compress( range(len(self.best_mask)), self.best_mask))
        train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]).T
        self.estimator.fit(X = train, y = y)
        
        return self
        
    def _improvise(self,pop, HMCR):
        """ Function that improvise a new harmony"""
    # HMCR = Harmonic Memory Considering Rate
        size = len(pop)
        new_harmony = self.toolbox.individual()
        
        rand_list = self.random_object.randint(0,size-1, len(new_harmony))
        
        for i in range(len(new_harmony)):
            new_harmony[i] = pop[rand_list[i]][i] 
        self.toolbox.mutate(new_harmony)
        self.toolbox.pitch_adjustament(new_harmony)
        
        return new_harmony
    
