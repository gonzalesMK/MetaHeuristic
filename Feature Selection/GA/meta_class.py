from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.externals import six
from random import sample
from random import randint
import random


from deap import base, creator
from deap import tools

import numpy as np

from itertools import compress
from datetime import datetime
from abc import ABCMeta
from warnings import warn

from sklearn.preprocessing import StandardScaler

def safe_mask(X, mask):
    """Return a mask which is safe to use on X.
    Parameters
    ----------
    X : {array-like, sparse matrix}
        Data on which to apply mask.
    mask : array
        Mask to be used on X.
    Returns
    -------
        mask
    """
    mask = np.asarray(mask)
    if np.issubdtype(mask.dtype, np.int):
        return mask

    if hasattr(X, "toarray"):
        ind = np.arange(mask.shape[0])
        mask = ind[mask]
    return mask


class SelectorMixin(six.with_metaclass(ABCMeta, TransformerMixin)):
    """
    Transformer mixin that performs feature selection given a support mask
    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_support_mask`.
    """

    def get_support(self, indices=False):
        """
        Get a mask, or integer index, of the features selected
        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.
        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]

    def _get_support_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """

    def transform(self, X):
        """Reduce X to the selected features.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.
        Returns
        -------
        X_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """
        X = check_array(X, accept_sparse='csr')
        mask = self.get_support()
        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
        return X[:, safe_mask(X, mask)]

class _BaseFilter(BaseEstimator, SelectorMixin):
    """Initialize the univariate feature selection.
    Parameters
    ----------
    score_func : callable
        Function taking two arrays X and y, and returning a pair of arrays
        (scores, pvalues) or a single array with scores.
    """

    def __init__(self):
        self.score_func = None

    def fit(self, X, y):
        """Run score function on (X, y) and get the appropriate features.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.
        y : array-like, shape = [n_samples]
            The target values (class labels in classification, real numbers in
            regression).
        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, ['csr', 'csc'], multi_output=True)

        if not callable(self.score_func):
            raise TypeError("The score function should be a callable, %s (%s) "
                            "was passed."
                            % (self.score_func, type(self.score_func)))

        self._check_params(X, y)
        score_func_ret = self.score_func(X, y)
        if isinstance(score_func_ret, (list, tuple)):
            self.scores_, self.pvalues_ = score_func_ret
            self.pvalues_ = np.asarray(self.pvalues_)
        else:
            self.scores_ = score_func_ret
            self.pvalues_ = None

        self.scores_ = np.asarray(self.scores_)

        return self

    def _check_params(self, X, y):
        pass
    
class _BaseMetaHeurisct(_BaseFilter):
    
    def _gen_in(self):
        """ Generate a individual, DEAP function

        """
        RND = randint(0,self.n_features)
        
        return   sample(list(np.concatenate( (np.zeros([self.n_features-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), self.n_features)

        # Evaluation Function 
    def _evaluate(self, individual, X, y, cv = 3):
        """ Evaluate method
    
        Parameters
        ----------
        individual: list [n_features]
                The input individual to be evaluated
    
        Return
        ----------
        
        Score of the individual : float
        """
        # Select Features
        features = list( compress( range(len(individual)), individual))
        train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]).T
        
        if( train.shape[1] == 0 ):
            return 0,
               
        # Applying K-Fold Cross Validation
        accuracies = cross_val_score( estimator = clone(self.estimator) , X = train, y = y, cv = 3)
        
        return accuracies.mean() - accuracies.std(), pow(sum(individual)/(len(X)*5),2),

    def transform(self, X,mask = None):
        """ Transform method
    
        Parameters
        ----------
        
        X : array of shape [n_samples, n_features]
                The input samples
                
        mask : list of {0,1} or boolean  [n_features]    
                The features with 1 will be selected, otherwise discarted
                
    
        Return
        ----------
        
        X' : new array of shape [n_samples, sum(n_features)]
        """
        if( type(mask) == type(None)):
            features = list( compress( range(len(self.best_mask)), self.best_mask))
            return np.reshape([X[:, i] for i in features], [ len(features),  len(X)]).T
        else:
            features = list( compress( range(len(mask)), mask))
            return np.reshape([X[:, i] for i in features], [ len(features),  len(X)]).T

    def _get_support_mask(self):
        check_is_fitted(self, 'support_')
        return self.support_
    
    def predict(self,X):
        
        if( self.predict_with == 'best'):
            X_ = self.transform(X)
            return self.estimator.predict(X_)
        
        elif(self.predict_with == 'all'):
            predict_ = []
            for mask in self.mask:
                X_ = self.transform(X, mask = mask)
                self.estimator.fit(X = self.transform(self.X, mask = mask), y = self.y)                    
                predict_.append(self.estimator.predict(self.transform(X, mask = mask)))
                     
            return predict_
    
    def score_func_to_grid_search(_, estimator,X_test, y_test):
        """ Function to be given as a scorer function to Grid Search Method.
        It is going to transform the matrix os predicts generated by 'all' option
        to an final accuracy score.
        """

        p1 = 0
        
        if( len(estimator.mask) == 0):  
            print("No masks to test")
            print(estimator.get_params)
            return 0

        y_pred = estimator.predict(X_test)
        
        for y in y_pred:
            p1 = sum(y_test == y) + p1
        
        return p1/( len(y_test) * len(estimator.mask))

class GeneticAlgorithm(_BaseMetaHeurisct):
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

    def __init__(self, estimator, X=None, y=None, cross_over_prob = 0.2, 
                 individual_mut_prob = 0.05, gene_mutation_prob = 0.05, 
                 number_gen = 20, size_pop = 40, verbose = 0, repeat_ = 1, 
                 predict_with = 'best', make_logbook = False):
        
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
                    if( g % self.verbose == 0)
                    print("Generation: ", g + 1 , "/", self.number_gen, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
                    
            best.update(hof)
        
        self.best_mask = best[0][:]
        self.best_fitness = best[0].fitness.values
        features = list( compress( range(len(self.best_mask)), self.best_mask))
        train =  np.reshape([self.X[:, i] for i in features], [ len(features),  len(self.X)]).T
        self.estimator.fit(X = train, y = self.y)
        
        return self
    
class HarmonicSearch(_BaseMetaHeurisct):    
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

    def __init__(self, estimator, X=None, y=None, HMCR = 0.95, indpb = 0.05, 
                 pitch = 0.05, number_gen = 20, size_pop = 40, verbose = 0,
                 repeat_ = 1, predict_with = 'best', make_logbook = False):

        creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.Fitness)
        
        self.HMCR = HMCR
        self.indpb = indpb
        self.pitch = pitch
        self.number_gen = number_gen
        self.size_pop = size_pop
        self.score_func = None
        self.estimator = estimator
        self.repeat_ = repeat_
        self.fitness = []
        self.mask = []
        self.predict_with =  predict_with
        self.make_logbook = make_logbook
        self.verbose = verbose
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("get_worst", tools.selWorst, k = 1)
        self.toolbox.register("evaluate", self._evaluate)
        self.toolbox.register("map", map)
        self.toolbox.register("improvise", self._improvise, HMCR = self.HMCR)
        self.toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = self.indpb)
        self.toolbox.register("pitch_adjustament", tools.mutFlipBit, indpb = self.pitch)
        #toolbox.register("map", futures.map)
        
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        if( make_logbook ):
            self.logbook = tools.Logbook()
            self.logbook.header = ["gen"] + ["best_fit"] + self.stats.fields
        
        if( verbose):
            self.toolbox.register("print", print)
        else:
            self.toolbox.register("print", lambda *args, **kwargs: None)
        
        if(( type(X) != type(None) and type(y) == type(None) ) or (type(X) == type(None) and type(y) != type(None))):
                raise ValueError("It's necessary to input both X and y datasets")
        elif(type(X) != type(None) and type(y) != type(None)):
            self.toolbox.register("evaluate", self._evaluate, X = X, y = y)
            self.X = X
            self.y = y
        else:   
            self.X = None
            self.y = None

    def fit(self,X = None, y = None, **arg):

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

        self.n_features = len(X)   
        self.toolbox.register("improvise", self._improvise, HMCR = self.HMCR)
        self.toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = self.indpb)
        self.toolbox.register("pitch_adjustament", tools.mutFlipBit, indpb = self.pitch)
        self.toolbox.register("evaluate", self._evaluate, X = X, y = y)
        
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
        train =  np.reshape([self.X[:, i] for i in features], [ len(features),  len(self.X)]).T
        self.estimator.fit(X = train, y = self.y)
        
        # Function that improvise a new harmony
    def _improvise(self,pop, HMCR):
    # HMCR = Harmonic Memory Considering Rate
        size = len(pop)
        new_harmony = self.toolbox.individual()
        for i,x in enumerate(pop):
            new_harmony[i] = pop[randint(0,size-1)][i] 
        self.toolbox.mutate(new_harmony)
        self.toolbox.pitch_adjustament(new_harmony)
        
        return new_harmony
    
