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
    

    
class genetic_algorithm(_BaseFilter):
    """Implementation of a Genetic Algorithm for Feature Selection 
    
    Parameters
    ----------
    predict_with : strin
        'all' - will predict the X with all masks and return the mean
        'best' - will predict the X wihtl the mask with the best fitness
        you may use 'all' to make grid search for hyperparameters
    
    X : array of shape [n_samples, n_features]
            The input samples.
    
    y : array of shape [n_samples, 1]            
            The input of labels 
    
    Cross_over_prob :  float in [0,1]
            Probability of happening a cross-over in a chromosome 
    
    individual_mutation_probability : float in [0,1]
            Probability of happening mutation in a chromosome 
            
    gene_mutation_prob : float in [0,1]
            For each gene in the chromosome chosen for mutation, 
            is the probability of it being mutate
            
    number_gen : positive integer
            Number of generations
            
    size_pop : positive integer
            Number of individuals in the population
            
    verbose : boolean
            Print information
            
    repeat : positive int
            Number of times to repeat the fitting process
    """

    def __init__(self, estimator, X=None, y=None, cross_over_prob = 0.2, individual_mutation_probability = 0.05, gene_mutation_prob = 0.05, number_gen = 20, size_pop = 40, verbose = False, repeat_ = 1, predict_with = 'best'):
        
        self.mutation_prob = individual_mutation_probability
        self.number_gen = number_gen
        self.cross_over_prob = 0.2
        self.size_pop = size_pop
        self.score_func = None
        self.estimator = estimator
        self.repeat = repeat_
        self.fitness = []
        self.mask = []
        self.predict_with =  predict_with
        
        creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self._gen_in)
        self.toolbox.register("individual", tools.initIterate, creator.Individual,
                         self.toolbox.attribute)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = gene_mutation_prob)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("map", map)

        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)

        self.logbook = tools.Logbook()
        self.logbook.header = ["gen"] + self.stats.fields        
        
        if( verbose):
            self.toolbox.register("print", print)
        else:
            self.toolbox.register("print", lambda *args, **kwargs: None)
        
        if(( type(X) != type(None) and type(y) == type(None)) or (type(X) == type(None) and type(y) != type(None))):
                raise ValueError("It's necessary to input both X and y datasets")
        
        if(type(X) != type(None) and type(y) != type(None)):
            self.toolbox.register("evaluate", self._evaluate, X = X, y = y)
            self.X = X
            self.y = y
        else:   
            self.X = None
            self.y = None
        
    def fit(self,X = None, y = None, normalize = True):
        
        if( type(X) == type(None)):
            if(type(self.X) == type(None)):
                raise ValueError("You need to input X data")
            else:
                X = self.X
        else:
            self.x = X

        if( type(y) == type(None)):
            if(type(self.y) == type(None)):
                raise ValueError("You need to input y data")
            else:
                y = self.y
        else:
            self.y = y
                        
        self.n_features = len(X)   
        
        if( normalize ):
            sc_X = StandardScaler()
            X = sc_X.fit_transform(X)

        self.toolbox.register("evaluate", self._evaluate, X = X, y = y)        
        
        for i in range(self.repeat):
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
                    if random.random() < self.mutation_prob:
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
                self.logbook.record(gen=g, **self.stats.compile(pop))
                self.toolbox.print("Generation: ", g + 1 , "/", self.number_gen, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
                
            self.mask.append(hof[0][:])
            self.fitness.append(hof[0].fitness.values)
        
        indx, indy = np.argmax(self.fitness, axis = 0)
        self.best_mask = self.mask[indx]
        features = list( compress( range(len(self.best_mask)), self.best_mask))
        train =  np.reshape([self.X[:, i] for i in features], [ len(features),  len(self.X)]).T
        self.estimator.fit(X = train, y = self.y)
        
    def _gen_in(self):
        RND = randint(0,self.n_features)
        
        return   sample(list(np.concatenate( (np.zeros([self.n_features-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), self.n_features)

        # Evaluation Function 
    def _evaluate(self, individual, X, y, cv = 3):

        # Select Features
        features = list( compress( range(len(individual)), individual))
        train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]).T
        
        if( train.shape[1] == 0 ):
            return 0,
               
        # Applying K-Fold Cross Validation
        accuracies = cross_val_score( estimator = clone(self.estimator) , X = train, y = y, cv = 3)
        
        return accuracies.mean() - accuracies.std(), pow(sum(individual)/(len(X)*5),2),

    def transform(self, X,mask = None):
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
            predict = []
            for mask in self.mask:
                X_ = self.transform(X, mask = mask)
                self.estimator.fit(X = self.transform(self.X, mask = mask), y = self.y)                    
                predict.append(self.estimator.predict(X))
                     
            return predict
    
    def score_func_to_grid_search(y_true, y_pred):
        p1 = 0
        for y in y_pred:
            p1 = sum(y_true == y_pred) + p1
        return p1/( len(y_true) + len(y_pred))