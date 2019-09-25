from abc import ABCMeta
from warnings import warn
from itertools import compress
from random import sample
import random
import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
import six
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
from sklearn.svm import SVC
from deap import base
from deap import tools
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y
import copy
from timeit import time

from collections import Counter
class Fitness(base.Fitness):

    def __init__(self, weights=(1, -1e-5), values=(0, 0)):
        self.weights = weights
        super(Fitness, self).__init__(values)


class BaseMask(list, object):

    def __init__(self, mask):
        self[:] = mask
        self.fitness = Fitness((1, -1e-5), (0, 0))

class SelectorMixin(six.with_metaclass(ABCMeta, TransformerMixin)):
    """
    Transformer mixin that performs feature selection given a support mask
    This mixin provides a feature selector implementation with `transform` and
    `inverse_transform` functionality given an implementation of
    `_get_best_mask_mask`.
    """
    @staticmethod
    def safe_mask(x, mask):
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

        if np.issubdtype(mask.dtype, np.unsignedinteger) or np.issubdtype(mask.dtype, np.signedinteger) or np.issubdtype(np.dtype(mask.dtype).type, np.dtype(np.bool).type):
            if x.shape[1] != len(mask):
                raise ValueError("X columns %d != mask length %d"
                                 % (x.shape[1], len(mask)))
        else:
            raise ValueError("Mask type is {} not allowed".format(mask.dtype))

        return mask

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
        mask = self._get_best_mask()
        mask = mask if not indices else np.where(mask)[0]
        return np.asarray(mask, dtype=np.bool)

    def _get_best_mask(self):
        """
        Get the boolean mask indicating which features are selected
        Returns
        -------
        support : boolean array of shape [# input features]
            An element is True iff its corresponding feature is selected for
            retention.
        """
        check_is_fitted(self, 'best_')
        return np.asarray(self.best_[0][:], dtype=bool)

    def transform(self, X, mask=None):
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

        if mask is None:
            mask = self.get_support()

        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))

        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")

        return X[:, self.safe_mask(X, mask)]


class _BaseMetaHeuristic(BaseEstimator, SelectorMixin, MetaEstimatorMixin):

    def __init__(self, name, estimator=None, number_gen=20,
                 verbose=0, repeat=1, parallel=False,
                 make_logbook=False, random_state=None,
                 cv_metric_function=make_scorer(matthews_corrcoef),
                 ):

        self.name = name
        self.estimator = estimator
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_function

        np.random.seed(self.random_state)

    def _gen_in(self):
        """ 
            Generate a individual, DEAP function
        """
        random_number = self._random_object.randint(1, self.n_features_ + 1)
        zeros = (np.zeros([self.n_features_-random_number, ], dtype=int))
        ones = np.ones([random_number, ], dtype=int)
        return sample(list(np.concatenate((zeros, ones), axis=0)), self.n_features_)

    def _evaluate(self, individual, X, y, cv=2):
        """ 
            Evaluate method. Each individual is a mask of features.

            Given one individual, train the estimator on the dataset and get the scores 

        Parameters
        ----------
        individual: list [n_features]
                The input individual to be evaluated


        Return
        ----------
        Score of the individual : turple( cross valid score, feature length score)
        """
        # Select Features
        features = list(compress(range(len(individual)), individual))
        train = np.reshape([X[:, i] for i in features],
                           [len(features), len(X)]).T

        if train.shape[1] == 0:
            return 0, 1,

        try:
            index = self._all_solutions['individuals'].index(individual[:])
            return self._all_solutions['results'][index]
        except ValueError:
            pass

        # Applying K-Fold Cross Validation
        accuracies = cross_val_score(estimator=clone(self._estimator), X=train,
                                     y=y, cv=cv,
                                     scoring=self.cv_metric_function)

        if hasattr(self, 'features_metric_function'):
            if self.features_metric_function == None:
                feature_score = sum(individual) / len(individual)
            else:
                feature_score = self.features_metric_function(individual)
        else:
            feature_score = sum(individual) / len(individual)

        self._all_solutions['individuals'].append(individual[:])
        self._all_solutions['results'].append( (accuracies.mean(), feature_score )  )

        return accuracies.mean(), feature_score

    def predict(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError('fit')

        if self.normalize_:
            X = self._sc_X.fit_transform(X)

        X_ = self.transform(X)
        y_pred = self._estimator.predict(X_)
        return self.classes_.take(np.asarray(y_pred, dtype=np.intp))

        #        elif self.predict_with == 'all':
        #
        #            predict_ = []
        #
        #            for mask in self.mask_:
        #                self.estimator.fit(X=self.transform(self.X_, mask=mask), y=self.y_)
        #                X_ = self.transform(X, mask=mask)
        #                y_pred = self.estimator.predict(X_)
        #                predict_.append(self.classes_.take(np.asarray(y_pred, dtype=np.intp)))
        #            return np.asarray(predict_)
    '''
    @staticmethod # is different, but I don't think that we use it anyway
    def score_func_to_gridsearch(estimator, X_test=None, y_test=None):
         Function to be given as a scorer function to Grid Search Method.
        It is going to transform the matrix os predicts generated by 'all' option
        to an final accuracy score. Use a high value to CV
        
        if not hasattr(estimator, 'fitnesses_'):
            raise ValueError("Fit")

        return sum([ i[0]-i[1] for i in estimator.fitnesses_]) / float(len(estimator.fitnesses_))
    '''

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 1:
            # print(y)
            raise ValueError("The number of classes has to be at least one;"
                             "got %d" % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')

    def fit_transform(self, X, y, normalize=False, **fit_params):
        """Fit to data, then transform it.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """

        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, normalize, **fit_params).transform(X)


    #@staticmethod
    #def _get_accuracy(ind):
    #    return ind.fitness.values[0]
    
    @staticmethod
    def _get_features(ind):
        return sum(ind)

    
    def __getstate__(self):

        self_dict = self.__dict__.copy()
        if '_toolbox' in self_dict:
            del self_dict['_toolbox']

        if 'print_fnc' in self_dict:
            del self_dict['print_fnc']

        if 'stats' in self_dict:
            del self_dict['stats']
        
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def _make_stats(self):
        stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(key=sum)
        self.stats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.stats.register("25_percentile", np.percentile, q=25, interpolation='higher')
        self.stats.register("50_percentile", np.percentile, q=50, interpolation='higher')
        self.stats.register("75_percentile", np.percentile, q=75, interpolation='higher')
        self.logbook = [tools.Logbook() for i in range(self.repeat)]

        for i in range(self.repeat):
            self.logbook[i].header = "gen", 'hallOfFame', "paretoFront", "time", "fitness", "size"

        for i in range(self.repeat):
            self.logbook[i].chapters["fitness"].header = self.stats.fields
            self.logbook[i].chapters["size"].header = self.stats.fields

    def _set_dataset(self, X, y, normalize):
        """  Standard datset pre-process:
        Using standardScaler()
        Validating/setting the correct size of X and y
        Validating the number of classes of y

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X : numpy array of shape [n_samples, n_features_new]
            Training set pre-processed

        y : numpy array of shape [n_samples]
            Training set pre-processed
        """

        if normalize:
            X = X.astype(np.float64, copy=False)
            self._sc_X = StandardScaler()
            X = np.asarray(X, dtype=np.float64)
            X = self._sc_X.fit_transform(X)
        self.normalize_ = normalize

        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64,
                         order='C', accept_sparse='csr')

        self.n_features_ = X.shape[1]

               

        return X, y

    def _set_fit(self):
        pass

    def _make_generation_log(self, gen, repetition, pop, hof, pareto_front):
        self.i_gen_pareto_.append(copy.deepcopy(pareto_front))
        self.i_gen_hof_.append(copy.deepcopy(hof))

        record = self.stats.compile(pop)

        self.logbook[repetition].record(gen=gen, hallOfFame=copy.deepcopy(hof), paretoFront=copy.deepcopy(pareto_front), time=time.clock(), **record)

        if self.verbose:
            self._toolbox.print("*********    Report {}       ************* ".format(self.name), end='\n\n')
            self._toolbox.print("\tmin\t25%\t50%\t75%\tmax")
            self._toolbox.print("Score:\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}".format( record['fitness']['min'], record['fitness']['25_percentile'],record['fitness']['50_percentile'],record['fitness']['75_percentile'], record['fitness']['max']))
            self._toolbox.print("Size:\t{:d}\t{:d}\t{:d}\t{:d}\t{:d}".format( record['size']['min'], record['size']['25_percentile'],record['size']['50_percentile'],record['size']['75_percentile'], record['size']['max']), end='\n\n')

            if( hasattr(self, "number_gen") and gen != 0 ):
                eta = (time.clock() - self._initial_time) * self.number_gen / (gen+1)
                self._toolbox.print("Gen: {}/{} Time Elapsed: {}min {}sec Estimated Time: {} min {} s".format(gen, self.number_gen, 
                    np.floor((time.clock()-self._initial_time)/60),(time.clock()-self._initial_time)%60, 
                    np.floor(eta/60), eta%60))

                
            self._toolbox.print()

    def _make_repetition_log(self, hof, pareto_front):
        self.best_.update(hof)
        self.best_pareto_front_.update(pareto_front)
        if self.make_logbook:
            self.pareto_front_.append( copy.deepcopy(pareto_front))
            self.hof_.append(copy.deepcopy(hof))
            self.gen_pareto_.append(self.i_gen_pareto_)
            self.gen_hof_.append(self.i_gen_hof_)
            self.i_gen_pareto_ = []
            self.i_gen_hof_ = []

    def _setup(self, X, y, normalize):
        " Initialize the toolbox and statistical variables"

        if(hasattr(self, '_all_solutions')):
            del self._all_solutions
        self._all_solutions = {'results':[], 'individuals':[]}

        if(not hasattr(self, "_toolbox")):
            self._toolbox = base.Toolbox()

        if hasattr(self, "print_fnc"):
            if (self.print_fnc == None):
                self._toolbox.register("print", print)
            else:
                self._toolbox.register("print", self.print_fnc)
        else:
            self._toolbox.register("print", print)

        if self.parallel:
            from multiprocessing import Pool
            self._toolbox.register("map", Pool().map)
        else:
            self._toolbox.register("map", map)

        if self.make_logbook:
            self._make_stats()
            self.pareto_front_ = []
            self.hof_ = []
            self.gen_hof_ = []
            self.gen_pareto_ = []
            self.i_gen_hof_ = []
            self.i_gen_pareto_ = []

        if self.estimator == None:
            self._estimator = SVC(kernel='linear', gamma="auto")
        else:
            self._estimator = self.estimator

        self._random_object = check_random_state(self.random_state)
        np.random.seed(self.random_state)

        self.best_ = tools.HallOfFame(1)
        self.best_pareto_front_ = tools.ParetoFront()

        self._toolbox.register('clone', copy.deepcopy)

        if hasattr(self, 'skip'):
            self._skip = self.skip
        else:
            self._skip = 0

        X, y = self._set_dataset(X=X, y=y, normalize=normalize)

                # Check the number of members in each class:
        min_n_classes =  min( Counter(y).values() )
        if( min_n_classes >= 5  ):
            self._toolbox.register("evaluate", self._evaluate, X=X, y=y, cv=5)
        else :
            self._toolbox.register("evaluate", self._evaluate, X=X, y=y, cv=2)
        
        return X, y

    def best_pareto(self):
        return self.best_pareto_front_

    def all_paretos(self):
        return self.pareto_front_

    def best_solution(self):
        return self.best_[0]

    def all_solutions(self):
        return self.hof_

    def set_params(self, **params):

        super().set_params(**params)

        return self

    def _do_generation(self, pop, hof, paretoFront):
        pass

    def fit(self, X, y, time_limit = None, normalize=False, **arg):
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
        

        X,y = self._setup(X, y, normalize)

        self.set_params(**arg)
        
        self._initial_time = time.clock()
        for i in range(self.repeat):

            pop = self._toolbox.population(n=self.size_pop)
            
            fitnesses = self._toolbox.map(self._toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses): 
                ind.fitness.values = fit            
            
            hof = tools.HallOfFame(1)
            hof.update(pop)
            pareto_front = tools.ParetoFront()
            pareto_front.update(pop)

            for g in range(self.number_gen):

                pop, hof, pareto_front = self._do_generation( pop, hof, pareto_front)

                if self._skip == 0 or g % self._skip == 0:
                    if self.make_logbook:
                        self._make_generation_log(g, i, pop, hof, pareto_front)

                if self.verbose and not self.make_logbook:
                    self._toolbox.print( "Generation: ", g, " Repetition: ", i, " Clock:", time.clock() - self._initial_time)

                if time_limit is not None and time.clock() - self._initial_time > time_limit:
                    break
                
            self._make_repetition_log(hof, pareto_front)

        self._estimator.fit(X=self.transform(X), y=y)

        return self
