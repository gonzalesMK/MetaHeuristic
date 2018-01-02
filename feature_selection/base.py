from abc import ABCMeta
from warnings import warn
from itertools import compress
from random import sample
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.model_selection import cross_val_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
from sklearn.utils.validation import check_array, check_is_fitted, column_or_1d
from sklearn.externals import six
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils import check_random_state
from sklearn.svm import  SVC
from deap import  base
from deap import tools
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_X_y

class Fitness(base.Fitness):

    def __init__(self, weights=(1,-1), values=(0,0)):
        self.weights = weights
        super(Fitness, self).__init__(values)

class BaseMask(list, object):

    def __init__(self, mask):
        self[:] = mask
        self.fitness = Fitness((1, -1), (0, 0))

#    def __getstate__(self):
#        self_dict = self.__dict__.copy()
#        return self_dict
#
#    def __setstate__(self,state):
#        self.__dict__.update(state)
#


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

        if np.issubdtype(mask.dtype, np.int) or np.issubdtype(mask.dtype, np.bool):
            if x.shape[1] != len(mask):
                raise ValueError("X columns %d != mask length %d"
                                 % (x.shape[1], len(mask)))

    # I don't see utility in here
#        if hasattr(x, "toarray"):
#            ind = np.arange(mask.shape[0])
#            mask = ind[mask]
#
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
        return mask if not indices else np.where(mask)[0]

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


class _BaseMetaHeuristic(BaseEstimator, SelectorMixin, ClassifierMixin):

    def __init__(self, name,classifier=None, number_gen=20,
                 verbose=0, repeat=1, parallel=False,
                 make_logbook=False, random_state=None,
                 cv_metric_fuction=make_scorer(matthews_corrcoef),
                 features_metric_function=None,print_fnc = None):

        self._name = name
        self.estimator = SVC(kernel='linear', max_iter=10000) if classifier is None else clone(classifier)
        self.number_gen = number_gen
        self.verbose = verbose
        self.repeat = repeat
        self.parallel=parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function= cv_metric_fuction
        self.features_metric_function= features_metric_function
        self._random_object = check_random_state(self.random_state)
        self.print_fnc =  print_fnc               
        random.seed(self.random_state)
        
        self.toolbox = base.Toolbox()
        if print_fnc == None:
            self.toolbox.register("print", print)
        else:
            self.toolbox.register("print", print_fnc)

    def _gen_in(self):
        """ Generate a individual, DEAP function

        """
        random_number = self._random_object.randint(1, self.n_features_ + 1)
        zeros = (np.zeros([self.n_features_-random_number,], dtype=int))
        ones = np.ones([random_number,], dtype=int)
        return   sample(list(np.concatenate((zeros, ones), axis=0)), self.n_features_)

    def _evaluate(self, individual, X, y, cv=3):
        """ Evaluate method

        Parameters
        ----------
        individual: list [n_features]
                The input individual to be evaluated

        Return
        ----------
        Score of the individual : turple( cross_val_score, feature score)
        """
        # Select Features
        features = list(compress(range(len(individual)), individual))
        train = np.reshape([X[:, i] for i in features],
                           [len(features), len(X)]).T

        if train.shape[1] == 0:
            return 0,1,

        # Applying K-Fold Cross Validation
        accuracies = cross_val_score(estimator=clone(self.estimator), X=train,
                                     y=y, cv=cv,
                                     scoring=self.cv_metric_function)

        if self.features_metric_function == None :
            feature_score = pow(float(sum(individual))/(len(individual)*5), 2)
        else:
            feature_score = self.features_metric_function(individual)

        return accuracies.mean() - accuracies.std(), feature_score

    def predict(self, X):
        if not hasattr(self, "classes_"):
            raise ValueError('fit')

        if self.normalize_:
            X = self._sc_X.fit_transform(X)

        X_ = self.transform(X)
        y_pred = self.estimator.predict(X_)
        return   self.classes_.take(np.asarray(y_pred, dtype=np.intp))

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

    @staticmethod
    def score_func_to_gridsearch(estimator, X_test=None, y_test=None):
        """ Function to be given as a scorer function to Grid Search Method.
        It is going to transform the matrix os predicts generated by 'all' option
        to an final accuracy score. Use a high value to CV
        """
        if not hasattr(estimator, 'fitnesses_'):
            raise ValueError("Fit")

        return sum([ i[0]-i[1] for i in estimator.fitnesses_]) / float(len(estimator.fitnesses_))

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError("The number of classes has to be greater than one;"
                             "got %d" % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')

    def plot_results(self):
        """ This method plots all the statistics for each repetition
        in a graph.
            The curves are minimun, average and maximun accuracy
        """
        if not self.make_logbook:
            raise ValueError("You need to set make_logbook to true")

        for i in range(self.repeat):
            gen = self.logbook[i].select("gen")
            acc_mins = self.logbook[i].select("min")
            acc_maxs = self.logbook[i].select("max")
            acc_avgs = self.logbook[i].select("avg")

            _, ax1 = plt.subplots()
            line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Acc")
            line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Acc")
            line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Acc")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Accuracy")

            lns = line1 + line2 + line3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc="center right")
            ax1.set_title(self._name +" Repetition: " + str(i+1))


    def fit_transform(self, X, y, normalize = False, **fit_params):
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

    @staticmethod
    def _get_accuracy(ind):
        return ind.fitness.wvalues[0]

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        if 'toolbox' in self_dict:
	        del self_dict['toolbox']
        if 'print_fnc' in self_dict:
	        del self_dict['print_fnc']

        return self_dict

    def __setstate__(self,state):
        self.__dict__.update(state)

    def _make_stats(self):
        self.stats = tools.Statistics(self._get_accuracy)
        self.stats.register("avg", np.mean)
        self.stats.register("std", np.std)
        self.stats.register("min", np.min)
        self.stats.register("max", np.max)
        self.logbook = [tools.Logbook() for i in range(self.repeat)]
        for i in range(self.repeat):
            self.logbook[i].header = ["gen"] + self.stats.fields

    def _set_dataset(self, X, y, normalize):
        if normalize:
            self._sc_X = StandardScaler()
            X = self._sc_X.fit_transform(X)
        self.normalize_ = normalize

        y = self._validate_targets(y)
        X, y = check_X_y(X, y, dtype=np.float64, order='C', accept_sparse='csr')

        self.n_features_ = X.shape[1]

        self.toolbox.register("evaluate", self._evaluate, X=X, y=y)

        return X,y

    def _set_fit(self):
        if self.make_logbook:
            self._make_stats()
            self.pareto_front_ = []
            self.hof_ = []
            self.gen_hof_ = []
            self.gen_pareto_ = []
            self.i_gen_hof_ = []
            self.i_gen_pareto_ = []

        self._random_object = check_random_state(self.random_state)
        random.seed(self.random_state)

        self.best_ = tools.HallOfFame(1)
        self.best_pareto_front_ = tools.ParetoFront()

    def _make_generation(self, hof, pareto_front):
            self.i_gen_pareto_.append(pareto_front[:])
            self.i_gen_hof_.append(hof[0])

    def _make_repetition(self, hof, pareto_front):
        self.best_.update(hof)
        self.best_pareto_front_.update(pareto_front)
        if self.make_logbook:
            self.pareto_front_.append(pareto_front[:])
            self.hof_.append(hof[0])
            self.gen_pareto_.append(self.i_gen_pareto_)
            self.gen_hof_.append(self.i_gen_hof_)
            self.i_gen_pareto_=[]
            self.i_gen_hof_=[]

    def best_pareto(self):
        return self.best_pareto_front_

    def all_paretos(self):
        return self.pareto_front_

    def best_solution(self):
        return self.best_[0]

    def all_solutions(self):
        return self.hof_
    
    def _print(self,gen, rep, initial_time, final_time):
        self.toolbox.print("""Repetition: {:d} \t Generation: {:d}/{:d} 
                Elapsed time: {:.4f} \r""".format( rep+1,gen + 1,
                self.number_gen,final_time - initial_time))