import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone

from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted,column_or_1d
from sklearn.externals import six
from sklearn.utils.multiclass import check_classification_targets
from abc import ABCMeta
from warnings import warn
from itertools import compress

from random import sample

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
        check_is_fitted(self, 'support_')
        return self.support_

    
    def transform(self, X,mask = None):
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
        
        if mask == None:
            mask = self.get_support()
            
        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((X.shape[0], 0))
        
        if len(mask) != X.shape[1]:
            raise ValueError("X has a different shape than during fitting.")
            
        return X[:, safe_mask(X, mask)]

    
class _BaseMetaHeuristic(BaseEstimator,SelectorMixin, ClassifierMixin):
    
    def _gen_in(self):
        """ Generate a individual, DEAP function

        """
        RND = self.random_object.randint(0, self.n_features)
        return   sample(list(np.concatenate( (np.zeros([self.n_features-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), self.n_features)

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
        
        return accuracies.mean() - accuracies.std(), pow(sum(individual)/(X.shape[1]*5),2),

    
    def predict(self,X):
        
        if( self.predict_with == 'best'):
            X_ = self.transform(X)
            y_pred = self.estimator.predict(X_)
            
            return   self.classes_.take(np.asarray(y_pred, dtype=np.intp))
        
        elif(self.predict_with == 'all'):
            predict_ = []
            for mask in self.mask:
                X_ = self.transform(X, mask = mask)
                self.estimator.fit(X = self.transform(self.X, mask = mask), y = self.y)                    
                y_pred = self.estimator.predict(self.transform(X, mask = mask))
                predict_.append(self.classes_.take(np.asarray(y_pred, dtype=np.intp)))
                     
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

    def _validate_targets(self, y):
        y_ = column_or_1d(y, warn=True)
        check_classification_targets(y)
        cls, y = np.unique(y_, return_inverse=True)
        if len(cls) < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d"
                % len(cls))

        self.classes_ = cls

        return np.asarray(y, dtype=np.float64, order='C')
    
    def plot_results(self):
        
        for i in range(self.repeat_):
            gen = self.logbook[i].select("gen")
            acc_mins = self.logbook[i].select("min")
            acc_maxs = self.logbook[i].select("max")
            acc_avgs = self.logbook[i].select("avg")
                    
            fig, ax1 = plt.subplots()
            line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Acc")
            line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Acc")
            line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Acc")
            ax1.set_xlabel("Generation")
            ax1.set_ylabel("Accuracy")

            lns = line1 + line2 + line3
            labs = [l.get_label() for l in lns]
            ax1.legend(lns, labs, loc="center right")
            ax1.set_title("Repetition: " + str(i+1) )
            plt.show()