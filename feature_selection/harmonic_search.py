from __future__ import print_function
from timeit import time
from deap import base
from deap import tools
from .meta_base import *
from .meta_base import _BaseMetaHeuristic
from sklearn.svm import SVC
import random
from sklearn.base import clone
from multiprocessing import Pool


class HarmonicSearch(_BaseMetaHeuristic):
    """Implementation of a Harmonic Search Algorithm for Feature Selection

    Parameters
    ----------
    estimator : sklearn estimator , (default=SVM)
            Any estimator that adheres to the scikit-learn API

    number_gen : positive integer, (default=100)
            Number of generations
 
    HMCR : float in [0,1], (default=0.95)
            Is the Harmonic Memory Considering Rate

    size_pop : positive integer, (default=50)
            Size of the Harmonic Memory 
    
    sorting_method: one of {'best', 'NSGA2', 'SPEA2'}, (default='NSGA2')
             How to sort the population in order to choose the Elite solutions

             - If 'best', then sort by the score parameter only
             - If 'NSGA2', use the NSGA2 sorting mechanism [2]
             - If 'SPEA2', use the SPEA2 sorting mechanism [3]

    skip: positive integer, (default=10)
            This parameter is important when ``make_logbook`` is True. Usually, a lot
            of generations are needed by HS to achieve convergence, given that 
            one solution is evaluated at a time. Consequently, the amount of data 
            saved by the ``logbook`` can be bigger than necessary. So, one can
            choose to log data only every ``skip``iteration

    verbose : boolean, (default=False)
            If true, print information in every generation

    repeat : positive int, (default=1)
            Number of times to repeat the fitting process

    parallel : boolean, (default=False)
            Set to True if you want to use multiprocessors

    make_logbook : boolean, (default=False)
            If True, a logbook from DEAP will be made

    cv_metric_function : callable, (default=matthews_corrcoef)
            A metric score function as stated in the sklearn http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    features_metric_function : callable, (default=pow(sum(mask)/(len(mask)*5), 2))
            A function that return a float from the binary mask of features
    
    
    References
    ----------
    .. [1] X. Z. Gao, V. Govindasamy, H. Xu, X. Wang, and K. Zenger, 
            “Harmony Search Method: Theory and Applications",
            Computational Intelligence and Neuroscience, vol. 2015,
    
    .. [2] Deb, Pratab, Agarwal, and Meyarivan, “A fast elitist non-dominated sorting 
	       genetic algorithm for multi-objective optimization: NSGA-II”, 2002.

	.. [3] Zitzler, Laumanns and Thiele, “SPEA 2: Improving the strength Pareto evolutionary algorithm”, 2001.
    """
    def __init__(self, 
                 estimator=None, 
                 HMCR=0.95,
                 number_gen=100, 
                 size_pop=50, 
                 sorting_method='NSGA2',
                 skip=10,
                 verbose=0, 
                 repeat=1,
                 make_logbook=False, 
                 random_state=None, 
                 parallel=False,
                 cv_metric_function=None):

        self.estimator = estimator
        self.number_gen = number_gen
        self.HMCR = HMCR
        self.size_pop = size_pop
        self.sorting_method = sorting_method
        self.skip = skip
        self.verbose = verbose
        self.repeat = repeat
        self.parallel = parallel
        self.make_logbook = make_logbook
        self.random_state = random_state
        self.cv_metric_function = cv_metric_function
        np.random.seed(self.random_state)

    def _setup(self, X, y, normalize):
        X, y = super()._setup(X,y,normalize)

        self._toolbox.register("attribute", self._gen_in)

        self._toolbox.register("individual", tools.initIterate,
                               BaseMask, self._toolbox.attribute)

        self._toolbox.register("population", tools.initRepeat,
                               list, self._toolbox.individual)

        if(self.sorting_method == 'best'):
            self._toolbox.register( "sort", sorted, key=lambda ind: ind.fitness.values[0], reverse=True)
        elif(self.sorting_method == 'NSGA2'):
            self._toolbox.register( "sort", tools.selNSGA2, k=self.size_pop+1)
        elif(self.sorting_method == 'SPEA2'):
            self._toolbox.register( "sort", tools.selSPEA2, k=self.size_pop+1)
        else:
            raise ValueError("The {} sorting method is not valid".format(self.sorting_method))

        

        if( self.HMCR < 0 or self.HMCR > 1):
            raise ValueError("The HMCR param is {}, but should be in the interval [0,1]".format(self.HMCR))
        
        self._toolbox.register("mutate", tools.mutFlipBit, indpb=1-self.HMCR)

        return X, y
    def _do_generation(self, harmony_mem, hof, paretoFront):
                
        # Improvise a New Harmony
        new_harmony = self._improvise(harmony_mem)
        new_harmony.fitness.values = self._toolbox.evaluate(new_harmony)
        harmony_mem.append(new_harmony)

        # Remove the Worst Harmony
        harmony_mem = self._toolbox.sort(harmony_mem)
        harmony_mem.pop()

        # Log statistic
        hof.update(harmony_mem)
        paretoFront.update(harmony_mem)

        return harmony_mem, hof, paretoFront

    def _improvise(self, pop):
        """ Function that improvise a new harmony"""
        # HMCR = Harmonic Memory Considering Rate
        # pylint: disable=E1101
        new_harmony = self._toolbox.individual()

        rand_list = self._random_object.randint(low=0, high=len(pop),
                                                size=len(new_harmony))

        for i in range(len(new_harmony)):
            new_harmony[i] = pop[rand_list[i]][i]

        self._toolbox.mutate(new_harmony)

        return new_harmony
