import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from feature_selection import HarmonicSearch
from feature_selection import GeneticAlgorithm
from feature_selection import RandomSearch
from feature_selection import BinaryBlackHole
from feature_selection import SimulatedAnneling
from feature_selection import BRKGA
from feature_selection import SPEA2
from feature_selection import PSO
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns
import nose.plugins.multiprocess 
# Those are nose tests: to run it, write: python -m nose

_multiprocess_can_split_ = True

METACLASSES = [
        SimulatedAnneling, PSO, HarmonicSearch, GeneticAlgorithm, RandomSearch,
        BinaryBlackHole,  BRKGA,
        SPEA2]

def test_check_estimator():
    for metaclass in METACLASSES:
        print("check_estimator: ", metaclass(estimator=SVC()).name)
        check_estimator(metaclass)        
    
    
def test_overall():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC(gamma='auto')

    for metaclass in METACLASSES:
        meta = metaclass(estimator=clf, random_state=0, verbose=False,
                        make_logbook=True, repeat=1, number_gen=2,
                        size_pop=2)
        
        print("Checking: ", meta.name)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(estimator=clf, random_state=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
        
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)

        assert_array_equal(X_1, X_2)
        meta.best_pareto()
        meta.all_paretos()
        meta.best_solution()
        meta.all_solutions()
    
def test_parallel():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC(gamma="auto")

    for metaclass in METACLASSES :
        meta = metaclass(estimator=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, parallel=True, verbose=True,
                        size_pop=2)
        print("Checking parallel ", meta.name)
        
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(estimator=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, parallel=True, size_pop=2)
    
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)
    
        # Check Function
        assert_array_equal(X_1, X_2)
  
def test_unusual_errors():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC(gamma='auto')
    
    for metaclass in METACLASSES:
        meta = metaclass(estimator=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
        print("Checking unusual error: ", meta.name)
        meta.fit(X, y, normalize=True)
    
        # Let's suppose you have a empty best 
        assert_raises(ValueError, meta.safe_mask, X, [])

    meta = metaclass(estimator=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
    
    #assert_raises(ValueError, meta.score_func_to_gridsearch, meta)
    
    for metaclass in [BRKGA]:
        meta = metaclass(estimator=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2,
                        elite_size=5)
        assert_raises(ValueError, meta.fit, [ [1, 1, 1], [1,2,3] ], [1, 0])
            
def test_predict():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    sa = SimulatedAnneling(size_pop=2, number_gen=2)
    sa.fit(X,y, normalize=True)
    sa.predict(X)



"""
def test_score_grid_func():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()

    for metaclass in METACLASSES:
        meta = metaclass(classifier=clf, random_state=0, verbose=True,
                        make_logbook=True, repeat=1, number_gen=3,
                        size_pop=2)
        
        print("Checking Grid: ", meta.name)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # See score 
        meta.score_func_to_gridsearch(meta)
""" 