import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from feature_selection import HarmonicSearch, HarmonicSearch2
from feature_selection import GeneticAlgorithm
from feature_selection import RandomSearch
from feature_selection import BinaryBlackHole
from feature_selection import SimulatedAnneling
from feature_selection import BRKGA, BRKGA2
from feature_selection import SPEA2
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns

METACLASSES = [HarmonicSearch, GeneticAlgorithm, RandomSearch,
               BinaryBlackHole, SimulatedAnneling, BRKGA, BRKGA2, 
               HarmonicSearch2, SPEA2]

NSGA2 = [BRKGA2, HarmonicSearch2, SPEA2]

def test_check_estimator():
    for metaclass in METACLASSES:
        print("check_estimator: ", metaclass()._name)
        check_estimator(metaclass)        
    
def test_plot():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()

    for metaclass in METACLASSES:
        meta = metaclass(classifier=clf, random_state=0, verbose=False,
                        make_logbook=True, repeat=1, number_gen=2,
                        size_pop=2)
        
        print("Checking plotting: ", meta._name)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(classifier=clf, random_state=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
        
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)

        assert_array_equal(X_1, X_2)
        meta.best_pareto()
        meta.all_paretos()
        meta.best_solution()
        meta.all_solutions()
    
        # Plot the results of each test
        meta.plot_results()
        
    ga = GeneticAlgorithm(classifier=clf, random_state=1,
                          make_logbook=False, repeat=1)
    
    # check for error in plot
    ga.fit(X, y, normalize=True)
    assert_raises(ValueError, ga.plot_results)
    
def test_parallel():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()

    for metaclass in METACLASSES :
        meta = metaclass(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, parallel=True, verbose=True,
                        size_pop=2)
        print("Checking parallel ", meta._name)
        
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, parallel=True, size_pop=2)
    
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)
    
        # Check Function
        assert_array_equal(X_1, X_2)

def test_score_grid_func():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()

    for metaclass in METACLASSES:
        meta = metaclass(classifier=clf, random_state=0, verbose=True,
                        make_logbook=True, repeat=1, number_gen=3,
                        size_pop=2)
        
        print("Checking Grid: ", meta._name)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # See score 
        meta.score_func_to_gridsearch(meta)
    
def test_unusual_errors():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    for metaclass in METACLASSES:
        meta = metaclass(classifier=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
        print("Checking unusual error: ", meta._name)
        meta.fit(X, y, normalize=True)
    
        # Let's suppose you have a empty best 
        assert_raises(ValueError, meta.safe_mask, X, [])

    meta = metaclass(classifier=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2)
    
    assert_raises(ValueError, meta.score_func_to_gridsearch, meta)
    
    for metaclass in [BRKGA, BRKGA2]:
            assert_raises(ValueError, metaclass,classifier=clf, random_state=0, verbose=0,
                        make_logbook=True, repeat=1, number_gen=2, size_pop=2,
                        elite_size=5)
            
def test_nsga2():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])

    for metaclass in NSGA2:
        meta = metaclass(random_state=0, verbose=0, make_logbook=True, repeat=1, 
                    number_gen=2, size_pop=2, features_metric_function='poly')
        print("Checking NSGA2: ", meta._name)
        meta.fit(X, y, normalize=True)
        meta = metaclass(random_state=0, verbose=0, make_logbook=True, repeat=1, 
                    number_gen=2, size_pop=2, features_metric_function='uou')
        assert_raises(ValueError, meta.fit, X,y)

def test_predict():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    sa = SimulatedAnneling(size_pop=2, number_gen=2)
    sa.fit(X,y, normalize=True)
    sa.predict(X)