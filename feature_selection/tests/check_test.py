import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from feature_selection import HarmonicSearch, GeneticAlgorithm, RandomSearch, BinaryBlackHole
from sklearn.utils.testing import assert_raises


def test_ga():
    check_estimator(GeneticAlgorithm)        
    
def test_hs():
    check_estimator(HarmonicSearch)   

def test_rdm():
    check_estimator(RandomSearch)

def test_bbha():
    check_estimator(BinaryBlackHole)
    
def test_plot():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()

    for metaclass in [HarmonicSearch, GeneticAlgorithm, RandomSearch, BinaryBlackHole]:
        meta = metaclass(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=10)
        
        print("Checking Class: ", meta._name)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=10)
        
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)

        assert_array_equal(X_1, X_2)
    
        # Plot the results of each test
        meta.plot_results()
        
    ga = GeneticAlgorithm(classifier=clf, random_state=1,
                          make_logbook=False, repeat=1)
    
    # check for error in plot
    ga.fit(X, y, normalize=True)
    assert_raises(ValueError, ga.plot_results)
    
def test_all_prediction():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    for metaclass in [HarmonicSearch, GeneticAlgorithm, RandomSearch, BinaryBlackHole]:
        meta = metaclass(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, predict_with='all')
        print("Checking Class: ", meta._name)
    # Checks for error
        assert_raises(ValueError, meta.score_func_to_grid_search, meta, X, y)
    
        # Fit the classifier
        meta.fit(X, y, normalize=True)
    
        # Transformed dataset
        X_1 = meta.transform(X)
    
        meta = metaclass(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=2, predict_with='all')
    
        # Fit and Transform
        X_2 = meta.fit_transform(X=X, y=y, normalize=True)
    
        # Check Function
        meta.score_func_to_grid_search(meta, X, y)
        assert_array_equal(X_1, X_2)

def test_unusual_errors():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=10)
    
    hs.fit(X, y, normalize=True)
    # Let's suppose you have a empty array 
    hs.support_ = np.array([])
    hs.transform(X)