import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from feature_selection import HarmonicSearch, GeneticAlgorithm, RandomSearch
from sklearn.utils.testing import assert_raises

def test_ga():
    check_estimator(GeneticAlgorithm)        
    
def test_hs():
    check_estimator(HarmonicSearch)   

def test_rdm():
    check_estimator(RandomSearch)

def test_plot():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=10)
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose=10,
                          make_logbook=True, repeat=1)
            
    rd = RandomSearch(classifier=clf, random_state=1, verbose=10, number_gen=1,
                      make_logbook=True, repeat=1)
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)
    rd.fit(X, y, normalize=True)
    
    # Transformed dataset
    X_hs1 = hs.transform(X)
    X_ga1 = ga.transform(X)
    X_rd1 = rd.transform(X)
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=10)
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose = 10,
                          make_logbook=True, repeat=1)
    rd = RandomSearch(classifier=clf, random_state=1, verbose=10, number_gen=1,
                      make_logbook=True, repeat=1)
    
    # Fit and Transform
    X_hs2 = hs.fit_transform(X=X, y=y, normalize=True)
    X_ga2 = ga.fit_transform(X=X, y=y, normalize=True)
    X_rd2 = rd.fit_transform(X=X, y=y, normalize=True)
    
    assert_array_equal(X_hs1, X_hs2)
    assert_array_equal(X_ga1, X_ga2)
    assert_array_equal(X_rd1, X_rd2)
    
    # Plot the results of each test
    hs.plot_results()
    ga.plot_results()
    rd.plot_results()
    
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
    
    hs = HarmonicSearch(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=10, predict_with='all')
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, make_logbook=False,
                          repeat=2, predict_with='all')
    
    rd = RandomSearch(classifier=clf, random_state=1, make_logbook=False, 
                      repeat=2, number_gen=1, predict_with='all')

    # Checks for error
    assert_raises(ValueError, ga.score_func_to_grid_search, ga, X, y)
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)
    rd.fit(X, y, normalize=True)
    
    # Transformed dataset
    X_hs1 = hs.transform(X)
    X_ga1 = ga.transform(X)
    X_rd1 = rd.transform(X)
    
    hs = HarmonicSearch(classifier=clf, random_state=0, make_logbook=False,
                        repeat=2, number_gen=10, predict_with='all')
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, make_logbook=False,
                          repeat=2, predict_with='all')
    
    rd = RandomSearch(classifier=clf, random_state=1, make_logbook=False, 
                      repeat=2, number_gen=1, predict_with='all')

    # Fit and Transform
    X_hs2 = hs.fit_transform(X=X, y=y, normalize=True)
    X_ga2 = ga.fit_transform(X=X, y=y, normalize=True)
    X_rd2 = rd.fit_transform(X=X, y=y, normalize=True)
    
    assert_array_equal(X_hs2, X_hs1)
    assert_array_equal(X_ga2, X_ga1)
    assert_array_equal(X_rd2, X_rd1)

    