import numpy as np
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.testing import assert_array_equal
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from feature_selection import HarmonicSearch, GeneticAlgorithm

def test_ga():
    check_estimator(GeneticAlgorithm)        
    
def test_hs():
    check_estimator(HarmonicSearch)   

def test_plot():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=100)
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose=10,
                          make_logbook=True, repeat=1)
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)
    
    # Fit and Transform
    
    print("Number of Features Selected: \n \t HS: ", sum(hs.support_)/X.shape[1],
          "% \t GA: ", sum(ga.support_)/X.shape[1], "%")
    
    print("Accuracy of the classifier: \n \t HS: ", hs.fitness_[0], "\t GA: ",
          ga.fitness_[0])

    # Transformed dataset
    X_hs1 = hs.transform(X)
    X_ga1 = ga.transform(X)
    
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=100)
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose = 10,
                          make_logbook=True, repeat=1)
    
    # Fit and Transform
    X_hs2 = hs.fit_transform(X=X, y=y, normalize=True)
    X_ga2 = ga.fit_transform(X=X, y=y, normalize=True)
    
    # Plot the results of each test
    hs.plot_results()
    ga.plot_results()
    
    # Assert equal
    assert_array_equal(X_hs2, X_hs1)
    assert_array_equal(X_ga2, X_ga1)
    
def test_all_prediction():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=2, number_gen=100,
                        predict_with='all')
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose=10,
                          make_logbook=True, repeat=2, predict_with='all')
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)

    # Check if output is float
    if type(hs.score_func_to_grid_search(hs, X, y)) != np.float64:
        raise ValueError("It is not float")
        
    if type(ga.score_func_to_grid_search(ga, X, y)) != np.float64:
        raise ValueError("It is not float")

    # Transformed dataset
    X_hs1 = hs.transform(X)
    X_ga1 = ga.transform(X)

    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=2, number_gen=100,
                        predict_with='all')
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose=10,
                          make_logbook=True, repeat=2, predict_with='all')

    # Fit and Transform
    X_hs2 = hs.fit_transform(X=X, y=y, normalize=True)
    X_ga2 = ga.fit_transform(X=X, y=y, normalize=True)

    assert_array_equal(X_hs2, X_hs1)
    assert_array_equal(X_ga2, X_ga1)
    
def test_error():
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose=50,
                        make_logbook=True, repeat=1, number_gen=100)
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose=10,
                          make_logbook=True, repeat=1)
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)
    