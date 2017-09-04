from sklearn.utils.estimator_checks import check_estimator
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
    
    hs = HarmonicSearch(classifier=clf, random_state=0, verbose = 10,
                        make_logbook=True, repeat=2)
    
    ga = GeneticAlgorithm(classifier=clf, random_state=1, verbose = 10,
                          make_logbook=True, repeat=2)
    
    # Fit the classifier
    hs.fit(X, y, normalize=True)
    ga.fit(X, y, normalize=True)
    
    # Fit and Transform
    
    print("Number of Features Selected: \n \t HS: ", sum(hs.support_)/X.shape[1],
          "% \t GA: ", sum(ga.support_)/X.shape[1], "%")
    
    print("Accuracy of the classifier: \n \t HS: ", hs.fitness_[0], "\t GA: ",
          ga.fitness_[0])

    # Transformed dataset
    X_hs = hs.transform(X)
    X_ga = ga.transform(X)
    
    # Fit and Transform
    X_hs = hs.fit_transform(X=X, y=y)
    X_ga = ga.fit_transform(X=X, y=y)
    
    # Plot the results of each test
    hs.plot_results()
    ga.plot_results()
    
    