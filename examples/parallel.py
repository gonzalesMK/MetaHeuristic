"""
============================
Parallel Example
============================

An example plot of :class:`feature_selection.HarmonicSearch
"""
from feature_selection import BRKGA
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

# It is very necessary to include if __name__ == "__main__"
if __name__ == "__main__":
    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
    # Classifier to be used in the metaheuristic
    clf = SVC()
    
    print("Starting Algorithm in Parallel")
    ga =BRKGA(classifier=clf, make_logbook=True, repeat=2, parallel=True,
                          verbose=True, size_pop=100)
    
    # Fit the classifier
    ga.fit(X, y, normalize=True)
    print("Number of Features Selected: \n \t HS: " , sum(ga.best_solution())/X.shape[1], "%")
    print("Accuracy of the classifier: \n \t HS: ", ga.best_solution().fitness.values[0])
    
    # Plot the results of each test
    ga.plot_results()

    print("Starting Algorithm")
    ga =BRKGA(classifier=clf, make_logbook=True, repeat=2, parallel=False,
                          verbose=True, size_pop=100)
    
    # Fit the classifier
    ga.fit(X, y, normalize=True)
    print("Number of Features Selected: \n \t HS: " , sum(ga.best_solution())/X.shape[1], "%")
    print("Accuracy of the classifier: \n \t HS: ", ga.best_solution().fitness.values[0])
    
    # Plot the results of each test
    ga.plot_results()
