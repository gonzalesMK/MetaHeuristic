"""
============================
Plotting MetaHeuristics - Basic Use
============================
"""
from feature_selection import HarmonicSearch, GeneticAlgorithm
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])

# Classifier to be used in the metaheuristic
clf = SVC()

hs = HarmonicSearch(classifier = clf,
                    random_state = 0,
                    make_logbook = True,
                    repeat_ = 2)

ga = GeneticAlgorithm(
                    classifier = clf,
                    random_state = 1,
                    make_logbook = True,
                    repeat_ = 2)

# Fit the classifier
hs.fit(X, y, normalize= True)
ga.fit(X, y, normalize= True)

print("Number of Features Selected: \n \t HS: ", sum(hs.support_)/X.shape[1],"% \t GA: ", sum(ga.support_)/X.shape[1],"%")
print("Accuracy of the classifier: \n \t HS: ", hs.best_fitness[0],"\t GA: ", ga.best_fitness[0])

# Transformed dataset
X_hs = hs.transform(X)
X_ga = ga.transform(X)

# Plot the results of each test
hs.plot_results()
ga.plot_results()