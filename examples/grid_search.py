"""
============================
Grid Search Example
============================

An example of how to use Metaheuristics and GridSearch
"""
from feature_selection import HarmonicSearch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Classifier to be used in the metaheuristic
clf = SVC()
clf = RandomForestClassifier()
clf.fit(X,y)
clf.predict(X) == y
# Parameter Grid
param_grid= {
    "HMCR":[0, 0.5, 0.95],
    "indpb":[0.05, 0.5, 1],
    "pitch":[0.05, 0.5, 1],
    "repeat":[3]
     }
hs = HarmonicSearch(classifier=clf, make_logbook=True)
grid_search = GridSearchCV(hs, param_grid=param_grid, scoring=hs.score_func_to_gridsearch, cv=4,
                           verbose=2)
grid_search.fit(X,y)

grid_search.best_params_
results = pd.DataFrame.from_dict(grid_search.cv_results_)
