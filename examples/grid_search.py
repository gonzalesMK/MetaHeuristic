"""
============================
Plotting MetaHeuristics - Basic Use
============================

An example plot of :class:`feature_selection.HarmonicSearch
"""
from feature_selection import HarmonicSearch, GeneticAlgorithm
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from six.moves import cPickle
import pickle

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])

_sc_X = StandardScaler()
X = _sc_X.fit_transform(X)
            
param_grid = [
  {
   'HMCR': [0, 0.5, 0.9, 0.95, 1],
   'indpb':[0,0.05,0.1,0.25,0.5], 
   'pitch': [0, 0.05, 0.1, 0.25, 0.5],
   'number_gen':(150,200), 
   'mem_size':(100,50),
   'predict_with':['all'], 
   'random_state':[0], 
   'repeat':[10]
   }  
 ]

param_grid = [
  {
   'HMCR': [0.95, 1],  'indpb':[0.05,1], 'number_gen':(50,100),
   'predict_with':('all',), 'random_state':[0], 'repeat':[2]
   }  
 ]

hs = HarmonicSearch()


clf = GridSearchCV(hs, param_grid, scoring=hs.score_func_to_grid_search, verbose=1,pre_dispatch=None)
clf.fit(X, y)
clf.cv_results_
clf.best_score_
clf.estimator.best_mask_
hs.fit(X,y)

cPickle.dump(clf, open('saved.p', 'wb'))
c = cPickle.load(open('save.p', 'rb'))
print(c.cv_results_)
