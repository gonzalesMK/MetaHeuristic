"""
============================
Comparing MetaHeuristics 
============================

An example comparing:
 :class:`feature_selection.HarmonicSearch
 :class:`feature_selection.SimulatedAnneling
"""
# Applying consecutive algorithms is a good business
import numpy as np
from feature_selection import HarmonicSearch, SimulatedAnneling
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])

_sc_X = StandardScaler()
X = _sc_X.fit_transform(X)

hs = HarmonicSearch(classifier=SVC())
hs.fit(X,y)

sa = SimulatedAnneling(classifier=SVC(),initial_temp=10, number_gen = 50, repetition_schedule=3)
sa.fit(X,y)

sv = SVC()
sv.fit(X,y)

print( "Simulated Anneling #features: " ,sum(sa.best_mask_))
print( "Accuracy: ", sa.fitness_[0], " Fitness:", sum(sa.fitness_))

print( "Harmonic Search #features: " ,sum(hs.best_mask_))
print( "Accuracy: ", hs.fitness_[0], " Fitness:", sum(hs.fitness_))    

print( "Standart Accuracy: ", sv.score(X,y))    

mask = np.array([sa.best_mask_[i] and hs.best_mask_[i] for i in range(len(sa.best_mask_))])
print( "Number of Features that were equally selected: ", sum(mask))
      
hs.transform(X, mask)

X_ = sa.transform(X)
sv = SVC()
sv.fit(X_,y)
print("Final Accuracy: ", sv.score(X_,y))