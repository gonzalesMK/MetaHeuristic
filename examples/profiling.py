# -*- coding: utf-8 -*-
# Commmand "python -m cProfile -o profile_output script.py"
from feature_selection import BRKGA2
from sklearn.datasets import load_breast_cancer
import cProfile

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])

# Classifier to be used in the metaheuristic
print("Starting Algorithm")
ga =BRKGA2(make_logbook=True, repeat=2, parallel=True,
           verbose=True,
       size_pop=100, elite_size=30, mutant_size=10)

cProfile.run('ga.fit(X,y)', 'restats')

import pstats
p = pstats.Stats('restats')
p.sort_stats('cumulative').print_stats(50)


