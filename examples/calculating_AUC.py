"""
============================
How to Compare NSGA2 Results - AUC
============================

An example plot of :class:`feature_selection.HarmonicSearch
"""

from feature_selection import BRKGA2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from sklearn.metrics import auc
from sklearn.datasets import load_breast_cancer

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
sc_X = StandardScaler()
X = sc_X.fit_transform(X)
 
ga = BRKGA2(number_gen=10, size_pop=20, elite_size=10, mutant_size=2,
            make_logbook=True, repeat=2, random_state=1,
            features_metric_function='poly')

# Fit the classifier
ga.fit(X, y, normalize=True)

colors = cm.rainbow(np.linspace(0, 1, len(ga.pareto_front_)))

for i in range(len(ga.pareto_front_)):
    obj1=[]
    obj2=[]
    auc_=[]
    for j in range(len(ga.pareto_front_[i])):
        obj1.append(ga.all_paretos()[i][j].fitness.values[0])
        obj2.append(ga.all_paretos()[i][j].fitness.values[1])
    obj1.append(obj1[0])
    obj2.append(1)
    plt.scatter(obj2, obj1, color=colors[i], 
                label="AUC {:.3f} ({:d})".format( auc(obj2, obj1, reorder=True), i))

obj1=[]
obj2=[]
for i in range(len(ga.best_pareto_front_)):
    obj1.append(ga.best_pareto()[i].fitness.values[0])
    obj2.append(ga.best_pareto()[i].fitness.values[1])

plt.scatter(obj2, obj1, marker='+', label="Best Pareto", color="black")
plt.title(" Pareto front and Area under Curve")
plt.xlabel("#SelectedFeatures/#TotalFeatures")
plt.ylabel("Accuracy - Variation")
plt.legend()

