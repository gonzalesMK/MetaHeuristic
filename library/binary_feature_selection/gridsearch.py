import numpy as np
import sys
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
sys.path.insert(0, 'C\\Users\\Juliano D. Negri\\Documents\\Facul\\IC - Andre\\MetaHeuristic\\library\\binary_feature_selection')
from metaheuristics import HarmonicSearch, GeneticAlgorithm
import arff


# Importing the dataset
dataset = arff.load(open('Colon.arff'))
data = np.array(dataset['data'])
label = data[:,-1]
X = data[:,:-1].astype(float)

# Encoding categorical data in Y
#labelencoder_y = LabelEncoder()
Y = label # labelencoder_y.fit_transform(label)

sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Cleaning variabels
del dataset, data, label


def main():
    classifier = SVC(kernel = 'linear', verbose=  False, max_iter = 10000)
#    parameters =[ {  'number_gen': [50],
#                    'size_pop': [40],
#                    'cross_over_prob': [0.1],
#                    'gene_mutation_prob': [0.1],
#                    'individual_mut_prob': [0.1],
#                    'repeat_': [3],
#   d                 'verbose': [True],
#                    'estimator': [classifier],
#                    'predict_with': ['all']} ]
#    
    
#    clf = GridSearchCV(estimator = ga_, param_grid= parameters, scoring = ga_.score_func_to_grid_search, verbose = 1)    
#    clf.fit(X,Y)

    hs =  HarmonicSearch(classifier = classifier,
                         number_gen = 200,
                         mem_size = 50,
                         make_logbook = True,
                         random_state = 2,
                         repeat_ = 1)
    hs.fit(X= X, y = Y)

    return hs


if __name__ == "__main__":
    hs = main()
    hs.plot_results()
    
    
    
    print(hs.best_fitness)


     
    

