import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from meta_class import HarmonicSearch, GeneticAlgorithm
import arff


# Importing the dataset
dataset = arff.load(open('Colon.arff'))
data = np.array(dataset['data'])
label = data[:,-1]
X = data[:,:-1].astype(float)

# Encoding categorical data in Y
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(label)

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
#                    'verbose': [True],
#                    'estimator': [classifier],
#                    'predict_with': ['all']} ]
#    
    
#    clf = GridSearchCV(estimator = ga_, param_grid= parameters, scoring = ga_.score_func_to_grid_search, verbose = 1)    
#    clf.fit(X,Y)
    ga = GeneticAlgorithm(estimator = classifier, verbose = False)
    hs = HarmonicSearch(estimator = classifier, repeat_ = 2)
    
    hs.fit(X= X, y = Y)
    ga.fit(X= X, y = Y)
    return hs, ga


if __name__ == "__main__":
    hs,ga = main()
    print(hs.best_fitness)
    print(ga.best_fitness)


     
    

