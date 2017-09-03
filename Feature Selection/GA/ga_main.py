import numpy as np
from six.moves import cPickle
import os
from functools import partial

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
from ga_class import genetic_algorithm
import arff

from sklearn.model_selection import GridSearchCV

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

# Parameters to be tuned    


# Cleaning variabels
del dataset, data, label


def main():

    classifier = SVC(kernel = 'linear', verbose=  False, max_iter = 10000)
    
    for cop in cross_over_prob:
        for gmp in gene_mutation_prob:
            for imp in individual_mutation_probability:
                for n,size in zip(number_gen, size_pop):
                    print('n_' + str(n) + '_size_' + str(size) + '_cop_' + str(cop) +
                             '_gmp_' + str(gmp) + '_imp_' + str(imp))            
                    ga = genetic_algorithm(estimator= classifier, number_gen = n, size_pop = size, 
                                           cross_over_prob = cop, individual_mutation_probability = imp,
                                           X = X, y = Y.ravel(), verbose = False, gene_mutation_prob = gmp,
                                           repeat_ = 10)
    
                    ga.fit(X,Y)

                    f = open('n_' + str(n) + '_size_' + str(size) + '_cop_' + str(cop) +
                             '_gmp_' + str(gmp) + '_imp_' + str(imp) + '.save', 'wb')
                    for obj in [ga.logbook, ga.best_mask, ga.fitness]:
                        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
                    f.close()


    obj = os.listdir()    
    
    files = []    
    for file in obj:
        f = open(file, 'rb')
        loaded_objects = []
        for i in range(3):
            loaded_objects.append(cPickle.load(f))
        files.append(loaded_objects)
    
    'number_gen': [50, 100, 200],
                    'size_pop': [40, 20,  10 ],
    # Parameters
    parameters =[ {  'number_gen': [50],
                    'size_pop': [40],
                    'cross_over_prob': [0.1, 0.3, 0.7, 1],
                    'gene_mutation_prob': [0, 0.05, 0.1],
                    'individual_mut_prob': [0, 0.05, 0.1],
                    'repeat_': [10]}]
    parameters =[ {  'number_gen': [50],
                    'size_pop': [40],
                    'cross_over_prob': [0.1],
                    'gene_mutation_prob': [0.1],
                    'individual_mut_prob': [0.1],
                    'repeat_': [1]}]
    classifier = SVC(kernel = 'linear', verbose=  False, max_iter = 10000)
    ga_ = genetic_algorithm( estimator = classifier, repeat_ = 10, size_pop = 40)
    clf = GridSearchCV(estimator = ga_, param_grid= parameters, scoring = ga_.score_func_to_grid_search, verbose = 10)    
    clf.fit(X,Y)
    ga_.get_params()
    
    