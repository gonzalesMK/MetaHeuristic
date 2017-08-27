import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import  SVC
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import cross_val_score
from ga_class import genetic_algorithm
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
    CXPB, MUTPB, NGEN = 0.5, 0.2, 
    
    classifier = SVC(kernel = 'linear', verbose=  False, max_iter = 10000)
    
    ga = genetic_algorithm(score_func = sum, estimator= classifier, number_gen = 100, size_pop = 10, X = X, y = Y.ravel())
    
    ga.fit()
    
    ga.fitness    
    sum(ga.mask)
    
    
    
    