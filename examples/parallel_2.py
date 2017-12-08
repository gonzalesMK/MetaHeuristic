# Importing Libraries 
" Algorithm is working by know. But it is too slow. Calculating all those symmetrical"
"uncertanties is unfeasible. There is a space to some random search in it."

## Import dataset
from multiprocessing import Process, Pool
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import arff


from feature_selection import HarmonicSearch
from feature_selection import GeneticAlgorithm
from feature_selection import RandomSearch
from feature_selection import BinaryBlackHole
from feature_selection import SimulatedAnneling
from feature_selection import BRKGA
from sklearn.naive_bayes import GaussianNB
from six.moves import cPickle
from functools import partial

if __name__ == "__main__":

    # Importing the dataset
    dataset = arff.load(open('Colon.arff'))
    data = np.array(dataset['data'])
    label = data[:, -1]
    X = data[:,:-1].astype(float)
    
    # Encoding categorical data in Y
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(label) #.reshape((-1,1))
    
    # Feature Scaling in X
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    
    
    # Cleaning variabels
    del dataset, data, label
    
    test = [1,2,3]
    clf1 = []
    
    file = "Testing_number_gen_1"
    f = open(file +'.save', 'wb')
    f.writable()
    
    jobs = []
    for i in test:
        print("Parameter: ", i)
        classifier = BRKGA(number_gen = i,repeat = 1, make_logbook=True,
                           verbose=True,)
        f = partial(classifier.fit, X=X, y=Y)
        p = Process(target=f)
        
        jobs.append(p)
        p.start()
    
    clf1.append(classifier)
    cPickle.dump(clf1, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    