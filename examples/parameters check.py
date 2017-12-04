# Importing Libraries 
" Algorithm is working by know. But it is too slow. Calculating all those symmetrical"
"uncertanties is unfeasible. There is a space to some random search in it."

## Import dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import arff

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

from feature_selection import HarmonicSearch
from feature_selection import GeneticAlgorithm
from feature_selection import RandomSearch
from feature_selection import BinaryBlackHole
from feature_selection import SimulatedAnneling
from feature_selection import BRKGA
from sklearn.naive_bayes import GaussianNB
from six.moves import cPickle


if __name__ == "__main__":
    
    test = [50,100,200,300]
    repetition = 50
    print("Testing number generation: \n\t", test)
    
    clf1 = []
    clf2 = []
    
    file = "Testing_number_gen_1"
    f = open(file +'.save', 'wb')
    f.writable()
    
    for i in test:
        print("Parameter: ", i)
        classifier = BRKGA(number_gen = i,repeat = repetition, make_logbook=True, parallel=True,
                           verbose=True, cxUniform_indpb=0.6).fit(X,Y)
        clf1.append(classifier)
    cPickle.dump(clf1, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    
    file = "Testing_number_gen_2"
    f = open(file +'.save', 'wb')
        
    for i in test:
        print("Parameter: ", i)
        classifier = BRKGA(size_pop=80, mutant_size=20, elite_size=20,parallel=True,
                           number_gen = int(i/2),repeat = repetition, make_logbook=True, 
                           verbose=True, cxUniform_indpb=0.9).fit(X,Y)
        clf2.append(classifier)
    
    cPickle.dump(clf2, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
        
    for i,n in zip(test,range(len(test))):
        plt.scatter(np.repeat(i,repetition), np.sum(clf1[n].mask_,axis=1), marker='.' )
        plt.scatter(i, np.median(np.sum(clf1[n].mask_,axis=1) ))
        plt.show()

    for i,n in zip(test,range(len(test))):
        plt.scatter(np.repeat(i+0.01,repetition), np.sum(clf2[n].mask_,axis=1), marker='.' )
        plt.scatter(i+0.01, np.median(np.sum(clf2[n].mask_,axis=1) ))
        plt.show()
    