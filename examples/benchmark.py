# Importing Libraries 
" Algorithm is working by know. But it is too slow. Calculating all those symmetrical"
"uncertanties is unfeasible. There is a space to some random search in it."

## Import dataset
import numpy as np
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

mi = mutual_info_classif(X,Y)
selected = mi != 0
X = X[:, selected]

i = []
for i in 
clf = BRKGA(number_gen = 100,repeat = 10, make_logbook=True, 
            verbose=True, cxUniform_indpb=0.8).fit(X,Y)
np.sum(clf.mask_, axis = 1)



clf.plot_results()
