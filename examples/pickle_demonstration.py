"""
============================
Using Pickle to save models
============================

You can use pickle to save your model
"""
from feature_selection import HarmonicSearch
from sklearn.datasets import load_breast_cancer
from six.moves import cPickle

dataset = load_breast_cancer()
X, y = dataset['data'], dataset['target_names'].take(dataset['target'])
    
hs = HarmonicSearch(random_state=0, make_logbook=True,
                    repeat=2)

hs._get_accuracy
hs.fit(X,y, normalize=True)
    
file = "HarmonicSearch"

f = open(file +'.save', 'wb')
cPickle.dump(hs, f, protocol=cPickle.HIGHEST_PROTOCOL)
f.close()

