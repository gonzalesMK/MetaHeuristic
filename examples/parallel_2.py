"""
============================
Parallel Example
============================

If a lot of cycles and tests are needed, this approach will lead to more fast
results. Instead of making parallel the classifier, each metaheuristics will run
into a different process.
"""

## Import dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
from feature_selection import BRKGA
from six.moves import cPickle
from multiprocessing import Pool
import time
from sklearn.datasets import load_breast_cancer

def f(i):
    print("Now in: ", int(i))
    a = BRKGA(size_pop=10, mutant_size=2, elite_size=2,
                  number_gen = int(i),repeat = repetition, make_logbook=True, 
                  verbose=False, cxUniform_indpb=0.9).fit(X,y)
    return a

if __name__ == "__main__":

    dataset = load_breast_cancer()
    X, y = dataset['data'], dataset['target_names'].take(dataset['target'])    
    # Feature Scaling in X
    sc_X = StandardScaler()
    X = sc_X.fit_transform(X)
    
    # Cleaning variabels
    del dataset, data, label
    
    # Teste A
    print("Teste A")
    t0 = time.time()
    
    number_gen = np.linspace(1,4,num=3)
    repetition = 2

    pool = Pool()              # start 4 worker processes
    
    clfsA=list( pool.map(f,number_gen))
    pool.close()
    
    print("Final Time: ", time.time()- t0)
    file = open('teste_1_A2.save', 'wb')
    cPickle.dump(clfsA, file, protocol=cPickle.HIGHEST_PROTOCOL)
    file.close()   
           