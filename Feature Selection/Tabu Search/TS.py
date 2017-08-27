# Importing Libraries
"Simple Tabu search does not seem as a good algorithm for feature selection for big feature"
" sets. It is too slow to look for one different feature at a time. So, there is a need for "
" more complex moves. I still need to do more research "
 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import benchmarks
from deap import base, creator
from deap import tools

import scoop
# python -m scoop -n 4 your_program.py
from scoop import futures

from itertools import compress
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

from sklearn.neighbors import KernelDensity

from random import sample
from random import randint

from datetime import datetime
import arff

# Parameters
TABU_SIZE = 40

# Importing the dataset
dataset = arff.load(open('Colon.arff'))
data = np.array(dataset['data'])
label = data[:, -1]
X = data[:,:-1].astype(float)

# Encoding categorical data in Y
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(label) #.reshape((-1,1))
Y.shape = (len(Y),1)

# Feature Scaling in X
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

# Cleaning variabels
del dataset, data, label

toolbox = base.Toolbox()
toolbox.register("map", map)
#toolbox.register("map", futures.map)
#toolbox.register("print", scoop.logger.info)
toolbox.register("print", print)

N_FEATURES = len(X[0])
#N_FEATURES = 5

def gen_in():
    RND = randint(0,N_FEATURES)
    return   sample(list(np.concatenate( (np.zeros([N_FEATURES-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), N_FEATURES)

def evaluate(individual):
    # Select Features
    features = list( compress( range(len(individual)), individual))
    train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]  ).T
    
    # Feature Scaling
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    
    # Create SVM Classifier
    classifier = SVC(kernel = 'linear')

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y.ravel(), cv = 3)
    
    return accuracies.mean() - accuracies.std() +  pow( len(features) + 1000 ,-1) ,

def walk(individual):
    
    
    i = randint(0, N_FEATURES-1)
    
    while(i in individual.tabu[-TABU_SIZE:]):
        i = randint(0, N_FEATURES-1)

    individual[i] = int(not individual[i])        
    individual.tabu.append(i)
    individual.tabu
    
        
        
creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.Fitness, tabu = None)


toolbox.register("attribute", gen_in)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = 0.05)
toolbox.register("evaluate", evaluate)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen"] + ["best_fit"] + stats.fields

np.diag(np.ones(N_FEATURES, dtype = int))

def main():
    
    individual = toolbox.individual()
    hof = tools.HallOfFame(1)
    hof.update([individual])
    GEN = 30

    for g in range(GEN):
    
#    individual.fitness.values = toolbox.evaluate(individual)
    hof.update([individual])
    
    
if __name__ == "__main__":
   swarm, logbook, best = main()

