""" Binary Particle Swarm Optimization - Implemented as the paper in this directory 
Still not working."""

import operator
import random
from random import sample
from random import randint

import numpy as np

from deap import base
from deap import creator
from deap import tools

from itertools import compress
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from scipy.special import expit

import networkx
import arff

dataset = arff.load(open('Colon.arff'))
data = np.array(dataset['data'])
label = data[:,-1]
X = data[:,:-1].astype(float)

# Encoding categorical data in Y
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(label)

# Cleaning variabels
del dataset, data, label

N_FEATURES = len(X[0])


# Particle initialization Function
def generate(size, pmin, pmax, smin, smax):
    
    RND = randint(0,N_FEATURES)
    part = creator.Particle(sample(list(np.concatenate( (np.zeros([size-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), size))
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# Evaluation Function
def evaluate(individual):
    # First Check
    if( sum(individual) == 0):
        return 0 , 
    # Select Features
    features = list( compress( range(len(individual)), individual))
    train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]  ).T
    
    # Feature Scaling
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    
    # Create SVM Classifier
    classifier = SVC(kernel = 'linear')

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y, cv = 3)
    
    return accuracies.mean() - accuracies.std(), pow(sum(individual)/10000,2)


# Update particle attributes function
def updateParticle(part, best, phi1, phi2):
    if not part.best or part.best.fitness < part.fitness:
        part.best = creator.Particle(part)
        part.best.fitness.values = part.fitness.values
    
    c1 = (random.uniform(0, phi1) for _ in range(len(part)))
    c2 = (random.uniform(0, phi2) for _ in range(len(part)))
    vel1 = toolbox.map(operator.mul, c1, toolbox.map(operator.sub, part.best, part))
    vel2 = toolbox.map(operator.mul, c2, toolbox.map(operator.sub, best, part))

    part.speed[:] = list(toolbox.map(operator.add, part.speed, toolbox.map(operator.add, vel1, vel2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    
    part[:] = (expit(part) > random.uniform(0,1)).astype(int)
    
    return part,

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, smin=None, smax=None, best=None)

toolbox = base.Toolbox()
toolbox.register("particle", generate, size=N_FEATURES, pmin=-1, pmax=1, smin=-0.3, smax=0.3)
toolbox.register("swarm", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=1.0, phi2=1.0)
toolbox.register("evaluate", evaluate)
#toolbox.register("map", future.map)
toolbox.register("map", map)

# Statistic 
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
hof = tools.HallOfFame(1)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

# history
history = tools.History()
toolbox.decorate("update", history.decorator)

def main(flag = True):
    swarm = toolbox.swarm(n=10)
    history.update(swarm)
    GEN = 2

    for g in range(GEN):
        print(g)
        # Evaluate the entire population
        fitnesses = toolbox.map(toolbox.evaluate, swarm)
        for ind, fit in zip(swarm, fitnesses):
            ind.fitness.values = fit
    
        # Update Global Information
        hof.update(swarm)    
    
        # Update particles
        for part in swarm:
            toolbox.update(part, hof[0])
            
        # Log statistic
        logbook.record(gen=g, evals=len(swarm), **stats.compile(swarm))

    print(logbook)
    
   
    return swarm, logbook, hof[0]

if __name__ == "__main__":
   swarm, logbook, best = main()


# Additional Information
    # https://matplotlib.org/examples/color/colormaps_reference.html with colors    
#import matplotlib.pyplot as plt
#from matplotlib import colors as mcolors
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#from sklearn.preprocessing import MinMaxScaler
# A beautiful plot of the position of the particles and it's fitness as a colormap

#    norm = mcolors.Normalize(vmin = 0, vmax = 2) 
#    a = np.asarray(swarm)
#    plt.scatter(x = a[:,0] ,y = a[:,1], c = np.asarray([i.fitness.values for i in swarm ]) , cmap= plt.get_cmap('inferno'), marker = '.', norm = norm)

#if( flag):
#        graph = networkx.DiGraph(history.genealogy_tree)
#        graph = graph.reverse()     # Make the grah top-down
#        colors = [toolbox.evaluate(history.genealogy_history[i])[0] for i in graph]
#        networkx.draw(graph, node_color=colors)
#        plt.show()
#    