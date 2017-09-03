# Importing Libraries 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from deap import benchmarks
from deap import base, creator
from deap import tools

import scoop
from scoop import futures

from itertools import compress
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVC
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

from random import sample
from random import randint

from datetime import datetime
import arff

# Importing the dataset
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
#N_FEATURES = 5
# Harmony initialization Function
def gen_in():
    RND = randint(0,N_FEATURES)
    return   sample(list(np.concatenate( (np.zeros([N_FEATURES-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), N_FEATURES)

# Evaluation Function 
def evaluate(individual):
    # Select Features
    features = list( compress( range(len(individual)), individual))
    train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]  ).T
    
    # Feature Scaling
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    
    # Create SVM Classifier
    classifier = SVC(kernel = 'linear')
    classifier.fit(X,y)

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y, cv = 3)
    
    return accuracies.mean() - accuracies.std() +  pow( len(features) + 1000 ,-1) ,

#def evaluate(individual):
#    return sum(individual), 

# Function that improvise a new harmony
def improvise(pop, HMCR):
# HMCR = Harmonic Memory Considering Rate
    size = len(pop)
    new_harmony = toolbox.individual()
    for i,x in enumerate(pop):
        new_harmony[i] = pop[randint(0,size-1)][i] 
    toolbox.mutate(new_harmony)
    toolbox.pitch_adjustament(new_harmony)
    
    return new_harmony
    

creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attribute", gen_in)
toolbox.register("improvise", improvise, HMCR = 0.95)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = 0.05)
toolbox.register("pitch_adjustament", tools.mutFlipBit, indpb = 0.05)
toolbox.register("get_worst", tools.selWorst, k = 1)
toolbox.register("evaluate", evaluate)
toolbox.register("map", map)
#toolbox.register("map", futures.map)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen"] + ["best_fit"] + stats.fields

def main(graph = False, log = False):
    
    harmony_mem = toolbox.population(n=50) 
    hof = tools.HallOfFame(1)
    NGEN = 1000

    # Evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, harmony_mem)
    for ind, fit in zip(harmony_mem, fitnesses):
        ind.fitness.values = fit
    
    for g in range(NGEN):

        # Improvise a New Harmony 
        new_harmony = toolbox.improvise(harmony_mem)
        new_harmony.fitness.values = toolbox.evaluate(new_harmony)
        
        # Select the Worst Harmony
        worst = toolbox.get_worst(harmony_mem)[0]
        
        # Check and Update Harmony Memory
        if( worst.fitness.values < new_harmony.fitness.values):
            worst[:] = new_harmony[:]
            worst.fitness.values = new_harmony.fitness.values
            
        
        # Log statistic
        hof.update(harmony_mem)
        logbook.record(gen=g, best_fit= hof[0].fitness.values[0], **stats.compile(harmony_mem))
        if( g % 100 == 0):
            print("Generation: ", g + 1 , "/", NGEN, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
        #scoop.logger.info("Generation: %d", g)
        
        
    
    if(log):  
        print(logbook)
    
    if(graph):
        gen = logbook.select("gen")
        acc_mins = logbook.select("min")
        acc_maxs = logbook.select("max")
        acc_avgs = logbook.select("avg")

        
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Acc")
        line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Acc")
        line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Acc")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Accuracy")
        
        lns = line1 + line2 + line3 
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        plt.show()
    
    
    return hof[0], logbook

if __name__ == "__main__":
    best, logbook = main( graph = True)

#del CXPB, MUTPB, NGEN, offspring, fitnesses,invalid_ind,mutant, ind, fit, g, child1,child2, N_FEATURES





### Código guardado
#  1 - A ) GENERAL INITIALIZATION
#toolbox.register("attribute", random.randrange, 2)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                 toolbox.attribute, n= N_FEATURES)
## Problema na inicialização acima: a maioria dos indivíduos terá um número de 
    
## atributos selecionados próximo de 50 % 
