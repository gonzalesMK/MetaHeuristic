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
import random

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

# Individual initialization Function
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

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y, cv = 3)
    
    return accuracies.mean() - accuracies.std(), pow(sum(individual)/10000,2)

creator.create("FitnessMin", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", gen_in)
toolbox.register("individual", tools.initIterate, creator.Individual,
                 toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = 0.05)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)
#toolbox.register("map", futures.map)
toolbox.register("map", map)

# Statistics
#stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values[0])
#stats_n = tools.Statistics( key=sum)
#stats = tools.MultiStatistics( Accuracy = stats_fit, N = stats_n)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Logbook
#logbook = tools.Logbook()
#logbook.header = ["gen"] + stats.fields
#logbook.chapters["Accuracy"].header = "min", "avg", "max", "std"
#logbook.chapters["# of Features"].header = "min", "avg", "max", "std" # number of features

logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

def main(graph = False, log = True):
    scoop.logger.info("Generating Population")

    pop = toolbox.population(n=40) 
    hof = tools.HallOfFame(1)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 10

    # Evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for g in range(NGEN):
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring)) 
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness ( new individuals)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
    
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Log statistic
        hof.update(pop)
        logbook.record(gen=g, **stats.compile(pop))
        print("Generation: ", g + 1 , "/", NGEN, "TIME: ", datetime.now().time().minute, ":", datetime.now().time().second)
        #scoop.logger.info("Generation: %d", g)
        
        
    # Plottig Estatistics 
    if( graph ):
        gen = logbook.select("gen")
        acc_mins = logbook.chapters["Accuracy"].select("min")
        acc_maxs = logbook.chapters["Accuracy"].select("max")
        acc_avgs = logbook.chapters["Accuracy"].select("avg")
        n_feat = logbook.chapters["N"].select("avg")
        
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Acc")
        line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Acc")
        line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Acc")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Accuracy")
        
        ax2 = ax1.twinx()
        line4 = ax2.plot(gen, n_feat, "y-", label="Avg Features")
        ax2.set_ylabel("Size", color="y")
        for tl in ax2.get_yticklabels():
            tl.set_color("y")
        
        lns = line1 + line2 + line3 + line4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        
        plt.show()
    
    if(log):  
        print(logbook)
    
    return hof[0]

if __name__ == "__main__":
    best = main()

#del CXPB, MUTPB, NGEN, offspring, fitnesses,invalid_ind,mutant, ind, fit, g, child1,child2, N_FEATURES





### Código guardado
#  1 - A ) GENERAL INITIALIZATION
#toolbox.register("attribute", random.randrange, 2)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                 toolbox.attribute, n= N_FEATURES)
## Problema na inicialização acima: a maioria dos indivíduos terá um número de 
    
## atributos selecionados próximo de 50 % 
