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

from random import sample
from random import randint
import random

from datetime import datetime

# Functions in this code are one dimensional and bounded
BOUND_LOW , BOUND_UP = -600, 600
NDIM = 100

# Individual initialization Function
def uniform(low, up, size=None):
    try:
        return [random.uniform(a, b) for a, b in zip(low, up)]
    except TypeError:
        return [random.uniform(a, b) for a, b in zip([low] * size, [up] * size)]
    
creator.create("Fitness", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attribute", uniform, BOUND_LOW, BOUND_UP, NDIM)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0)
toolbox.register("mutate", tools.mutPolynomialBounded, low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=1/NDIM)
toolbox.register("select", tools.selTournament, tournsize=3)
#toolbox.register("evaluate", benchmarks.h1)
toolbox.register("evaluate", benchmarks.griewank)
#toolbox.register("map", futures.map)
toolbox.register("map", map)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
hof = tools.HallOfFame(1)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen"] + stats.fields
#logbook.chapters["Accuracy"].header = "min", "avg", "max", "std"

def main(graph = False, log = True):
    scoop.logger.info("Generating Population")

    pop = toolbox.population(n=80) 
    hof = tools.HallOfFame(1)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 500

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
        
    if( graph ):
        gen = logbook.select("gen")
        acc_mins = logbook.select("min")
        acc_maxs = logbook.select("max")
        acc_avgs = logbook.select("avg")
        
        fig, ax1 = plt.subplots()
        ax1.set_title(("Genetic Algorithm, Pop Size:" ,len(pop)))
        line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Fitness")
        line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Fitness")
        line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Fitness")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness")
        
        lns = line1 + line2 + line3
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        
        plt.show()        
        
    if(log):  
        print(logbook)
    
    return hof[0]

if __name__ == "__main__":
    best = main(graph = True, log = False)
    print("Best Solution:", best, " Fitness: ", best.fitness)
    

#del CXPB, MUTPB, NGEN, offspring, fitnesses,invalid_ind,mutant, ind, fit, g, child1,child2, N_FEATURES





### Código guardado
#  1 - A ) GENERAL INITIALIZATION
#toolbox.register("attribute", random.randrange, 2)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                 toolbox.attribute, n= N_FEATURES)
## Problema na inicialização acima: a maioria dos indivíduos terá um número de 
    
## atributos selecionados próximo de 50 % 
