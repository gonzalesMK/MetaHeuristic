import operator
import random

import numpy as np

from deap import base
from deap import benchmarks
from deap import creator
from deap import tools


# Particle initialization Function
def generate(size, pmin, pmax, smin, smax):
    part = creator.Particle(random.uniform(pmin, pmax) for _ in range(size)) 
    part.speed = [random.uniform(smin, smax) for _ in range(size)]
    part.smin = smin
    part.smax = smax
    return part

# Update particle attributes function
def updateParticle(part, best, phi1, phi2):
    if not part.best or part.best.fitness < part.fitness:
        part.best = creator.Particle(part)
        part.best.fitness.values = part.fitness.values
    c1 = (random.uniform(0, phi1) for _ in range(len(part)))
    c2 = (random.uniform(0, phi2) for _ in range(len(part)))
    vel1 = toolbox.map(operator.mul, c1, toolbox.map(operator.sub, part.best, part))
    vel2 = toolbox.map(operator.mul, c2, toolbox.map(operator.sub, best, part))
    part.speed = list(toolbox.map(operator.add, part.speed, toolbox.map(operator.add, vel1, vel2)))
    for i, speed in enumerate(part.speed):
        if speed < part.smin:
            part.speed[i] = part.smin
        elif speed > part.smax:
            part.speed[i] = part.smax
    part[:] = list(toolbox.map(operator.add, part, part.speed))

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Particle", list, fitness=creator.FitnessMax, speed=list, 
    smin=None, smax=None, best=None)
toolbox = base.Toolbox()
toolbox.register("particle", generate, size=2, pmin=-6, pmax=6, smin=-3, smax=3)
toolbox.register("swarm", tools.initRepeat, list, toolbox.particle)
toolbox.register("update", updateParticle, phi1=2.0, phi2=2.0)
toolbox.register("evaluate", benchmarks.h1)
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


def main():
    swarm = toolbox.swarm(n=10)
    GEN = 100

    for g in range(GEN):
        
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
   main()


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