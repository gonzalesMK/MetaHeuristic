# Implementação do paper nesta pasta, 
# Porém, a parte diferente é que a atualização das estrelas é feita conjuntamente e, não, unitariamente
# como descrita no paper
# ERRO ?! : 
# No paper, também é verificado, provavelmente, um erro de conceito. Primeiro, note que o RAIO do buraco negro
# ,calculado como o valor de fitness do buraco negro divido pela soma das  outras fitness, é comparado com a 
# distância euclidiana entre a estrela e o buraco negro. Entretanto, eu imagino que o raio seja um valor < 1, enquanto o menor
# valor para a distância é 1 ! Ou seja, as estrelas só irão sumir quando forem iguais ao buraco negro!

# O que melhorar:
    # O cálculo do Raio do Buraco Negro
    # Escolher classifier adequado

import numpy as np
import random
from random import sample
from random import randint

# Parallel computation with SCOOP
import scoop
from scoop import futures

# For classification  and preprocessing purpose
from itertools import compress
from sklearn.preprocessing import StandardScaler
from sklearn.svm import  SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

# DEAP libraries
from deap import base
from deap import creator
from deap import tools

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

# Individual initialization Function
def generate():
    RND = randint(0,N_FEATURES)
    return   sample(list(np.concatenate( (np.zeros([N_FEATURES-RND,], dtype=int), np.ones([RND,], dtype=int)), axis = 0)), N_FEATURES)


# Evaluation Function
def evaluate(individual):
    # Select Features
    if( sum(individual) == 0):
        return 0 , 
    features = list( compress( range(len(individual)), individual))
    train =  np.reshape([X[:, i] for i in features], [ len(features),  len(X)]  ).T
    
    # Feature Scaling
    sc_X = StandardScaler()
    train = sc_X.fit_transform(train)
    
    # Create Classifier
    classifier = SVC(kernel = 'rbf')
#   classifier = RandomForestClassifier()

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y, cv = 10)
    
    return accuracies.mean() - accuracies.std() +  pow( len(features) + 1000 ,-1) ,

# Update particle attributes function
def updateStar(star , blackhole):
    if dist(star, blackhole) < blackhole.radius :
        star[:] = toolbox.galaxy(n=1)[0]            
    else:
        star[:] = [ 1 if  abs(np.tanh(star[x] + random.uniform(0,1) * (blackhole[x] - star[x]))) > random.uniform(0,1) else 0 for x in range(0,N_FEATURES)]

# Calculate Euclidean distance beetween star
def dist( star, blackhole):
    return np.linalg.norm([blackhole[i] - star[i] for i in range(0,N_FEATURES)])

creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Star", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attribute", generate)
toolbox.register("star", tools.initIterate, creator.Star, toolbox.attribute)
toolbox.register("galaxy", tools.initRepeat, list, toolbox.star)
toolbox.register("update", updateStar)
toolbox.register("evaluate", evaluate)
#toolbox.register("map", futures.map)
toolbox.register("map", map)

# Statistic 
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)
hof = tools.HallOfFame(2)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen", "evals"] + stats.fields

def main():
    galaxy = toolbox.galaxy(n=20)
    GEN = 60

    for g in range(GEN):
        print("\033[2A Generation: ", g )
        # scoop.logger.info("Generation: %d ", g)
 
       # Evaluate the entire population
        fitnesses = toolbox.map(toolbox.evaluate, galaxy)
        for ind, fit in zip(galaxy, fitnesses):
            ind.fitness.values = fit
    
        # Update Global Information
        hof.update(galaxy)    
        hof[0].radius = hof[0].fitness.values[0] / sum( i.fitness.values[0] for i in galaxy )
        
        # Update particles
        for part in galaxy:
            toolbox.update(part, hof[0])
            
        # Log statistic
        logbook.record(gen=g, evals=len(galaxy), **stats.compile(galaxy))

    print(logbook)
         
    return galaxy, logbook, hof[0]
    
if __name__ == "__main__":
   galaxy, logbook, hof = main()
   


# Additional Information0
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
