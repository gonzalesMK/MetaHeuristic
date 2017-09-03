# Importing Libraries 
" Algorithm is working by know. But it is too slow. Calculating all those symmetrical"
"uncertanties is unfeasible. There is a space to some random search in it."

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

    # Applying K-Fold Cross Validation
    accuracies = cross_val_score( estimator = classifier, X = train, y = Y.ravel(), cv = 3)
    
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
    
def entropy(vec, base=2):
    " Returns the empirical entropy H(X) in the input vector."
    # Calcula a densidade de probabilidade de vec - P(vec)
    if (vec.dtype != np.int64): # To continuous variables
#        GM = GaussianMixture(verbose=False, n_components = 2)
#        GM.fit(vec)
#        prob_vec = np.exp(GM.score_samples(vec))
        kernel = KernelDensity()
        kernel.fit(vec)
        prob_vec = np.exp(kernel.score_samples(vec))
        
    else: # For discrete variabes
        _, vec = np.unique(vec, return_counts=True)
        prob_vec = np.array(vec/float(sum(vec)))
    
    if base == 2:
        logfn = np.log2
    if base == 10:
        logfn = np.log10
    else:
        logfn = np.log
    return -(prob_vec.dot(logfn(prob_vec)))
   

def conditional_entropy(x,y):
    "Returns H(X|Y)."
    if( y.dtype != np.int64 ):
        # Calcula a densidade de probabilidade de y - P(y)
#        GM = GaussianMixture(verbose=False, n_components = 2)
#        GM.fit(y)
#        Py = np.exp(GM.score_samples(y))        
        kernel = KernelDensity()
        kernel.fit(y)
        Py = np.exp(kernel.score_samples(y))
        
        # Calcula a probabilidade de x e y- P(x,y)
        GM = GaussianMixture(verbose=False, n_components = 2)
        GM.fit(np.append(x,y, axis=1))
        Pxy = np.exp(GM.score_samples(np.append(x,y, axis=1)))
        
        return -sum( Pxy * np.log( Pxy / Py))
    else:
        GM = GaussianMixture(verbose=False, n_components = 2)
        GM.means_init = np.array([x[y == 1].mean(axis=0) for i in range(2)]).reshape(-1,1)    
        GM.fit(x)
        Px_y = GM.predict_proba(x)[:,0:1]  # P(x|y)
        _ , ny = np.unique(y, return_counts=True) 
        Py = np.array(ny/float(sum(ny))).reshape((-1,1))    # P(y)
        Pxy = Px_y.copy()
        Pxy[y==0] = Px_y[y==0] * Py[0]
        Pxy[y==1] = Px_y[y==1] * Py[1]
            
        return - sum(Pxy * np.log(Px_y))

     
def mutual_information(x, y):
    " Returns the information gain/mutual information [H(X)-H(X|Y)] between two random vars x & y."
    return entropy(x) - conditional_entropy(x, y)

def symmetrical_uncertainty(x, y):
    " Returns 'symmetrical uncertainty' (SU) - a symmetric mutual information measure."
    return 2.0*mutual_information(x, y)/(entropy(x) + entropy(y))    

def c_correlation(X, y):
    toolbox.print("C_correlations")
    """
    Returns SU values between each feature and class.
    
    Parameters:
    -----------
    X : 2-D ndarray
        Feature matrix.
    y : ndarray
        Class label vector
        
    Returns:
    --------
    su : ndarray
        Symmetric Uncertainty (SU) values for each feature.
    """
    su = np.zeros(X.shape[1])
    symmetrical = toolbox.map(lambda i: symmetrical_uncertainty(X[:,i:i+1], y), np.arange(X.shape[1]))
    for i, sui in zip(np.arange(X.shape[1]), symmetrical):
#        if (i % 100 == 0):
#            toolbox.print(chr(27) + "[2J",i/max(y.shape),"%")
        su[i] = sui
    return su


def density_prob(x):
    estimators = [(KernelDensity()) for i in range(x.shape[1])]
    P = np.zeros(x.shape)
    for index, kernel in enumerate(estimators):
        kernel.fit(x[:,index:index+1])
        P[:,index] = np.exp(kernel.score_samples(x[:,index:index+1]))
    return P

def add(correlation, individual):
    # Excluded Subset   
    excluded_features = np.array(individual) == 0
    highest = np.argmax(correlation * excluded_features)
    individual[highest] = 1

def delete(correlation, individual):   
    
    selected = np.array(individual) == 1
    highest = np.argmax(correlation * selected)        
    su = toolbox.map( lambda i: symmetrical_uncertainty(X[:,i:i+1], X[:,highest:highest+1]),list( compress( range(len(individual)), individual)))
    for i, sui in zip(list( compress( range(len(individual)), individual)),su):
        if( (sui > correlation[i]) and (i != highest)):
            individual[i] = 0
    
creator.create("Fitness", base.Fitness, weights=(1.0, ))
creator.create("Individual", list, fitness=creator.Fitness)


toolbox.register("attribute", gen_in)
toolbox.register("improvise", improvise, HMCR = 0.95)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attribute)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mutate", tools.mutUniformInt,low = 0, up = 1, indpb = 0.05)
toolbox.register("pitch_adjustament", tools.mutFlipBit, indpb = 0.05)
toolbox.register("get_worst", tools.selWorst, k = 1)
toolbox.register("evaluate", evaluate)
c = np.nan_to_num(c_correlation(X, Y))
P = density_prob
toolbox.register("add", add, c)
toolbox.register("delete", delete, c)

# Statistics
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

# Logbook
logbook = tools.Logbook()
logbook.header = ["gen"] + ["best_fit"] + stats.fields


def main(graph = False, log = True):

    harmony_mem = toolbox.population(n=10) 
    hof = tools.HallOfFame(1)
    NGEN = 5

    # Evaluate the entire population
    fitnesses = toolbox.map(toolbox.evaluate, harmony_mem)
    for ind, fit in zip(harmony_mem, fitnesses):
        ind.fitness.values = fit
    del fit,ind
    
    for g in range(NGEN):
        toolbox.print(chr(27)) #+ "[2J")
        toolbox.print("Busca Global - ", g/NGEN * 100, "%")
        # Improvise a New Harmony 
        new_harmony = toolbox.improvise(harmony_mem)
        new_harmony.fitness.values = toolbox.evaluate(new_harmony)
        fit = new_harmony.fitness.values[0]
        
        i = 0
        # Local search
        while( (fit >= new_harmony.fitness.values[0]) and (i < 3)):
            toolbox.print("Busca Local - ", i/3, "%")
            toolbox.add(new_harmony)
            toolbox.delete(new_harmony)
            new_harmony.fitness.values = toolbox.evaluate(new_harmony)
            i = i+1
        if((fit >= new_harmony.fitness.values[0])):
            toolbox.print("improved")
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
    best, logbook = main()

#del CXPB, MUTPB, NGEN, offspring, fitnesses,invalid_ind,mutant, ind, fit, g, child1,child2, N_FEATURES





### Código guardado
#  1 - A ) GENERAL INITIALIZATION
#toolbox.register("attribute", random.randrange, 2)
#toolbox.register("individual", tools.initRepeat, creator.Individual,
#                 toolbox.attribute, n= N_FEATURES)
## Problema na inicialização acima: a maioria dos indivíduos terá um número de 
    
## atributos selecionados próximo de 50 % 

### Check the distribuition 
## Checking the data to gene f
#f = 10
#
## Check the real distribution
#plt.hist(X[Y==1,f:f+1], alpha = 0.75, normed= True, color = "red")
#plt.hist(X[Y==0,f:f+1], alpha = 0.75, normed= True, color = "blue")
#
## Feature Quantization
#plt.figure()
#GM = GaussianMixture(verbose=True, n_components = 2)
#GM.fit(X[:,f:f+1], y = Y)
#Samples, labels = GM.sample(500)
#plt.hist(Samples[labels==1], alpha = 0.75, normed= True, color = "red")
#plt.hist(Samples[labels==0], alpha = 0.75, normed= True, color = "blue")
# Gaussian Mix
#f = 10
#GM = GaussianMixture(verbose=True, n_components = 2)
#GM.fit(X[:,f:f+1])
#Post = GM.predict_proba(X[:,f+1:f+2])
#Pred = GM.predict(X[:,f+1:f+2])
#Score2 = np.exp(GM.score_samples(X[:,f:f+1]))
#
#del f


# Calcula o erro bayesiano    
#def state_gene(x,y,Y):
#    GM = GaussianMixture(verbose=True, n_components = 2)
#    GM.means_init = np.array([x[Y == 1].mean(axis=0) for i in range(2)]).reshape(-1,1)    
#    GM.fit(x)
#    Px1 = np.exp(GM.score_samples(x)) # Probabilistic Density: the change to get each gene
#    Pz0 = GM.predict_proba(x)[:,0]
#    Pz1 = GM.predict_proba(x)[:,1]
#    df0 = np.asarray(~(Pz0 > 0.5), dtype = int)    
#    e = (sum((Pz0[Y==1]*(Y[Y==1] == df0[Y==1])))+ sum(Pz1[Y==0]*(Y[Y==0] != df0[Y==0])))/len(Y) # (Falta relacionar cada Pz0 com o fato de estar certo ou errado)
#    Px2_x1 = GM.predict_proba(y) 
