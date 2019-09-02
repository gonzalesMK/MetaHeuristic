from pylab import arange, pi, sin, cos, sqrt
import random
from collections import Counter
import pylab
from math import sqrt
import os
from pylatex.utils import italic, bold
from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis, \
    Plot, Figure, Matrix, Alignat, MultiColumn
import lzma
from sklearn.metrics import auc
import warnings
import numpy as np
from matplotlib import pyplot as plt
from six.moves import cPickle
import matplotlib
# matplotlib.use('ps')
matplotlib.use('Qt5Agg')

"""
Sintetiza os resultados dos experimentos para cada conjunto de dados

Figura 1:

Geração vs Média de cada Geração do coeficiente de Matthews (entre todas as logfiles)

-> Útil para verificar a convergência (ou não) do algoritmo no quesito otimizador

Figura 2: 

Geração vs Média de cada Geração da Seleçao de Características (entre todas as logfiles)

-> Útil para verificar a convergência (ou não) do algoritmo no quesito FS

Figura 3:
    Plot de cada um dos Pareto Fronts 

Figura 4:
    Histograma dos resultados CV

Print:
 * Média do Score do Cross Validation ( Teste)
 * Média do Score final de treino durante as etapas de treinamento do CV (é útil para verificar overfit do modelo )
 * Tempo de treinamento


"""

home = "/media/gonzales/DATA/TCC_Andre/datasets/results/"

# Get the name of all datasets in the Experiment Folder
dataset_names = os.listdir(home)

colors = ["b", "g", "r", "c", "m", "y", "k", "w"]

# For each dataset
for name, c in zip(dataset_names, colors):
    
    # Get all the metaheuristics names from files
    filenames = os.listdir(home + name)
    meta_names = [fn.split('.')[0] for fn in filenames]
    
    # Get number of experiments for each metaheuristic
    counter_meta = Counter(meta_names)
    
    for meta, n_experiments in list(counter_meta.items()):
        log_results = []
        
        # Gather Results
        for ind in range(n_experiments):
            filename = home + name + '/' + meta + '.' + str(ind) + '.lzma'
            
            try:
                file = lzma.open(filename, 'rb')
                print("Openned: ", filename)
                log_results = log_results + cPickle.load(file)
            except FileNotFoundError:
                print("ERROR WITH :", filename)
        
        # Get statistics: 
        test_scores = [ log['test_score'] for log in log_results] 
        final_train_scores = [ log['best_mask_'].fitness.values[0] for log in log_results]
        fit_times = [log['fit_time']   for log in log_results] 
        pareto_fronts = [log['best_pareto'] for log in log_results]
        
        print("Algoritmo: " + meta, end=' ')
        print("Dataset: " + name)
        print("\tAverage Final Test Score: {}".format( np.mean(test_scores)))
        print("\tAverage Final Train Score: {}".format( np.mean(final_train_scores)))
        print("\tAverage Fit Time: {}".format( np.mean(fit_times)))

        # Plotar Figura 4
        plt.figure(4)
        plt.hist(test_scores)
        plt.title("Final CV's Test Score Distribution")

        # Plotar Figura 3
        plt.figure(3)
        for pf in pareto_fronts:
            pf = [i for i in pf ]
            pf.sort( key = lambda ind: ind.fitness.values[1])
            plt.plot( [sum(i) for i in pf] , [i.fitness.values[0] for i in pf])
        plt.title("Pareto Front Found")

        # # Get training results
        size_pop = log_results[0]['params']['size_pop']
        n_gen = log_results[0]['params']['number_gen']
        gen_n_individuals = [(n+1) * size_pop for n in range(n_gen)]

        gen_avg_matthews = np.mean( [log['logbook'][0].chapters["fitness"].select("avg") for log in log_results], axis = 0)
        gen_min_matthews = np.mean( [log['logbook'][0].chapters["fitness"].select("min") for log in log_results], axis = 0)
        gen_max_matthews = np.mean( [log['logbook'][0].chapters["fitness"].select("max") for log in log_results], axis = 0)
        gen_avg_feature   = np.mean( [log['logbook'][0].chapters["size"].select("avg") for log in log_results] , axis = 0)
        gen_min_feature =  np.mean( [log['logbook'][0].chapters["size"].select("min") for log in log_results] , axis = 0)
        gen_max_feature =  np.mean( [log['logbook'][0].chapters["size"].select("max") for log in log_results] , axis = 0)
        
        # Plotar Figura 1
        plt.figure(1)
        plt.plot(gen_n_individuals, gen_avg_matthews, '--', label=meta, color=c)
        plt.plot(gen_n_individuals, gen_max_matthews, label=meta + "- Max", color=c)
        plt.plot(gen_n_individuals, gen_min_matthews, label=meta + "- Min", color=c)  

        # Plotar Figura 2
        plt.figure(2)
        plt.plot(gen_n_individuals, gen_avg_feature, '--', label=meta + " - Média", color=c) 
        plt.plot(gen_n_individuals, gen_min_feature, label=meta + "- Min", color=c)
        plt.plot(gen_n_individuals, gen_max_feature, label=meta + "- Max", color=c)

    plt.figure(1)
    plt.title("Otimização do Coeficiente de Matthews durante os treinamentos")
    plt.legend()
        
    plt.figure(2)
    plt.title("Otimização do número de atributos durante os treinamentos")
    plt.legend()
    plt.show()
