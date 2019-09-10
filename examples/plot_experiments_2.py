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
from matplotlib.ticker import FormatStrFormatter
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


home = "/mnt/BCD29E9DD29E5C08/TCC/Experiments/convergence/"

# Get the name of all datasets in the Experiment Folder
meta_names = os.listdir(home)

for i in reversed(range(len(meta_names))):
    name=meta_names[i]
    if( name.endswith(".py") or name.endswith(".png")):
        print("deleting {}".format(name))
        del meta_names[i]

name = meta_names[0]
colors = ["b", "g", "r", "c", "m",'y', "k", "w"]
c = colors[0]
# For each dataset
plt.rcParams["figure.autolayout"] = True
plt.rcParams["figure.figsize"] =  [9.6, 7.2]
for name, c in zip(meta_names, colors):
    
    # Get all the metaheuristics names from files
    filenames = os.listdir(home + name)
    dataset_names = [fn.split('.')[0] for fn in filenames]
    
    # Get number of experiments for each metaheuristic
    counter_meta = Counter(dataset_names)
    
    meta, n_experiments = list(counter_meta.items())[0]

    for meta, n_experiment in list(counter_meta.items()):
        log_results = []
        
        # Gather Results
        for n_experiments, filename in enumerate(filenames):
            try:
                file = lzma.open(home + name + "/" + filename, 'rb')
                print("Openned: ", filename)
                result = cPickle.load(file)
                file.close()
            except FileNotFoundError:
                print("ERROR WITH :", filename)
        
            # Get statistics: 
            logbook = result.logbook[0]
            
            # # Get training results
            #size_pop = logbook['params']['size_pop']
            #n_gen = log_results[0]['params']['number_gen']
            gen_n_individuals = [log['gen'] for log in logbook]

            gen_50_matthews =  logbook.chapters["fitness"].select("50_percentile")
            gen_25_matthews =  logbook.chapters["fitness"].select("25_percentile")
            gen_75_matthews =  logbook.chapters["fitness"].select("75_percentile")
            gen_min_matthews = logbook.chapters["fitness"].select("min")
            gen_max_matthews = logbook.chapters["fitness"].select("max")

            gen_50_feature =  logbook.chapters["size"].select("50_percentile")
            gen_25_feature =  logbook.chapters["size"].select("25_percentile")
            gen_75_feature =  logbook.chapters["size"].select("75_percentile")
            gen_avg_feature = logbook.chapters["size"].select("avg")
            gen_min_feature = logbook.chapters["size"].select("min")
            gen_max_feature = logbook.chapters["size"].select("max")
            
            
            try:
                gen_time = [ "{:.0f}".format(np.floor( (i['time']-logbook[0]['time'])/60)) for i in logbook]
            except:
                print("ERROR IN {} {}".format(meta,n_experiments))
                continue
            
            fig, axs = plt.subplots(2)
            
            
            # Plotar Figura 1
            axs[0].set_title("Otimização do Coeficiente de Matthews em {}".format(result.name))
            axs[0].plot(gen_n_individuals, gen_50_matthews, '--', label=meta, color=c, linewidth=1)
            axs[0].fill_between(gen_n_individuals, gen_25_matthews, gen_75_matthews, alpha=0.1, color=c, label="25|75 percentile")
            axs[0].plot(gen_n_individuals, gen_max_matthews, label=meta + "- Max", color=c, linewidth=0.5)
            axs[0].plot(gen_n_individuals, gen_min_matthews, label=meta + "- Min", color=c, linewidth=0.5)  
            axs[0].set_xlabel("Generation")
            axs[0].set_ylabel("Matthews Coef.")
            ax2 = axs[1].twiny()
            ax2.set_xlim(axs[1].get_xlim())
            ax2.set_xticks(gen_n_individuals[1::10])
            ax2.set_xticklabels(gen_time[1::10])
            ax2.xaxis.set_ticks_position('bottom') # set the position of the second x-axis to bottom
            ax2.xaxis.set_label_position('bottom') # set the position of the second x-axis to bottom
            ax2.spines['bottom'].set_position(('outward', 20))
            ax2.set_xlabel(r"Minutes")

            # Plotar Figura 2
            axs[1].set_title("Otimização do número de atributos em {}".format(result.name))
            axs[1].plot(gen_n_individuals, gen_50_feature, '--', label=meta + " - Média", color=c, linewidth=1) 
            axs[1].fill_between(gen_n_individuals, gen_25_feature, gen_75_feature, alpha=0.1, color=c, label="25|75 percentile")
            axs[1].plot(gen_n_individuals, gen_min_feature, label=meta + "- Min", color=c, linewidth=0.5)
            axs[1].plot(gen_n_individuals, gen_max_feature, label=meta + "- Max", color=c, linewidth=0.5)
            axs[1].set_ylabel("# of Features")
            ax0 = [axs[0].get_xlim(), axs[0].get_ylim()]  
            ax1 = [axs[1].get_xlim(), axs[1].get_ylim()]  
            # Annotate Best Scores
            max_index = np.argmax(gen_max_matthews)
            max_score = max(gen_max_matthews)
            axs[0].annotate("{:0.2f}".format(max_score), (gen_n_individuals[max_index]+1, max_score - 0.05))
            axs[0].plot([gen_n_individuals[max_index], ] * 2, [-1, max_score], linestyle='-.', color=c, marker='x', markeredgewidth=1, ms=8)
            
            max_score = gen_max_feature[max_index]
            axs[1].annotate("%0.2f" % max_score, (gen_n_individuals[max_index]+ 1, max_score - 5))
            axs[1].plot([gen_n_individuals[max_index], ] * 2, [-100, max_score], linestyle='-.', color=c, marker='x', markeredgewidth=1, ms=8)

            #axs[0].set_xlim(0, 0.65)
            axs[0].set_ylim(0, 0.7)
            #axs[1].set_xlim(ax1[0])
            axs[1].set_ylim(0,80)

            plt.legend()
            print(home+name+str(n_experiments))
            plt.savefig(home+name+str(n_experiments))
            #plt.show()

            
