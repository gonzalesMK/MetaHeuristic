from sklearn.utils.estimator_checks import check_estimator
import sys
sys.path.append("..")
sys.path.insert(0, '')
#from skltemplate import HarmonicSearch
sys.path.insert(0, 'C\\Users\\Juliano D. Negri\\Documents\\Facul\\IC - Andre\\MetaHeuristic\\library\\binary_feature_selection')

from metaheuristics import HarmonicSearch, GeneticAlgorithm
from sklearn.svm import  SVC

if __name__ == "__main__":
    check_estimator(GeneticAlgorithm)        
            
    