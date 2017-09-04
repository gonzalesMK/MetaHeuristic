from sklearn.utils.estimator_checks import check_estimator

from feature_selection import HarmonicSearch, GeneticAlgorithm

if __name__ == "__main__":
    check_estimator(GeneticAlgorithm)        
    check_estimator(HarmonicSearch)
    
    