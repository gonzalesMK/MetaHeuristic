from sklearn.utils.estimator_checks import check_estimator
import sys
sys.path.append("..")
sys.path.insert(0, '')
#from skltemplate import HarmonicSearch
from metaheuristics import HarmonicSearch
from sklearn.svm import  SVC

def test_estimator():
    return 

if __name__ == "__main__":
    check_estimator(HarmonicSearch)
    