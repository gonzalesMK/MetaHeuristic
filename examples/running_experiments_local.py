#!/home/gonzales/projeto_1/bin/python
#PBS -N JobSeq                                                                                                
#PBS -l select=2:ncpus=40                                                                                                   
#PBS -l walltime=12:00:00  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

"""
    The goal of this code is to choose N algorithms and M datasets and perfom Feature Selection 
"""
# Number of CPU to use in the cluster
n_cpus = 8

# Run after the experiment is done
def cleanup():
    pass

# Setup the Dataset 
def setup(string):
    import os
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder
    
    
    # Load datasets: they are all 'pandas.core.frame.DataFrame'
    home = "/media/gonzales/DATA/TCC_Andre/datasets/"
    filename = home + string + ".xz"
    dataframe = pd.read_csv(filename).dropna()
    
    dataframe_Y = dataframe.pop('class')
    
    # OneHotEncoder categorical variables
    is_categorical = dataframe.keys()[dataframe.dtypes == 'object']
    dataframe_X = pd.get_dummies(dataframe, prefix=is_categorical)
    
    # Creat Matrix    
    
    y = np.array(dataframe_Y)
    X = np.array(dataframe_X, dtype= np.float64)
    
    # Label Encondig Y
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(y) 
    
    # Cleaning variabels
    #file.close()

    return X, Y

# This is the Experiment
def Experiment(meta,ind, name, params):
    import os
    from sklearn.model_selection import cross_validate
    from sklearn.base import clone
    from sklearn.metrics import matthews_corrcoef, make_scorer
    import lzma # For compression
    
    X, Y = setup(name)
    
    meta_with_params = meta()
    meta_with_params.set_params(**params)

    param = {'normalize': True}

    result = cross_validate(estimator=meta_with_params, X=X, fit_params=param, pre_dispatch=5,
                            y=Y, cv=10, scoring=make_scorer(matthews_corrcoef), n_jobs=5,
                            return_estimator=True, return_train_score='warn', error_score='raise')

    
    # Log results from each run of cross_validate
    log_result = []
    for fit_time, meta, test_score, train_score in zip(result['fit_time'], result['estimator'], result['test_score'], result['train_score']):
        log = dict()
        log['gen_hof_'] = meta.gen_hof_
        log['gen_pareto_'] = meta.gen_pareto_
        log['best_pareto'] = meta.best_pareto_front_
        log['best_mask_'] = meta.best_[0]
        log['logbook'] = meta.logbook
        log['params'] = meta.get_params()
        log['fit_time'] = fit_time
        log['test_score'] = test_score
        log['train_score'] = train_score

        log_result.append(log)
    
    # Save the results
    home = "/media/gonzales/DATA/TCC_Andre/datasets/results/"
    
    
    # Do not override any older file with this new one :
    while( True):
        filename =  home + name +'/' + meta_with_params.name + '.{:d}.lzma'.format(ind)
        
        if( os.path.exists(filename) ):
            ind = ind + 1 
            continue

        try:
            file = lzma.open( filename, 'wb')
            
        except FileNotFoundError:
            os.mkdir(home+name)
            file = lzma.open(filename, 'wb')
        
        break
        
    cPickle.dump(log_result, file, protocol=cPickle.HIGHEST_PROTOCOL)

    file.close()
    
    
if __name__ == "__main__":
    ## Import dataset

    import numpy as np
    from six.moves import cPickle
    from feature_selection import BinaryBlackHole, RandomSearch, HarmonicSearch, HarmonicSearch2
    from multiprocessing import Process,Manager
    import time
    np.warnings.filterwarnings('ignore')

  
    # Number of times to repeat the experiment
    n_repetition = 5
    
    # Number of experiments to run in parallel
    n_possible_jobs = 1

    # Dataset list
    dataset_names =     [   
        "eucalyptus",
    ]

    # Metaheuristic list and its parameters
    metaheuristics = [
        (HarmonicSearch2,
          {'size_pop': 60,
           'HMCR': 0.95,
           'number_gen': 60*29,
           'repeat':1,
           'skip': 100,
           'make_logbook':True,
           'verbose':True
          }),
          (RandomSearch,
          {'size_pop':60,
           'number_gen':30,
           'repeat':1,
           'make_logbook':True,
           'verbose':True
          }),
          (HarmonicSearch,
          {'size_pop':60,
            'HMCR': 0.95,
            'skip': 100,
           'number_gen':60*29,
           'repeat':1,
           'make_logbook':True,
           'verbose':True
          }),          
          ]
    clusters_jobs=[]
    clusters=[]
    jobs=[]
    manager=Manager()

    # Create Jobs
    id = -1
    for name in dataset_names:
        for meta, params in metaheuristics: 
            # Submit Jobs
            results = manager.dict()
            for k in np.arange(n_repetition):
                id = id + 1
                job = Process(target=Experiment, args=(meta,k,name, params), name= 'Process: ' +str(id))
                job.id = id
                jobs.append(job)
                time.sleep(0.05)

    started = 0        
    finished = 0
    active_jobs = 0
    
    # Manage jobs and delete then when finished
    while jobs != []:
        
        active_jobs = started - finished
        to_do_jobs = len(jobs) - active_jobs
        
        if active_jobs < n_possible_jobs  and to_do_jobs > 0:
            print("starting job: {}".format(started))
            jobs[started - finished].start()    
            started = started + 1
            time.sleep(0.5)
        
        for i in np.arange(active_jobs):
            i = int(i)
            if jobs[i].is_alive() == False:
                print("Finished job: {}".format(finished))
                del jobs[i]
                finished = finished + 1
                break

        time.sleep(0.5)