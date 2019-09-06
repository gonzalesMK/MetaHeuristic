#!/home/gonzales/projeto_1/bin/python
#PBS -N JobSeq                                                                                                
#PBS -l select=2:ncpus=40                                                                                                   
#PBS -l walltime=12:00:00  
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Number of CPU to use in the cluster
n_cpus = 10

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
    
    # Load datasets
    home = os.environ["HOME"]
    file = open(home + '/datasets/' + string)
    dataset = np.asarray(pd.read_csv(file, delim_whitespace =True))
    y = dataset[:,-1]
    X = dataset[:,0:-1]
    
    # Label Encondig Y
    labelencoder_y = LabelEncoder()
    Y = labelencoder_y.fit_transform(y) 
    
    X = np.asarray(X, dtype= np.float64)
    
    # Cleaning variabels
    file.close()
    del dataset, y, home, file
    return X, Y

# This is the Experiment
def Experiment(meta,ind, name):
    import os
    from sklearn.model_selection import cross_validate
    from sklearn.base import clone
    from sklearn.metrics import matthews_corrcoef, make_scorer
    
    X, Y = setup(name)
    
    meta_with_params = meta(size_pop=60, number_gen=30, repeat=1,
                            make_logbook=True, verbose=1)
    param = {'normalize': True}

    result = cross_validate(estimator=meta_with_params, X=X, fit_params=param, pre_dispatch=10,
                            y=Y, cv=10, scoring=make_scorer(matthews_corrcoef), n_jobs=-1,
                            return_estimator=True, return_train_score='warn')

    
    # Log results from each run of cross_validate
    log_result = []
    for fit_time, meta, test_score, train_score in zip(result['fit_time'], result['estimator'], result['test_score'], result['train_score']):
        log = dict()
        log['gen_hof_'] = meta.gen_hof_
        log['gen_pareto_'] = meta.gen_pareto_
        log['best_mask_'] = meta.best_[0]
        log['logbook'] = meta.logbook
        log['params'] = meta.get_params()
        log['fit_time'] = fit_time
        log['test_score'] = test_score
        log['train_score'] = train_score

        log_result.append(log)
    
    # Save the results
    home = os.environ["HOME"]
    file = open(home + '/BBH/'+name+'teste_{:d}.save'.format(ind), 'wb')
    cPickle.dump(log_result, file, protocol=cPickle.HIGHEST_PROTOCOL)

    file.close()
    
    
if __name__ == "__main__":
    ## Import dataset

    import numpy as np
    from six.moves import cPickle
    from feature_selection import BinaryBlackHole
    from multiprocessing import Process,Manager
    import time
    np.warnings.filterwarnings('ignore')

  
    # Number of times to repeat the experiment
    n_repetition = 2
    
    # Number of experiments to run in parallel
    n_possible_jobs = n_cpus / 10 

    # Dataset list
    names = [   
                "BLCA.rnaseqv2.txt",
                "BRCA.rnaseqv2.txt",
                "CHOL.rnaseqv2.txt",
                "COAD.rnaseqv2.txt",
                "ESCA.rnaseqv2.txt",
                "HNSC.rnaseqv2.txt",
                "KICH.rnaseqv2.txt",
                "KIRC.rnaseqv2.txt",
                "KIRP.rnaseqv2.txt",
                "LIHC.rnaseqv2.txt",
                "LUAD.rnaseqv2.txt",
                "PRAD.rnaseqv2.txt",
                "LUSC.rnaseqv2.txt",
                "READ.rnaseqv2.txt",
                "STAD.rnaseqv2.txt",
                "THCA.rnaseqv2.txt",
                "UCEC.rnaseqv2.txt",
		]

    clusters_jobs=[]
    clusters=[]
    jobs=[]
    manager=Manager()

    # Create Jobs
    for j, name in enumerate(names):
        print(name)
        
        # Submit Jobs
        results = manager.dict()
        for k in np.arange(n_repetition):
            i = k + j * n_repetition
            job = Process(target=Experiment, args=(BinaryBlackHole,k,name))
            job.id = i
            jobs.append(job)
            time.sleep(0.05)
    del i
    started = 0        
    finished = 0
    active_jobs = 0
    
    # Manage jobs and delete then after finished
    while jobs != []:
        
        active_jobs = started - finished
        to_do_jobs = len(jobs)
        
        if active_jobs < n_possible_jobs  and to_do_jobs >= n_possible_jobs:
            print("starting job: {}".format(started))
            jobs[started - finished].start()    
            started = started + 1
            time.sleep(1)
        
        for i in np.arange(n_possible_jobs):
            i = int(i)
            if jobs[i].is_alive() == False:
                print("Finished job: {}".format(finished))
                del jobs[i]
                finished = finished + 1

        time.sleep(0.5)