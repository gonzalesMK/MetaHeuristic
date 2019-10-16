# 02_create_schema.py
import sqlite3
import os
import lzma
from six.moves import cPickle
import numpy as np
import io

"""
    
"""
COLUNAS = ["experiment_id","algorithm","dataset","gen","accumulativeTime",
"size_max","size_min","size_avg","size_std","size_25_percentile","size_50_percentile","size_75_percentile",
"fitness_max","fitness_min","fitness_avg","fitness_std","fitness_25_percentile","fitness_50_percentile","fitness_75_percentile"]
# Funções para armazenar numpy array no sqlite3
def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

db_description_name = "experiments_descriptions"
db_name = "experiments"

sqlite3.register_adapter(np.ndarray, adapt_array) # Converts np.array to TEXT when inserting
sqlite3.register_converter("array", convert_array) # Converts TEXT to np.array when selecting

# 1) Abrindo o Banco de Dados
conn = sqlite3.connect('results.db', detect_types=sqlite3.PARSE_DECLTYPES) # conectando...
cursor = conn.cursor()

# 1.a) crianr a tabela  (Se necessário)
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='{}';".format(db_description_name)) # check if table exists
exists = len(cursor.fetchall()) != 0
if( not exists):
    cursor.execute("""
    CREATE TABLE {}(
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            estimator TEXT,
            number_gen INTERGER NOT NULL,
            filepath TEXT NOT NULL UNIQUE      
    );""".format(db_description_name))
    conn.commit()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='{}';".format(db_name)) # check if table exists
exists = len(cursor.fetchall()) != 0
if( not exists):
    cursor.execute("CREATE TABLE {}( ".format(db_name) + """
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            experiment_id,
            algorithm TEXT NOT NULL,
            dataset TEXT NOT NULL,
            gen INTEGER NOT NULL,
            accumulativeTime DOUBLE NOT NULL,
            size_max INTEGER NOT NULL ,
            size_min INTEGER NOT NULL,
            size_avg DOUBLE NOT NULL,
            size_std DOUBLE NOT NULL,
            size_25_percentile INTEGER NOT NULL,
            size_50_percentile INTEGER NOT NULL,
            size_75_percentile INTEGER NOT NULL,
            fitness_max DOUBLE NOT NULL ,
            fitness_min DOUBLE NOT NULL,
            fitness_avg DOUBLE NOT NULL,
            fitness_std DOUBLE NOT NULL,
            fitness_25_percentile DOUBLE NOT NULL,
            fitness_50_percentile DOUBLE NOT NULL,
            fitness_75_percentile DOUBLE NOT NULL,
            FOREIGN KEY (experiment_id) REFERENCES experiments_descriptions(id));
            """)
    conn.commit()

# 2) Salvando dados de um experimento na tabela:

# 2.a) Lista das pastas Experimentos por metaheurística 
home = "/mnt/BCD29E9DD29E5C08/TCC/Experiments/results/"
meta_names = os.listdir(home)

for i in reversed(range(len(meta_names))):
    name=meta_names[i]
    if( name.endswith(".py") or name.endswith(".png")):
        print("deleting {}".format(name))
        del meta_names[i]

for name in meta_names:
    
    # 2.b) Lista de experimento por dataset:    
    filenames = os.listdir(home + name)
    
    for filename in filenames:
        # 2.c) Abrindo um experimento
        try:
            file = lzma.open(home + name + "/" + filename, 'rb')
            print("Openned: ", home + name + "/" + filename)
            results = cPickle.load(file)
            file.close()
        except FileNotFoundError:
            print("ERROR WITH :", filename)
        
        hof_results = []
        features_results = []
        
        for k in range( len(results['estimator'])):
            result = results['estimator'][k]
            logbook = result.logbook[0]
            
            hof_results.append( [ log['hallOfFame'][0].fitness.values[0] for log in logbook[-10:-1]])
            features_results.append( [ sum(log['hallOfFame'][0]) for log in logbook[-10:-1]])
        
        # 3) Adicionar a descrição do experimento na tabela de descrição
        keys = []
        values=[]
        keys.append("filepath")
        values.append( home + name + "/" + filename + ' [' + str(k) +']')

        keys.append("test_score")
        values.append(np.mean(results['test_score']))
        
        # 3.a) Organize parameters and exclude not usefull ones
        for key,value in zip( result.get_params().keys(), result.get_params().values()): 
            if key.find("__") != -1: # remove unused parameters (__)
                continue
            if key == 'estimator' and value != None: # rearrange estimator parameters
                estimator_keys = list(value.get_params().keys())
                index = estimator_keys.index('verbose')
                estimator_values = list(value.get_params().values())
                estimator_keys.pop(index)  # remove the verbose parameter
                estimator_values.pop(index)
                keys = keys + estimator_keys + ["estimator"]
                values = values + estimator_values + [str(value.__class__.__name__)]
                continue
            keys.append(key)
            values.append(value)

        # 3.b) Remove none keys
        none_index = list(reversed(list(np.where(np.array(values) == None)[0])))
        list(map(values.pop, none_index))
        list(map(keys.pop, none_index))

        # 3.c) Check if all columns exists -> add the new ones in case they don't
        cursor.execute('PRAGMA table_info({})'.format(db_description_name))
        colunas = [tupla[1] for tupla in cursor.fetchall()]
        for key in keys:
            try:
                colunas.index(key)
            except ValueError as error:
                t = type(key)
                if  t == type('sr'):
                    t = 'TEXT'
                elif t == int:
                    t = 'INTEGER'
                elif t == bool:
                    t = 'BOOLEAN'
                elif t == float:
                    t = 'DOUBLE'
                # Add new column
                cursor.execute("""  
                ALTER TABLE {}
                ADD COLUMN {} {};
                """.format( db_description_name, key, t))
            
        # 3.d) Transform the dict Keys in Strings
        c_values = ' '.join(", " + "?" + "" for x in values)[1:]
        columns = ', '.join("`" + str(x).replace('/', '_') + "`" for x in keys)
        
        # 3.e) Add new row
        command = "INSERT INTO "+db_description_name+"(" + columns +")\nVALUES ("+c_values+")\n"
        cursor.execute(command, (values) )

        # 4) Add Experiment Results in the Database
        
        # 4.a) Organize all attributes values in one list
        gen_n_individuals = np.reshape( [log['gen'] for log in logbook[-10:-1]], (-1,1))
        gen_time = np.reshape( [ np.round((i['time']-logbook[0]['time'])/60) for i in logbook[-10:-1]], (-1,1))
        gen_25_matthews = np.reshape(np.percentile(hof_results ,q=25, interpolation='higher', axis=0), (-1,1))
        gen_50_matthews = np.reshape(np.percentile(hof_results ,q=50, interpolation='higher', axis=0), (-1,1))
        gen_75_matthews = np.reshape(np.percentile(hof_results ,q=75, interpolation='higher', axis=0), (-1,1))
        gen_min_matthews= np.reshape(np.min(hof_results, axis=0), (-1,1))
        gen_avg_matthews= np.reshape(np.min(hof_results, axis=0), (-1,1))
        gen_max_matthews= np.reshape(np.max(hof_results, axis=0), (-1,1))
        gen_25_feature  = np.reshape(np.percentile(features_results ,q=25, interpolation='higher', axis=0), (-1,1))
        gen_50_feature  = np.reshape(np.percentile(features_results ,q=50, interpolation='higher', axis=0), (-1,1))
        gen_75_feature  = np.reshape(np.percentile(features_results ,q=75, interpolation='higher', axis=0), (-1,1))
        gen_min_feature = np.reshape(np.min(features_results, axis=0), (-1,1))
        gen_avg_feature = np.reshape(np.min(features_results, axis=0), (-1,1))
        gen_max_feature = np.reshape(np.max(features_results, axis=0), (-1,1))

        gen_std_feature = np.reshape(np.max(features_results, axis=0), (-1,1))
        gen_std_matthews= np.reshape(np.max(features_results, axis=0), (-1,1))
        
        # 4.b) Get experiment_id from the description
        cursor.execute("SELECT last_insert_rowid()")
        experiment_id = cursor.fetchall()[0][0]

        attrList = np.concatenate((
            np.reshape([experiment_id]*len(gen_n_individuals), (-1,1)),
            np.reshape([name]*len(gen_n_individuals), (-1, 1)) ,
            np.reshape([filename.split('.')[0]]*len(gen_n_individuals), (-1,1)),
            gen_n_individuals,
            gen_time,
            gen_max_matthews, gen_min_matthews, gen_avg_matthews, gen_std_matthews, gen_25_matthews, gen_50_matthews, gen_75_matthews,
            gen_max_feature,   gen_min_feature, gen_avg_feature, gen_std_feature, gen_25_feature, gen_50_feature, gen_75_feature, ), axis=1)

        attrList = attrList.tolist()
            
        colunas_str = ', '.join("`" + str(x).replace('/', '_') + "`" for x in COLUNAS)
        values_str = ' '.join(", " + "?" + "" for x in COLUNAS)[1:]
        command = "INSERT INTO "+db_name +"(" + colunas_str +")\nVALUES (" + values_str + ")\n"
        cursor.executemany(command, attrList)
        conn.commit()

        
conn.commit()
# desconectando...
conn.close()
