# Data Preprocessing Template
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X_train = pd.read_csv('arcene_train.data', delim_whitespace = True, header = None)
y_train = pd.read_csv('arcene_train.labels', delim_whitespace = True, header = None)
y_train = y_train.as_matrix().flatten()
X_test  = pd.read_csv('arcene_valid.DATA', delim_whitespace = True, header = None)
y_test  = pd.read_csv('arcene_valid.labels', delim_whitespace = True, header = None)
y_test  = (y_test.as_matrix()).flatten()

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#######################################################3

# Scoring Function
from sklearn.metrics import accuracy_score
def scorer( estimator, X, y):
    accuracy = accuracy_score(y_true = y, y_pred = estimator.predict(X), normalize = True)
    print(X.size/34)
    if( X.size/34 > 100 ):
        b = accuracy + 20 / (X.size/34 + 100)
    else:
        b = accuracy + 0.2
    return b

# Applying Recursive Feature Selection ( Wrap method) in SVM
from sklearn.feature_selection import RFECV
from sklearn.svm import  SVC
classifier = SVC(kernel = 'linear')
selector = RFECV(classifier, step = 50, cv = 3, verbose = 10, scoring = scorer)
selector.fit( X_train, y_train)
print("Optimal number of features : %d" % selector.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(selector.grid_scores_) + 1), selector.grid_scores_)
plt.show()

# Predict using the selected Features
selector.predict(X_test)


# Fitting SVM to the Training set
from sklearn.svm import  SVC
classifier = SVC(kernel = 'linear')
classifier.fit(X_train, y_train)

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy_score(y_true = y_test, y_pred = classifier.predict(X_test), normalize = True)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_train, classifier.predict(X_train)))
print(confusion_matrix(y_test, classifier.predict(X_test)))


# Applying K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score( estimator = classifier, X = X_test, y = y_test, cv = 10)
print(accuracies.mean())
print(accuracies.std())


# Build the Backward Elimination Filter method ( 4regression ) (?) 
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((len(X_train),1)), values = X_train,  axis = 1) 
X_opt = X
regressor_OLS = sm.OLS( endog = y_train, exog = X_opt).fit()
regressor_OLS.summary()

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA( n_components = 20)
X_train = pca.fit_transform(X_train)
X_test =  pca.transform(X_test)
reseexplained_variance = pca.explained_variance_ratio_

## Plot Statistics
    # Plottig Estatistics 
    if( graph ):
        gen = logbook.select("gen")
        acc_mins = logbook.chapters["Accuracy"].select("min")
        acc_maxs = logbook.chapters["Accuracy"].select("max")
        acc_avgs = logbook.chapters["Accuracy"].select("avg")
        n_feat = logbook.chapters["N"].select("avg")
        
        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, acc_mins, "r-", label="Minimun Acc")
        line2 = ax1.plot(gen, acc_maxs, "g-", label="Maximun Acc")
        line3 = ax1.plot(gen, acc_avgs, "b-", label="Average Acc")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Accuracy")
        
        ax2 = ax1.twinx()
        line4 = ax2.plot(gen, n_feat, "y-", label="Avg Features")
        ax2.set_ylabel("Size", color="y")
        for tl in ax2.get_yticklabels():
            tl.set_color("y")
        
        lns = line1 + line2 + line3 + line4
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")
        
        plt.show()
