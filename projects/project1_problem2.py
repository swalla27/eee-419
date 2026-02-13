# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 13 February 2026

# Project 1 Problem 2

# I did not use AI at all to complete this assignment

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

############################################
##### Code common to all ML algorithms #####
############################################

def print_results(model, model_name: str):

    y_pred = model.predict(X_test_std)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    y_combined_pred = model.predict(X_combined_std)

    # prev_hash = 0
    # if prev_hash == hash(str(y_combined_pred)):
    #     print(f'{model_name} results identical.')
    # else:
    #     print(f'{model_name} results different.')
    # prev_hash = hash(str(y_combined_pred))

    print(f'\n{model_name} Results:')
    print(f'\tAccuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'\tCombined Accuracy: {accuracy_score(y_combined, y_combined_pred):.2f}')


# First, read the heart data csv into a df. 
df = pd.read_csv('projects/heart1.csv')

X = df.iloc[:,0:12] # Everything except for the final column
y = df.iloc[:,13] # The final column, which is the output variable

# Split into test and training data 
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

######################
##### Perceptron #####
######################

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=10, tol=1e-6, eta0=0.001,
                 fit_intercept=True, random_state=0)
ppn.fit(X_train_std, y_train)
print_results(ppn, 'Perceptron')

###############################
##### Logistic Regression #####
###############################

c_val = 1
lr = OneVsRestClassifier(LogisticRegression(C=c_val, solver='liblinear', random_state=0))
lr.fit(X_train_std, y_train)
print_results(lr, 'Logistic Regression')

########################################
##### Support Vector Machine (SVM) #####
########################################

c_val = 1.0
svm = SVC(kernel='linear', C=c_val, random_state=0)
svm.fit(X_train_std, y_train)
print_results(svm, 'SVM')

#########################
##### Decision Tree #####
#########################

# https://stackoverflow.com/questions/74562712/warning-x-has-feature-names-but-decisiontreeclassifier-was-fitted-without-fea
X_train = StandardScaler().fit_transform(X_train)

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
print_results(tree, 'Decision Tree')

#########################
##### Random Forest #####
#########################

trees = 11
forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                random_state=1, n_jobs=4)
forest.fit(X_train, y_train)
print_results(forest, 'Random Forest')

#####################################
##### K-Nearest Neighbors (KNN) #####
#####################################

n = 5
knn = KNeighborsClassifier(n_neighbors=n, p=2, metric='minkowski')
knn.fit(X_train_std,y_train)
print_results(knn, 'KNN')