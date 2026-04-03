# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 3 April 2026

# Homework 591

# I did not use AI at all to complete this assignment

import pandas as pd
import numpy as np
import sys

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
    """This function will print the results of a given ML algorithm to the terminal. It accepts the model itself and the model name as inputs, and
       will print the test + combined accuracy to the terminal. It does access the X and y variables, but we throw away the predictions after.\n
       Those variables are not needed outside of this function, which returns nothing."""

    # Use the model to predict y values for the test data set.
    y_pred = model.predict(X_test_std)

    # Make stacks of the X and y variables which contain both test and training data.
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # Use the model to predict every data point, including both test and training. This is used for the accuracy score.
    y_combined_pred = model.predict(X_combined_std)

    # Print the accuracy scores to the terminal, one test accuracy and one combined accuracy for each model.
    print(f'\n{model_name} Results:')
    print(f'\tAccuracy: {accuracy_score(y_test, y_pred):.2f}')
    print(f'\tCombined Accuracy: {accuracy_score(y_combined, y_combined_pred):.2f}')

# First, read the heart data csv into a df.
DATA_FILE = '/home/steven-wallace/Documents/asu/eee-419/projects/proj1/proj1_data.csv'
df = pd.read_csv(DATA_FILE)

X = df.iloc[:,0:12] # Everything except for the final column.
y = df.iloc[:,13] # The final column, which is the output variable.

# Split into test and training data.
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

######################
##### Perceptron #####
######################

# Create and use the standard scalar object, which is used to normalize the training and test X values.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

# Create the perceptron model and store that in the variable "ppn" for use later.
ppn = Perceptron(max_iter=10, tol=1e-6, eta0=0.001,
                 fit_intercept=True, random_state=0)

# Fit the perceptron model based upon the training data.
ppn.fit(X_train_std, y_train)

# Call the function to print the results of the perceptron to the terminal.
print_results(ppn, 'Perceptron')

###############################
##### Logistic Regression #####
###############################

# Assign a c value of 1 (inverse of regularization strength) for use in the logistic regression algorithm. 
c_val = 1

# Create the logistic regression ML model and place that in the variable "lr".
lr = OneVsRestClassifier(LogisticRegression(C=c_val, solver='liblinear', random_state=0))

# Fit the logistic regression model using the training data.
lr.fit(X_train_std, y_train)

# Call the function to print the results of the logistic regression to the terminal.
print_results(lr, 'Logistic Regression')

########################################
##### Support Vector Machine (SVM) #####
########################################

# Assign a c value of 1.0 (regularization parameter) for use in the SVM algorithm.
c_val = 1.0

# Create the SVC model and store it in the "svm" variable. 
svm = SVC(kernel='linear', C=c_val, random_state=0)

# Fit the svm model using the training data.
svm.fit(X_train_std, y_train)

# Call the function to print the results of the logistic regression to the terminal.
print_results(svm, 'SVM')

#########################
##### Decision Tree #####
#########################

# This line of code was necessary to get any of the following algorithms working. My understanding is that this 
# normalizes the X training data such that each column has a title. This is necessary in the decision tree and random forest
# algorithms, but not the preceding ones. This stack overflow link is where I got that idea from.
# https://stackoverflow.com/questions/74562712/warning-x-has-feature-names-but-decisiontreeclassifier-was-fitted-without-fea
X_train = StandardScaler().fit_transform(X_train)

# Create the decision tree model and store it in the variable "tree".
tree = DecisionTreeClassifier()

# Fit the decision tree using the training data.
tree.fit(X_train, y_train)

# Call the function to print the results of the decision tree to the terminal.
print_results(tree, 'Decision Tree')

#########################
##### Random Forest #####
#########################

# Assign the hyperparameter "number of trees" to 11.
trees = 11

# Create a random forest classifier object, and store it in the variable "forest".
forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                random_state=1, n_jobs=4)

# Fit the random forest model using the training data.
forest.fit(X_train, y_train)

# Call the function and print the results of the random forest to the terminal.
print_results(forest, 'Random Forest')

#####################################
##### K-Nearest Neighbors (KNN) #####
#####################################

# Assign the hyperparameter "number of neighbors" to 5.
n = 5

# Create a k-nearest neighbors classifier object and store it in the variable "knn".
knn = KNeighborsClassifier(n_neighbors=n, p=2, metric='minkowski')

# Fit the k-nearest neighbor model using the training data
knn.fit(X_train_std,y_train)

# Call the function and print the results of the KNN algorithm to the terminal.
print_results(knn, 'KNN')

##################################
##### Homework 591 Additions #####
##################################

# Logistic Regression (0.84)> Random Forest (0.80) > KNN (0.79)
# > SVM (0.79) > Perceptron (0.79) > Decision Tree (0.74) 

lr_predictions = lr.predict(X_test_std)
forest_predictions = forest.predict(X_test_std)
knn_predictions = knn.predict(X_test_std)
svm_predictions = svm.predict(X_test_std)
ppn_predictions = ppn.predict(X_test_std)
tree_predictions = tree.predict(X_test_std)

def custom_ensemble(*models):

    majority = np.zeros_like(models[0])
    number_entries = models[0].size
    number_models = len(models)
    for idx in range(number_entries):

        group_a = 0
        print('votes')
        for model in models:
            if model[idx] == 1:
                group_a += 1
            print(model[idx])

        if (group_a / number_models) > 0.5:
            majority[idx] = 1
        else:
            majority[idx] = 2
        print('decision')
        print(majority[idx])
        # sys.exit()
    
    return majority

ensemble3 = custom_ensemble(lr_predictions, forest_predictions, 
                            knn_predictions)
acc3 = accuracy_score(y_test, ensemble3)

ensemble4 = custom_ensemble(lr_predictions, forest_predictions, 
                            knn_predictions, svm_predictions)
acc4 = accuracy_score(y_test, ensemble4)

ensemble5 = custom_ensemble(lr_predictions, forest_predictions, 
                            knn_predictions, svm_predictions, ppn_predictions)
acc5 = accuracy_score(y_test, ensemble5)

