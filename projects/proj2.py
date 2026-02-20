# Steven Wallace
# Dr. Ewaisha
# EEE 419
# 19 February 2026

# Project 2

# I did not use AI at all to complete this assignment

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from warnings import filterwarnings
filterwarnings('ignore')

##################################
##### Constants and Settings #####
##################################

# This random seed feeds into both the MLP Classifier and the test/train splitter.
RANDOM_SEED = 0

# Define the project folder, then construct paths to the data and output graph.
proj_folder = '/home/steven-wallace/Documents/asu/eee-419/projects'
data_path = os.path.join(proj_folder, 'proj2_data.csv')
graph_path = os.path.join(proj_folder, 'proj2_graph.png')

################################
##### Initial Data Shaping #####
################################

# Read the data from a csv file into a pandas dataframe.
df = pd.read_csv(data_path, header=None)

# This will drop the final column, because I want everything to be numeric. 
# That last column is strings and I don't want it.
df = df.loc[:, df.columns != 61]

# This will split the dataframe into inputs (columns 0-59) and the output (column 60).
X = df.loc[:, df.columns != 60]
y = df.loc[:, 60]

# I use the number of columns in my for loop below.
_, num_cols = X.shape

####################################
##### Machine Learning Section #####
####################################

# I am initializing empty dictionaries to store information about each run of the for loop.
# Technically, only the test_accuracies and conf_matrices variables are necessary, but I found the others useful during testing.
test_accuracies = dict()
comb_accuracies = dict()
num_missed_mines = dict()
conf_matrices = dict()

# This loop will train a machine learning algorithm, test it, and collect important output variables into dictionaries.
# Each iteration uses a different number of columns in the input space. The first one will only use a single column, and
# the second will use two. It will continue in this way until all columns are included, just as described in the prompt.

for N in range(1, num_cols+1):

    # This will filter the input data based on the variable N. Each iteration takes a different number of input columns.
    X_filtered = X.loc[:, :N]

    # Create the test and training split based upon the filtered input data.
    X_train, X_test, y_train, y_test = \
         train_test_split(X_filtered, y, test_size=0.3, random_state=RANDOM_SEED)
    
    # This section will use a standard scalar to transform the data.
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    # This section will define an instance of the MLPClassifier class with the listed settings.
    # I did test this out with the "GridSearchCV" function and a ton of different combinations,
    # and this was the best combination that I found. More details are included in the report.
    model = MLPClassifier(hidden_layer_sizes=(100), activation='relu', max_iter=2000, 
                        alpha=0.0001, solver='adam', tol=1e-5, random_state=RANDOM_SEED)

    # This will fit the model based upon the training data.
    model.fit(X_train, y_train)

    # Use the model to predict the outputs for the test data.
    y_pred = model.predict(X_test)

    # Find the test accuracy by comparing the actual outputs with the predicted outputs.
    test_accuracy = accuracy_score(y_test, y_pred)

    # Combine the training and testing data into input and output variables, then make predictions for the combined dataset.
    X_comb = np.vstack((X_train, X_test))
    y_comb = np.hstack((y_train, y_test))
    y_comb_pred = model.predict(X_comb)

    # Find the accuracy of the model for this combined dataset.
    comb_accuracy = accuracy_score(y_comb, y_comb_pred)

    # Create the confusion matrix for this iteration of the for loop.
    cmat = confusion_matrix(y_comb, y_comb_pred)

    # Update the dictionaries (test accuracy, combined accuracy, number of missed mines, and confusion matrices) with the output data.
    test_accuracies.update({N: test_accuracy})
    comb_accuracies.update({N: comb_accuracy})
    num_missed_mines.update({N: int(cmat[1][0])})
    conf_matrices.update({N: cmat})

    # Print a summary of the results for this iteration of the for loop as described in the prompt.
    print(f'Number of Components = {N}; Test Accuracy = {test_accuracy:.3f}')


#########################
##### Final Results #####
#########################

# Find the maximum test accuracy using that dictionary, then extract the N value which obtained it and the confusion matrix associated with it.
# I referenced this link below for the line of code that finds the key associated with the max value in this dictionary.
# https://www.geeksforgeeks.org/python/python-get-key-with-maximum-value-in-dictionary/
max_acc_Nvalue = sorted(test_accuracies, key=test_accuracies.get, reverse=True)[0]
max_acc = test_accuracies[max_acc_Nvalue]
optimal_cmat = conf_matrices[max_acc_Nvalue]

# Print a description of this run with the optimal test accuracy as descibed in the prompt.
print(f'\nMax Test Accuracy = {max_acc:.3f}; Number of Components Used = {max_acc_Nvalue}')
print(f'Optimal Confusion Matrix:\n{optimal_cmat}')

# Create a graph of the test accuracy vs the number of features included, complete with axis labels, a title, and grid.
plt.scatter(test_accuracies.keys(), test_accuracies.values())
plt.xlabel('Number of Components Used')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy for Mine Identification Task')
plt.grid(True)
plt.savefig(graph_path, dpi=300)
plt.show()