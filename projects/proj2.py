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
from sklearn.model_selection import GridSearchCV

from warnings import filterwarnings
filterwarnings('ignore')

RANDOM_SEED = 0

proj_folder = '/home/steven-wallace/Documents/asu/eee-419/projects'
data_path = os.path.join(proj_folder, 'proj2_data.csv')
graph_path = os.path.join(proj_folder, 'proj2_acc_graph.png')

df = pd.read_csv(data_path, header=None)

df = df.loc[:, df.columns != 61]


# new_col_names = dict()
# for idx, _ in enumerate(df.columns):
#     new_col_names.update({idx: 'Col' + str(idx)})

# df = df.rename(columns=new_col_names)

# corr = df.corr()
# corr *= np.tri(*corr.values.shape, k=-1).T
# corr_unstack = corr.unstack()
# x = corr_unstack.copy()
# x.sort_values(inplace=True, ascending=False)
# print(f'Top 10:\n{x[:10]}')
# print(f'Bottom 10:\n{x[-10:]}')

num_rows, num_cols = df.shape

X = df.loc[:, df.columns != 60]
y = df.loc[:, 60]


def custom_split(N: int, X: pd.DataFrame, y: pd.Series):

    X_filtered = X.loc[:, :N]

    X_train, X_test, y_train, y_test = \
         train_test_split(X_filtered, y, test_size=0.3, random_state=RANDOM_SEED)
    
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)

    return X_train, X_test, y_train, y_test


test_accuracies = dict()
comb_accuracies = dict()
num_missed_mines = dict()
conf_matrices = dict()




# for N in range(1, num_cols+1):
N = 50

X_train, X_test, y_train, y_test = custom_split(N, X, y)

model = MLPClassifier(hidden_layer_sizes=(100), activation='relu', max_iter=2000, 
                    alpha=0.0001, solver='adam', tol=1e-5, random_state=RANDOM_SEED)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_pred)

X_comb = np.vstack((X_train, X_test))
y_comb = np.hstack((y_train, y_test))

y_comb_pred = model.predict(X_comb)

comb_accuracy = accuracy_score(y_comb, y_comb_pred)

cmat = confusion_matrix(y_comb, y_comb_pred)

test_accuracies.update({N: test_accuracy})
comb_accuracies.update({N: comb_accuracy})
num_missed_mines.update({N: int(cmat[1][0])})
conf_matrices.update({N: cmat})

print(f'Number of Components = {N}; Test Accuracy = {test_accuracy:.3f}')




# results_df = pd.DataFrame([test_accuracies, comb_accuracies, num_missed_mines]).transpose()
# results_df.columns = ['Test Accuracies', 'Combined Accuracies', 'Number of Missed Mines']

# results_df.sort_values(by='Combined Accuracies', axis=0, ascending=True)

# print(results_df)

# max_acc_Nvalue = sorted(test_accuracies, key=test_accuracies.get, reverse=True)[0]
# max_acc = test_accuracies[max_acc_Nvalue]
# optimal_cmat = conf_matrices[max_acc_Nvalue]

# print(f'\nMax Test Accuracy = {max_acc:.3f}; Number of Components Used = {max_acc_Nvalue}')
# print(f'Optimal Confusion Matrix:\n{optimal_cmat}')

# plt.scatter(test_accuracies.keys(), test_accuracies.values())
# plt.xlabel('Number of Components Used')
# plt.ylabel('Test Accuracy')
# plt.title('Test Accuracy vs Number of Components Used')
# plt.grid(True)
# plt.savefig(graph_path, dpi=300)
# plt.show()

param_grid = {'hidden_layer_sizes': [10, 100, 200],
              'activation': ['relu', 'linear', 'logistic', 'tanh'],
              'max_iter': [1000, 2000, 5000],
              'alpha': [1e-4, 1e-5, 1e-6],
              'solver': ['lbfgs', 'sgd', 'adam'],
              'tol': [1e-3, 1e-4, 1e-5]}

model = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', max_iter=2000, 
                      alpha=0.00001, solver='adam', tol=0.0001, random_state=RANDOM_SEED)

gs_cv_model = GridSearchCV(model, param_grid, scoring='accuracy', 
                           cv=5, verbose=1, n_jobs=-2)


gs_cv_model.fit(X_train, y_train)
print('Best parameter set: %s' % gs_cv_model.best_params_)
print('CV Accuracy: %.3f' % gs_cv_model.best_score_)
clf = gs_cv_model.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))