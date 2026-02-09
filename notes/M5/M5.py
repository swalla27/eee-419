# ##### Classify number using if statement:
# def classify_number(num):
#     if num > 0:
#         return +1  # Positive number
#     else:
#         return -1  # Negative number
# # Test the function with some inputs
# test_numbers = [-5, -1, 0, 2, 10]
# for num in test_numbers:
#     result = classify_number(num)
#     print(f"Input: {num}, Prediction: {'+1 (Positive)' if result == 1 else '-1 (Negative)'}")


# #### ML: Perceptron: Positive and Negative Numbers
# import numpy as np

# # Generate artificial training data (real numbers)
# np.random.seed(42)  # For reproducibility
# X_train = np.random.uniform(-10, 10, 100)  # 100 real numbers in range [-10, 10]
# Y_train = np.where(X_train >= 0, 1, -1)  # Labels: +1 if positive, -1 if negative

# # Convert X_train to 2D array since we have a single feature
# X_train = X_train.reshape(-1, 1)

# # Perceptron Class
# class Perceptron:
#     def __init__(self, learning_rate=0.1, epochs=10):
#         self.learning_rate = learning_rate
#         self.epochs = epochs
#         self.weights = None
#         self.bias = None

#     def train(self, X, Y):
#         n_samples, n_features = X.shape
#         self.weights = np.zeros(n_features)
#         self.bias = 0

#         for _ in range(self.epochs):
#             for i in range(n_samples):
#                 # Perceptron rule: y = sign(Wx + b)
#                 y_pred = np.sign(np.dot(self.weights, X[i]) + self.bias)
#                 if y_pred == 0:
#                     y_pred = 1  # Handle zero case

#                 # Update weights if prediction is wrong
#                 if Y[i] != y_pred:
#                     self.weights += self.learning_rate * Y[i] * X[i]
#                     self.bias += self.learning_rate * Y[i]

#     def predict(self, X):
#         return np.sign(np.dot(X, self.weights) + self.bias)

# # Initialize and train the perceptron
# perceptron = Perceptron(learning_rate=0.1, epochs=10)
# perceptron.train(X_train, Y_train)

# # Test with new numbers
# test_numbers = np.array([-5, -1, 0, 2, 10]).reshape(-1, 1)
# predictions = perceptron.predict(test_numbers)

# # Output results
# for num, pred in zip(test_numbers.flatten(), predictions):
#     print(f"Input: {num}, Prediction: {'+1 (Positive)' if pred == 1 else '-1 (Negative)'}")




# ##### ML: Perceptron: using Python's built-in package:
# import numpy as np
# # Use the sklearn package. Needs installation of scikit-learn package:
# # Execute the following command: pip install scikit-learn
# from sklearn.linear_model import Perceptron
# from sklearn.metrics import confusion_matrix

# # Generate artificial training data
# np.random.seed(42)
# X_train = np.random.uniform(-10, 10, 100).reshape(-1, 1)  # 100 real numbers
# Y_train = np.where(X_train >= 0, 1, -1)  # Labels: +1 if positive, -1 if negative

# # Create and train the Perceptron
# model1 = Perceptron(max_iter=1000, eta0=0.1, random_state=42)
# model1.fit(X_train, Y_train.ravel())

# # Test with new numbers
# X_test = np.array([-5, -1, 0, 2, 10])
# Y_test = np.where(X_test >= 0, 1, -1)  # Labels: +1 if positive, -1 if negative
# test_numbers = X_test.reshape(-1, 1)
# Y_pred_perceptron = model1.predict(test_numbers)

# # Print Confusion Matrix for Perceptron
# print("Confusion Matrix for Perceptron:")
# print(confusion_matrix(Y_test, Y_pred_perceptron))

# # Output results
# for num, pred in zip(test_numbers.flatten(), Y_pred_perceptron):
#     print(f"Input: {num}, Prediction: {'+1 (Positive)' if pred == 1 else '-1 (Negative)'}")



# ###### ML: Logistic Regression:
# import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import confusion_matrix

# # Generate artificial training data
# np.random.seed(42)
# X_train = np.random.uniform(-10, 10, 100).reshape(-1, 1)  # 100 real numbers
# Y_train = np.where(X_train >= 0, 1, -1)  # Labels: +1 if positive, -1 if negative

# # Create and train the Logistic Regression model
# model = LogisticRegression()
# model.fit(X_train, Y_train)

# # Test with new numbers
# X_test = np.array([-5, -1, 0, 2, 10])
# Y_test = np.where(X_test >= 0, 1, -1)  # Labels: +1 if positive, -1 if negative
# test_numbers = X_test.reshape(-1, 1)
# Y_pred_logistic = model.predict(test_numbers)

# # Print Confusion Matrix for Logistic Regression
# print("Confusion Matrix for Logistic Regression:")
# print(confusion_matrix(Y_test, Y_pred_logistic))

# # Output results
# for num, pred in zip(test_numbers.flatten(), Y_pred_logistic):
#     print(f"Input: {num}, Prediction: {'+1 (Positive)' if pred == 1 else '-1 (Negative)'}")



# ######## Perceptron - 2 Features:
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron, LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns
# from mlxtend.plotting import plot_decision_regions

# # Generate Data: Parabola-based separation
# np.random.seed(42)
# X = np.random.uniform(-2, 2, (200, 2))  # Random (x, y) points in range [-2, 2]
# y = np.where(X[:, 1] > X[:, 0]**2 - 1, 1, -1)  # Class +1 if above parabola, else -1

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ### 1. Perceptron Model ###
# perceptron = Perceptron(max_iter=1000, random_state=42)
# perceptron.fit(X_train, y_train)
# y_pred_perceptron = perceptron.predict(X_test)

# # Perceptron Evaluation
# print("\nPerceptron Accuracy:", accuracy_score(y_test, y_pred_perceptron))
# print("Perceptron Confusion Matrix:\n", confusion_matrix(y_test, y_pred_perceptron))

# ### Plot Decision Boundaries ###
# fig, axes = plt.subplots(1, 1, figsize=(15, 5))

# # Perceptron Decision Boundary
# axes.set_title("Perceptron")
# plot_decision_regions(X, y, clf=perceptron)

# plt.show()


# ######## Comparing Perceptron, Logistic Regression and SVM:
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.linear_model import Perceptron, LogisticRegression
# from sklearn.svm import SVC
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, confusion_matrix
# import seaborn as sns
# from mlxtend.plotting import plot_decision_regions

# # Generate Data: Parabola-based separation
# np.random.seed(42)
# X = np.random.uniform(-2, 2, (200, 2))  # Random (x, y) points in range [-2, 2]
# y = np.where(X[:, 1] > X[:, 0]**2 - 1, 1, -1)  # Class +1 if above parabola, else -1

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ### 1. Perceptron Model ###
# perceptron = Perceptron(max_iter=1000, random_state=42)
# perceptron.fit(X_train, y_train)
# y_pred_perceptron = perceptron.predict(X_test)

# # Perceptron Evaluation
# print("\nPerceptron Accuracy:", accuracy_score(y_test, y_pred_perceptron))
# print("Perceptron Confusion Matrix:\n", confusion_matrix(y_test, y_pred_perceptron))

# ### 2. Logistic Regression Model ###
# log_reg = LogisticRegression()
# log_reg.fit(X_train, y_train)
# y_pred_log_reg = log_reg.predict(X_test)

# # Logistic Regression Evaluation
# print("\nLogistic Regression Accuracy:", accuracy_score(y_test, y_pred_log_reg))
# print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))

# ### 3. SVM ###
# # svm_poly = SVC(kernel="poly", degree=2, C=1)
# # svm_poly = SVC(kernel="poly", degree=3, C=1)
# svm_poly = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2, C=10.0)
# svm_poly.fit(X_train, y_train)
# y_pred_svm_poly = svm_poly.predict(X_test)

# # SVM Evaluation
# print("\nSVM (Polynomial Kernel) Accuracy:", accuracy_score(y_test, y_pred_svm_poly))
# print("SVM Confusion Matrix:\n", confusion_matrix(y_test, y_pred_svm_poly))

# ### Plot Decision Boundaries ###
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# # Perceptron Decision Boundary
# axes[0].set_title("Perceptron")
# plot_decision_regions(X, y, clf=perceptron, ax=axes[0])

# # Logistic Regression Decision Boundary
# axes[1].set_title("Logistic Regression")
# plot_decision_regions(X, y, clf=log_reg, ax=axes[1])

# # SVM Decision Boundary
# axes[2].set_title("SVM (Poly, d=2)")
# plot_decision_regions(X, y, clf=svm_poly, ax=axes[2])

# plt.tight_layout()
# plt.show()


# ### Overfitting effect:
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.svm import SVC
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Generate synthetic classification data
# X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
#                             n_informative=2, n_redundant=0, n_clusters_per_class=1, 
#                             random_state=42)

# # Split data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# # Store accuracy results for different polynomial degrees
# accuracies = []

# # Loop through polynomial degrees from 1 to 10
# for degree in range(1, 11):
#     # Create SVM classifier with polynomial kernel
#     svm_model = SVC(kernel='poly', degree=degree, random_state=42)
    
#     # Train the model
#     svm_model.fit(X_train, y_train)
    
#     # Make predictions
#     y_pred = svm_model.predict(X_test)
    
#     # Calculate accuracy
#     accuracy = accuracy_score(y_test, y_pred)
#     accuracies.append(accuracy)

# # Plot accuracy vs degree
# plt.plot(range(1, 11), accuracies, marker='o', color='b')
# plt.title('SVM with Polynomial Kernel: Accuracy vs Degree')
# plt.xlabel('Degree of Polynomial')
# plt.ylabel('Accuracy')
# plt.xticks(range(1, 11))
# plt.grid(True)
# plt.show()


# ###############
# ##### Plotting data
# import pandas as pd                              # needed to read the data
# import matplotlib.pyplot as plt                  # used for plotting
# import seaborn as sns                            # data visualization
# import numpy as np

# iris = pd.read_csv('m5_2_7_iris.csv')                # load the data
# print(iris)

# # Extract the header column:
# cols = iris.columns
# print(cols)
# sepal_length = iris['sepal length']
# sepal_length = iris[cols[0]]
# print(cols[0])
# #Perfect positive correlation:
# plt.figure(1)
# plt.scatter(iris[cols[0]],iris[cols[1]])

# #Perfect negative correlation:
# plt.figure(2)
# plt.scatter(-iris[cols[0]],iris[cols[0]])

# # No correlation:
# plt.figure(3)
# plt.scatter(iris[cols[0]][0:30],iris[cols[4]][0:30])

# # positive correlation (but noisy):
# plt.figure(4)
# plt.scatter(iris[cols[3]],iris[cols[2]])

# # negative correlation (but noisy):
# plt.figure(5)
# plt.scatter(iris[cols[3]],-iris[cols[2]])

# plt.show()

# # ## Correlation Coefficient
# corr = iris.corr().abs()
# print('\n\n\ncorrelation coeff. matrix:\n\n',corr)
# ## Taking the upper triangular matrix only:
# corr *= np.tri(*corr.values.shape, k=-1).T
# corr_unstack = corr.unstack()
# corr_unstack.sort_values(inplace=True,ascending=False)
# print('\n\n\ncorrelation coefficient matrix sorted:\n\n',corr_unstack)

# sns.set(style='whitegrid', context='notebook')   # set the apearance
# sns.pairplot(iris,height=1.5)                    # create the pair plots
# plt.show()                                       # and show them



# ###################################
# ###### # Showing Overfitting vs Underfitting:

# from m5_4_3_plotdr import plot_decision_regions         # plotting function
# import matplotlib.pyplot as plt                     # so we can add to the plot
# import numpy as np                                  # needed for math
# from sklearn.svm import SVC                         # the algorithm
# from sklearn.model_selection import train_test_split   # splits database
# from sklearn.metrics import accuracy_score             # grade the results
# from numpy import logspace

# np.random.seed(0)                                   # so we can reproduce the example
# X_xor = np.random.randn(200,2)                      # generate 200x2 array

# # X_xor[:,0]>0 will be TRUE if the particular entry is greater than 0
# # while it will be FALSE if the entry is less than or equal to 0
# # Then the xor will work as expected
# y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)   # XOR the two columns

# # the where function will test the entry in y_xor. It will
# # replace TRUE with 1 and FALSE with -1.
# y_xor = np.where(y_xor, 1, -1)                      # convert T/F to 1/-1

# X_train, X_test, y_train, y_test = \
#          train_test_split(X_xor,y_xor,test_size=0.3,random_state=0)
# # # Now show the plot
# # plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1],c='b',marker='x',label='1')
# # plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],c='r',marker='v',label='-1')
# # plt.ylim(-3.0)
# # plt.legend()
# # plt.show()

# # Support Vector Machine
# gam_val_vec = logspace(-3,3,num=7)
# accuracy_train = np.zeros(len(gam_val_vec))
# accuracy_test = np.zeros(len(gam_val_vec))
# print(gam_val_vec)
# for ind in range(0,len(gam_val_vec)):
#     gam_val = gam_val_vec[ind]
#     print("\n\n\nResults for Gamma =",gam_val)
#     svm = SVC(kernel='rbf', tol=1e-3, random_state=0,
#               gamma=gam_val, C=10.0, verbose=True)
#     svm.fit(X_train, y_train)                           # apply the algorithm

#     y_train_pred = svm.predict(X_train)           # try with the train data
#     accuracy_train[ind] = accuracy_score(y_train, y_train_pred)
#     print('Accuracy of training: %.2f' % accuracy_train[ind])
#     y_pred = svm.predict(X_test)           # now try with the test data
#     accuracy_test[ind] = accuracy_score(y_test, y_pred)
#     print('Accuracy of testing: %.2f' % accuracy_test[ind])
#     # plot_decision_regions(X_xor,y_xor,classifier=svm)      # visualize!
#     # plt.legend(loc='upper left')
#     # plt.title('Kernel SVM Gamma = '+str(gam_val))
#     # plt.show()
#     # print("")

# plt.semilogx(gam_val_vec,accuracy_train,label='Training')
# plt.semilogx(gam_val_vec,accuracy_test,label='Testing')
# plt.xlabel('gamma value')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()