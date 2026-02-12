# Logistic Regression example of Iris data set
# author: d updated by sdm

from m5_3_3_plotdr import plot_decision_regions            # plotting function
import matplotlib.pyplot as plt                        # so we can add to plot
from sklearn import datasets                           # read the data sets
import numpy as np                                     # needed for arrays
from sklearn.model_selection import train_test_split   # splits database
from sklearn.preprocessing import StandardScaler       # standardize data
from sklearn.linear_model import LogisticRegression    # the algorithm

iris = datasets.load_iris()                 # load the data set
X = iris.data[:,[2,3]]                      # separate the features we want
y = iris.target                             # extract the classifications

# split the problem into train and test
X_train, X_test, y_train, y_test = \
                 train_test_split(X,y,test_size=0.3,random_state=0)

sc = StandardScaler()                  # create the standard scalar
sc.fit(X_train)                        # compute the required transformation
X_train_std = sc.transform(X_train)    # apply to the training data
X_test_std = sc.transform(X_test)      # and SAME transformation of test data

# create logistic regression component.
# C is the inverse of the regularization strength. Smaller -> stronger!
#    C is used to penalize extreme parameter weights.
# solver is the particular algorithm to use
# multi_class determines how loss is computed
#    ovr -> binary problem for each label

for c_val in [1,10,100]:
    lr = LogisticRegression(C=c_val, solver='liblinear', multi_class='ovr', random_state=0)
    lr.fit(X_train_std, y_train)         # apply the algorithm to training data

    # combine the train and test data
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    # plot the results
    plot_decision_regions(X=X_combined_std, y=y_combined, \
                          classifier=lr, test_idx=range(105,150))
    plt.xlabel('petal length [standardized]')
    plt.ylabel('petal width [standardized]')
    plt.legend(loc='upper left')
    plt.title('Logistic Regression C = '+str(c_val))
    plt.show()
