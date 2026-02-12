# Random Forest example
# author: Allee, Hartin updated by sdm

from m5_5_4_plotdr import plot_decision_regions            # plot results
import matplotlib.pyplot as plt                        # add to plot
from sklearn import datasets                           # read data
import numpy as np                                     # for arrays
from sklearn.model_selection import train_test_split   # split data
from sklearn.ensemble import RandomForestClassifier    # the algorithm
from sklearn.metrics import accuracy_score             # grade result

iris = datasets.load_iris()           # read the data
X = iris.data[:,[2,3]]                # pick out the features to use
y = iris.target                       # pick out the classes

# split the data
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

for trees in [ 1, 5, 11, 101]:
    print("Number of trees: ",trees)
    # create the classifier and train it
    # n_estimators is the number of trees in the forest
    # the entropy choice grades based on information gained
    # n_jobs allows multiple processors to be used
    forest = RandomForestClassifier(criterion='entropy', n_estimators=trees, \
                                    random_state=1, n_jobs=4)
    forest.fit(X_train,y_train)

    y_pred = forest.predict(X_test)         # see how we do on the test data
    print('Number in test ',len(y_test))
    print('Misclassified samples: %d' % (y_test != y_pred).sum())

    print('Accuracy: %.2f \n' % accuracy_score(y_test, y_pred))

    # combine the train and test data
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    print('Number in combined ',len(y_combined))

    # see how we do on the combined data
    y_combined_pred = forest.predict(X_combined)
    print('Misclassified samples: %d' % (y_combined != y_combined_pred).sum())
    print('Combined Accuracy: %.2f' % \
          accuracy_score(y_combined, y_combined_pred))
    
    # and visualize the results
    plot_decision_regions(X=X_combined, y=y_combined, classifier=forest,
                          test_idx=range(105,150))
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.legend(loc='upper left')
    plt.title('Random Forest - ' + str(trees) + ' trees')
    plt.show()
    print("")
