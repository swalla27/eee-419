# Example Decision Tree
# author: d updated by sdm

from m5_5_4_plotdr import plot_decision_regions             # for plotting
import matplotlib.pyplot as plt                         # add to plot
from sklearn import datasets                            # read the data
import numpy as np                                      # for arrays
from sklearn.model_selection import train_test_split    # split the data
from sklearn.tree import DecisionTreeClassifier         # the algorithm
from sklearn.tree import export_graphviz                # a cool graph
from sklearn.metrics import accuracy_score              # grade result

iris = datasets.load_iris()        # load the data
X = iris.data[:,[2,3]]             # extract the features we want to use
y = iris.target                    # extract the classes

# split the data
X_train, X_test, y_train, y_test = \
         train_test_split(X,y,test_size=0.3,random_state=0)

# create the classifier and train it
tree = DecisionTreeClassifier(criterion='entropy',max_depth=10 ,random_state=0)
tree.fit(X_train,y_train)

# combine the train and test data
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# see how we do on the combined data
y_combined_pred = tree.predict(X_combined)
print('Misclassified combined samples: %d' % \
       (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

# and visualize it
plot_decision_regions(X=X_combined, y=y_combined, classifier=tree,
                      test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.title('Decision Tree')
plt.show()

# This exports the file tree.dot. To view this file, on the Mac:
# Install graphviz: brew install graphviz
# NOTE: You may have to install brew first...
# Then execute: dot -T png -O tree.dot
# Then execute: open tree.dot.png
export_graphviz(tree,out_file='tree.dot',
                feature_names=['petal length', 'petal width'])
