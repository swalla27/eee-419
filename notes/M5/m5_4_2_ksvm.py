# Example using kernel svm
# author: Allee, Hartin updated by sdm

from m5_3_3_plotdr import plot_decision_regions         # plotting function
import matplotlib.pyplot as plt                     # so we can add to the plot
import numpy as np                                  # needed for math
from sklearn.svm import SVC                         # the algorithm

np.random.seed(0)                                   # so we can reproduce the example
X_xor = np.random.randn(200,2)                      # generate 200x2 array

# X_xor[:,0]>0 will be TRUE if the particular entry is greater than 0
# while it will be FALSE if the entry is less than or equal to 0
# Then the xor will work as expected
y_xor = np.logical_xor(X_xor[:,0]>0,X_xor[:,1]>0)   # XOR the two columns

# the where function will test the entry in y_xor. It will
# replace TRUE with 1 and FALSE with -1.
y_xor = np.where(y_xor, 1, -1)                      # convert T/F to 1/-1

# Now show the plot
plt.scatter(X_xor[y_xor==1,0], X_xor[y_xor==1,1],c='b',marker='x',label='1')
plt.scatter(X_xor[y_xor==-1,0], X_xor[y_xor==-1,1],c='r',marker='v',label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

# NOTE: we didn't have to standardize due to how we created the data!
# NOTE: we didn't do train/test split as we are just illustrating the algorithm
#       in practice, we would have both standarized and split!

# Support Vector Machine
# Kernel rbf radial bias function
# gamma increases influence of each sample
# increasing C increases error penalties 

for gam_val in [.001, .1, 100]:
    print("Results for Gamma =",gam_val)
    svm = SVC(kernel='rbf', tol=1e-3, random_state=0,
              gamma=gam_val, C=10.0, verbose=True)
    svm.fit(X_xor, y_xor)                           # apply the algorithm

    plot_decision_regions(X_xor,y_xor,classifier=svm)      # visualize!
    plt.legend(loc='upper left')
    plt.title('Kernel SVM Gamma = '+str(gam_val))
    plt.show()
    print("")

 
