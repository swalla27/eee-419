# Example showing Density-Based clustering analysis
# author: d updated by sdm

import matplotlib.pyplot as plt             # needed for plotting
from sklearn.cluster import KMeans          # center-based analysis
from sklearn.datasets import make_moons     # create moon-shaped data sets
from sklearn.cluster import DBSCAN          # density-based analysis

# make_moons ALWAYS makes 2 interleaving half circles!
# will make 200 samples; noise is the standard deviation of Gaussian noise
X,y = make_moons(n_samples=200,noise=0.05,random_state=0)
plt.scatter(X[:,0],X[:,1],c='blue')
plt.title('DBSCAN data')
plt.show()                     # and show the moons

# use the KMeans algorithm and plot it - it fails to do a good job!
# (See comments in pml312 for an explanation of the parameters...)
km = KMeans(n_clusters=2,init='k-means++',random_state=0)
y_km = km.fit_predict(X)
plt.scatter(X[y_km==0,0],X[y_km==0,1],s=40,
            c='blue',marker='o',label='cluster 1')
plt.scatter(X[y_km==1,0],X[y_km==1,1],s=40,c='red',marker='s',label='cluster 2')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
            s=250,c='black',marker='*',label='centroids')
plt.legend()
plt.title('kmeans Attempting DBSCAN data')
plt.show()

# Now do density-based analysis
# eps is max distance for two samples to be considered in the same neighborhood
# eps is considered the most important parameter for dbscan!
# min_samples is the minimum number of samples in a neighborhood to form a core
# A cluster must be around a set of "core" samples.
# metric is how to measure the distance between points.

db = DBSCAN(eps=0.2,min_samples=5,metric='euclidean')
y_db = db.fit_predict(X)      # fit and predict the cluster labels

# and now plot the separated clusters
plt.scatter(X[y_db==0,0],X[y_db==0,1],
            c='blue',marker='o',s=40,label='cluster 1')
plt.scatter(X[y_db==1,0],X[y_db==1,1],c='red',marker='s',s=40,label='cluster 2')
plt.legend()
plt.title('DBSCAN')
plt.show()
