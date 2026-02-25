# Example program for clustering analysis using KMeans which tries for
# groups of equal variance, minimizing sum of squares or distance from a center
# author: d updated by sdm

from sklearn.datasets import make_blobs        # create clusters of data
from sklearn.cluster import KMeans             # cluster analysis algorithm
import matplotlib.pyplot as plt                # so we can plot the data

# create 3 blobs (centers) using 2 features (n_features) and 150 samples.
# cluster_std is the standard deviation of each blob
# shuffle the samples in random order (rather than 0s, 1s, then 2s)

X,y = make_blobs(n_samples=150,n_features=2,centers=3,
                 cluster_std=0.5,shuffle=True,random_state=0)
#print(X,"\n",y)                    # debug print to show samples

plt.scatter(X[:,0],X[:,1],c='red',marker='o',s=50)     # plot the data
plt.grid()
plt.title('kmeans cluster data')
plt.show()

# now train using KMeans
# We will do this 5 times, specifying 1-5 clusters
# use smart method for initial cluster centers (k-means++)
# n_init is the number of times to run with different centroid seeds
# max_iter limits the number of times the algorithm will be run
# tol is the convergence limit
# Create the KMeans widget and then fit and predict

mkrs = ['s','o','v','^','x']   # markers to use for each cluster
clrs = ['orange','green','blue','purple','gold']
inertia = []                   # track the Sum of Squares Error (SSE)
for numcs in range(1,6):
    km = KMeans(n_clusters=numcs,init='k-means++',
                n_init=10,max_iter=300,tol=1e-4,random_state=0)
    y_km = km.fit_predict(X)
    inertia.append(km.inertia_)   # built-in measure of SSE

    for clustnum in range(numcs):
        # X[y_km==clustnum,0] says use the entry in X if the corresponding value
        # in y_km is equal to clustnum. Same for the x and y coordinates
        plt.scatter(X[y_km==clustnum,0],X[y_km==clustnum,1], # select samples
                    c=clrs[clustnum],                        # pick color
                    s=50,                                    # marker size
                    marker=mkrs[clustnum],                   # which marker
                    label='cluster'+str(clustnum+1))         # which cluster

    # plot the centers
    plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],
                s=250,c='red',marker='*',label='centroids')

    plt.legend()
    plt.grid()
    plt.title('kmeans with ' + str(numcs) + ' clusters')
    plt.show()

plt.plot(list(range(1,len(inertia)+1)),inertia,marker='x')
plt.xlabel('number of clusters')
plt.ylabel('inertia')
plt.title('kmeans cluster analysis')
plt.show()

#print(X)
#print(y_km)
