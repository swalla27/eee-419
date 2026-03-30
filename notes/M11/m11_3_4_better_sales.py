# traveling salesman example
# author: allee updated by sdm

import numpy as np                     # import packages
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D    # special for plotting
from random import random, sample
from timeit import default_timer as timer  # track performance

N = 100            # number of cities
TMAX = 10.0        # starting temperature
TMIN = 1e-3        # minimum temperature
TAU = 1e4          # cooling rate

# function to calculate magnitude of vector; that is distance
def mag(x):
    return np.sqrt(x[0]*x[0] + x[1]*x[1])

# function to calculate total length of tour
def distance(cities):
    s = 0.0
    for i in range(N):
        s += mag(cities[i+1]-cities[i])  # yields 2-entry array of x,y distances
    return s
    
# graphics - draw the current path
def visualizepath(title,cities):
    plt.plot(cities[:,0],cities[:,1])
    plt.title(title)
    plt.show()
    return

# Load the cities from the file
cities = np.loadtxt('m12_cities.txt',float)

dist = distance(cities)             # this is the initial distance
print(dist)

# main loop
t = 0                               # initial time step
temp = TMAX                         # initial temperature
#visualizepath('initial',cities)     # show initial path

start = timer()                     # start time of first loop
while temp > TMIN:                  # while not too cold...
    t += 1                          # increment the time
    temp = TMAX*np.exp(-t/TAU)      # calculate the new temperature
    
    #if t % 5000 == 0:               # update visualization every 5000 moves
    #   visualizepath('iteration='+str(t),cities)
    
    # choose two cities to swap and make sure they are distinct
    # but never swap the start and end points!
    i,j = sample(range(1,N),2)      # select 2 samples without replacement
        
    # calc change in distance
    if ( j + 1 == i ):                           # just 1 apart so careful!
        deltadist  = -mag(cities[j]-cities[j-1])
        deltadist -= mag(cities[i]-cities[i+1])
        deltadist += mag(cities[j]-cities[i+1])
        deltadist += mag(cities[i]-cities[j-1])
    elif ( i + 1 == j ):                         # just 1 apart so careful!
        deltadist  = -mag(cities[i]-cities[i-1])
        deltadist -= mag(cities[j]-cities[j+1])
        deltadist += mag(cities[j]-cities[i-1])
        deltadist += mag(cities[i]-cities[j+1])
    else:
        deltadist  = -mag(cities[i]-cities[i-1])
        deltadist -= mag(cities[i]-cities[i+1])
        deltadist -= mag(cities[j]-cities[j-1])
        deltadist -= mag(cities[j]-cities[j+1])

        deltadist += mag(cities[i]-cities[j-1])
        deltadist += mag(cities[i]-cities[j+1])
        deltadist += mag(cities[j]-cities[i-1])
        deltadist += mag(cities[j]-cities[i+1])

    # if move is rejected, swap back
    # if the new distance is smaller, then the random number is always smaller!
    if ( deltadist < 0 ) or ( random() < np.exp(-deltadist/temp) ): # if bigger, maybe keep
        cities[i,0],cities[j,0] =  cities[j,0],cities[i,0]  # swap
        cities[i,1],cities[j,1] =  cities[j,1],cities[i,1]
        
end = timer()                             # calculate time taken
diff = end - start
print('Finished after {:5.2f}'.format(diff),'seconds')

print("Final length is",distance(cities)) # There was no need to track the path length!
visualizepath("final",cities)             # and visualize the final distance
