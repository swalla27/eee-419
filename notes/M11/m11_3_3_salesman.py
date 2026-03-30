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
cities = np.loadtxt('/home/steven-wallace/Documents/asu/eee-419/notes/M11/m11_3_2_cities.txt', float)

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
        
    # swap them and calculate change in distance
    olddist = dist                                       # remember old distance
    cities[i,0],cities[j,0] =  cities[j,0],cities[i,0]   # swap the x values
    cities[i,1],cities[j,1] =  cities[j,1],cities[i,1]   # swap the y values
    dist = distance(cities)                              # calc the new distance
    deltadist = dist - olddist                           # calc the change
    
    # if move is rejected, swap back
    # if the new distance is smaller, then the random number is always smaller!
    if random() > np.exp(-deltadist/temp):               # if bigger, maybe keep
        cities[i,0],cities[j,0] =  cities[j,0],cities[i,0]  # fail - swap back
        cities[i,1],cities[j,1] =  cities[j,1],cities[i,1]
        dist = olddist                                      # restore old dist
        
end = timer()                             # calculate time taken
diff = end - start
print('Finished after {:5.2f}'.format(diff),'seconds')

print("Final length is",dist)
visualizepath("final",cities)             # and visualize the final distance
