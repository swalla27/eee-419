# example 9.1 LaPlace Jacobi
# author: allee updated by sdm

import numpy as np                         # packages
import matplotlib.pyplot as plt
import matplotlib.cm as cm                 # color map
from timeit import default_timer as timer  # track performance

# Constants
M = 100       # grid squares on a side
V1 = 1.0      # voltage at top wall (other walls are 0!)
target = 1e-6 # target accuracy

# Create arrays to hold potential values
phi = np.zeros([M+1,M+1],float)    # M squares mean M+1 points
phi[0,:] = V1                      # top row is all V1; others 0
phiprime = np.zeros([M+1,M+1],float)
phiprime[0,:] = V1                 # top row is all V1; others 0

# Main loop
delta = 1.0                        # init so we get into the loop
print_lim = 1                      # initial print limit
iterations = 0                     # track number of iterations

start = timer()                    # start time of first loop
while delta > target:              # until good enough...
    # calculate new values of the potential
    # note ranges don't include first or last in rows or columns!
    for i in range(1,M):
        for j in range(1,M):  # new value is average of surrounding
            phiprime[i,j] = ( phi[i+1,j] + phi[i-1,j] +
                              phi[i,j+1] + phi[i,j-1] ) / 4
    
    # calculate maximum difference from old values
    delta = np.max(np.abs(phi-phiprime))
    
    # swap the two arrays; ~faster than phi[:,:] = phiprime[:,:]
    phi,phiprime = phiprime,phi
    iterations += 1
    if delta < print_lim:          # if less than limit, print!
        end = timer()              # calculate time taken 
        diff = end - start
        print('{:.10f}'.format(delta), 'after',
              '{:5d}'.format(iterations), 'iterations in',
              '{:5.2f}'.format(diff),'seconds')
        print_lim /= 10             # wait for next power of 10
        start = end

plt.imshow(phi,cmap=cm.hot)  # and show the results
plt.colorbar()
plt.title('Laplace Solved with Jacobi Method')
plt.show()
