# author: allee updated by sdm

import numpy as np                         # packages
import matplotlib.pyplot as plt
import matplotlib.cm as cm                 # color map
from timeit import default_timer as timer  # track performance

# Constants
M = 100                                    # grid squares on a side
a = 1                                      # 1 grid
eps = 8.854e-12                            # permittivity of free space F/m
k = a**2/(4*eps)                           # precompute since reused
target = 1e-6                              # target accuracy

# Create arrays to hold potential values
phi = np.zeros([M+1,M+1],float)            # start with 0 everywhere
rho = np.zeros([M+1,M+1],float)            # where fixed charges are
rho[25,25] = k * 10e-12                    # positive plate:  10 pC/m2
rho[75,75] = k * -10e-12                   # negative plate; -10 pC/m2
phiprime = np.zeros([M+1,M+1],float)       # init to 0 so ignore 0/M
iterations = 0                             # track number of iterations
print_lim = 1                              # initial print limit
start = timer()                            # start time of first loop
delta = 1.0                                # init so we get into the loop

# Main loop
while delta > target:                      # are we done?
    #calculate new values of the potential
    for i in range(1,M):       # edges are grounded, so no need to evaluate
        for j in range(1,M):   # edges are grounded, so no need to evaluate
            phiprime[i,j] = ( ( phi[i+1,j] + phi[i-1,j] +
                                phi[i,j+1] + phi[i,j-1] ) / 4 ) + rho[i,j]
    
    # calculate maximum difference from old values
    delta = np.max(np.abs(phi-phiprime))
    
    # swap the two arrays
    phi,phiprime = phiprime,phi
    iterations += 1
    if delta < print_lim:                  # if less than limit, print!
        end = timer()                      # calculate time taken
        diff = end - start
        print('{:.10f}'.format(delta), 'after',
              '{:5d}'.format(iterations), 'iterations in',
              '{:5.2f}'.format(diff),'seconds')
        print_lim /= 10                    # wait for next power of 10
        start = end

plt.imshow(phi,cmap=cm.hot)                # create the plot
plt.colorbar()
plt.title('Poisson Solved with Jacobi Method')
plt.show()
