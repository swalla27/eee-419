# example similar to example 9.2 and exercise 9.1 LaPlace Jacobi
# author: allee updated by sdm

import numpy as np                           # packages
import matplotlib.pyplot as plt
import matplotlib.cm as cm                   # color map
from timeit import default_timer as timer    # track performance

# Constants
M = 100                                      # grid squares on a side
a = 1                                        # 1 grid
eps = 8.854e-12                              # permittivity of free space F/m
k = a**2/(4*eps)                             # precompute since reused
target = 1e-6                                # target accuracy
w = 0.9                                      # overrelaxation parameter

# Create arrays to hold potential values
phi = np.zeros([M+1,M+1],float)              # start with 0 everywhere
rho = np.zeros([M+1,M+1],float)              # where the fixed charges are
rho[25,25] = k * 10e-12                      # positive plate:  10 pC/m2
rho[75,75] = k * -10e-12                     # negative plate: -10 pC/m2 
print_lim = 1                                # initial print limit
iterations = 0                               # track number of iterations
start = timer()                              # start time of first loop

# Main loop
flag = 1                                     # count how many exceed target
while flag != 0:
    flag = 0                                 # none exceed so far
    delta = 0                                # largest change so far

    #calculate new values of the potential
    for i in range(1,M):       # edges are grounded, so no need to evaluate
        for j in range(1,M):   # edges are grounded, so no need to evaluate
            phiold = phi[i,j]  # remember prior value for comparison

            phi[i,j] = ( (1+w)*((phi[i+1,j] + phi[i-1,j] +
                                 phi[i,j+1] + phi[i,j-1])/4 +
                                rho[i,j]) -
                         w*phi[i,j] )

            check = np.abs(phi[i,j]-phiold)  # check to see if too big...
            if check > target:               # track how many too large
                if check > delta:            # new largest difference
                    delta = check
                flag += 1                    # number that exceed target

    iterations += 1
    if delta < print_lim:                    # if less than limit, print!
        end = timer()                        # calculate time taken
        diff = end - start
        print('{:.10f}'.format(delta), 'after',
              '{:5d}'.format(iterations), 'iterations in',
              '{:5.2f}'.format(diff),'seconds')
        print_lim /= 10                      # wait for next power of 10
        start = end

#    print(flag)             # interesting to watch - eventually decreases fast

plt.imshow(phi,cmap=cm.hot)  # show the results
plt.colorbar()
plt.title('Poisson Solved with Gauss-Seidel Method')
plt.show()
