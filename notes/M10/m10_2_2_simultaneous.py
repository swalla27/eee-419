# simultaneous solution to differential equations
# author: olhartin@asu.edu updated by sdm

# dx/dt = xy-x                     # equations to solve
# dy/dt = y - xy + sin(wt)**2

import numpy as np                 # import packages
import matplotlib.pyplot as plt

################################################################################
# Function which returns the derivatives of both variables wrt time.           #
# Inputs:                                                                      #
#    r - an array of the current values of [x,y]                               #
# Outputs:                                                                     #
#    an array of the derivatives of x and y                                    #
################################################################################

def f(r,t):
    x = r[0]                       # extract x and y for clarity
    y = r[1]
    fx = x*y - x                   # compute the derivatives
    fy = y - x*y + np.sin(t)**2
    return np.array([fx,fy],float) # and return them

a = 0                              # bounds for the problem; from a...
b = 10.0                           # to b...
N = 1000                           # in 1000 steps
h = (b-a)/N                        # and the step size

tpoints = np.arange(a,b,h)         # create the array of steps
xpoints = []                       # create empty lists to hold values
ypoints = []

r = np.array([1.0,1.0],float)      # initialize an array with starting values
for t in tpoints:                  # then for every step in time...
    xpoints.append(r[0])           # save the current values
    ypoints.append(r[1])
    k1 = h*f(r,t)                  # compute the 4th-order Runge-Kutta values
    k2 = h*f(r+0.5*k1,t+0.5*h)
    k3 = h*f(r+0.5*k2,t+0.5*h)
    k4 = h*f(r+k3,t+h)
    r += (k1+2*k2+2*k3+k4)/6       # and update the values

plt.plot(tpoints,xpoints,'r',label='x')   # and plot the results
plt.plot(tpoints,ypoints,'b',label='y')
plt.xlabel('t')
plt.ylabel('x,y',rotation=0)
plt.legend()
plt.title('Simultaneous Equations')
plt.show()
