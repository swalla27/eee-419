# Example ODE using 2nd order Runge-Kutta
# author: olhartin@asu.edu updated by sdm

# dx/dt = -x**3 + sin(t)

import numpy as np                   # import packages
import matplotlib.pyplot as plt

################################################################################
# This is the function which returns the derivative of our target.             #
# Inputs:                                                                      #
#    x - current value of variable                                             #
#    t - time                                                                  #
# Output:                                                                      #
#    returns the derivative, or f(x,t)                                         #
################################################################################

def ode_ex(x,t):
    return(-x**3 + np.sin(t))

a = 0                               # bounds for the problem; from a...
b = 10                              # to b...
N = 1000                            # want 1000 steps
h = (b-a)/N                         # and this is the step size
x = 0.0                             # and this is the initial value

tpoints = np.arange(a,b,h)          # create the array of steps
xpoints = []                        # the empty list of results

for t in tpoints:                   # at each of the steps
    xpoints.append(x)               # put in the current value
    k1 = h*ode_ex(x,t)              # compute the intermediate factors
    k2 = h*ode_ex(x+0.5*k1,t+0.5*h)
    x += k2                         # and the new value

plt.plot(tpoints,xpoints)           # and create a plot to show the results
plt.xlabel('time t')
plt.ylabel('x(t)',rotation=0)
plt.grid()
plt.title('2nd-Order Runge-Kutta')
plt.show()
