# infinite time ODE example using 4th-order Runge-Kutta
# author: olhartin@asu.edu updated by sdm

import numpy as np                       # import packages
import matplotlib.pyplot as plt

# dx/dt = 1/(x**2 + t**2) and solve from t=0 to t=infinity
# so let t = u/(1-u)

################################################################################
# Function to return the derivative of dx/du.                                  #
# Inputs:                                                                      #
#    x - current value                                                         #
#    u - the variable substituted for time                                     #
# Output:                                                                      #
#    return the derivative                                                     #
################################################################################

def ode_inf(x,u):
    return( 1 / ( ( x**2 * ( 1 - u )**2 ) + u**2 ) )

a = 0                                 # 0->infinity becomes 0->1
b = 1
N = 100                               # do 100 time steps
h = (b-a)/N                           # the step size

upoints = np.arange(a,b,h)            # create the array of times
tpoints = []                          # hold the values converted from u to t
xpoints = []                          # hold the values of x

x = 1.0                               # initial value
for u in upoints:                     # for each of the time steps...
    tpoints.append(u/(1-u))           # convert back to t
    xpoints.append(x)                 # add to the value list
    k1 = h*ode_inf(x,u)               # compute the Runge-Kutta factors
    k2 = h*ode_inf(x+0.5*k1,u+0.5*h)
    k3 = h*ode_inf(x+0.5*k2,u+0.5*h)
    k4 = h*ode_inf(x+k3,u+h)
    x += (k1+2*k2+2*k3+k4)/6          # get the next value
    
plt.plot(tpoints,xpoints)             # and plot the result
plt.xlim(0,80)
plt.xlabel('time t')
plt.ylabel('x(t)',rotation=0)
plt.grid()
plt.title('Infinite Range Example')
plt.show()
