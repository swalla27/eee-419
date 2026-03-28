# allee updated by sdm
#  simultaneous solution to differential equations
#  dx/dt = S(y-x)
#  dy/dt = Rx - y - xz
#  dz/dt = xy - Bz

import numpy as np                        # import required packages
from scipy.integrate import odeint
import matplotlib.pyplot as plt

S = 10.0                                  # constant: sigma
R = 28.0                                  # constant: r
B = 8/3                                   # constant: b

################################################################################
# This is the function that computes each step for each of the three equations.#
# Inputs:                                                                      #
#    r - array holding the current value of each function                      #
#    t - the time vector - in this case it isn't used but odeint requires it!  #
# Outputs:                                                                     #
#    an array holding the value computed for each function for r and t         #
################################################################################

def lorenz(r,t):
    x = r[0]
    y = r[1]
    z = r[2]
    fx = S*(y-x)
    fy = (R*x) - y - (x*z)
    fz = (x*y) - (B*z)
    return np.array([fx,fy,fz],float)

t_start    = 0.0                                # range to analyze from t=0
t_end      = 50.0                               # to 50.0
num_points = 5000                               # steps

tpoints = np.linspace(t_start,t_end,num_points) # create array of steps

init = np.array([0.0,1.0,0.0],float)            # init vals for x, y, z
ans = odeint(lorenz,init,tpoints)               # solve it!

xpoints = ans[:,0]                              # extract the values
ypoints = ans[:,1]
zpoints = ans[:,2]

# Plot each of the functions versus t.
# Plot them individually as they lay on top of each other.

plt.plot(tpoints,xpoints)                       # x vs t
plt.ylabel('x',rotation=0,fontsize=16)
plt.xlabel('time',fontsize=16)
plt.title('x vs t')
plt.show()

plt.plot(tpoints,ypoints)                       # y vs t
plt.ylabel('y',rotation=0,fontsize=16)
plt.xlabel('time',fontsize=16)
plt.title('y vs t')
plt.show()

plt.plot(tpoints,zpoints)                       # z vs t
plt.ylabel('z',rotation=0,fontsize=16)
plt.xlabel('time',fontsize=16)
plt.title('z vs t')
plt.show()

# now plot z vs x to see the butterfly
plt.plot(xpoints,zpoints)
plt.ylabel('z',rotation=0,fontsize=16)
plt.xlabel('x',fontsize=16)
plt.title('z vs x: The Butterfly')
plt.show()
