# Ex5.14 p187
# allee updated by millman

import numpy as np                               # import math functions
from scipy.integrate import dblquad              # import integration function
import matplotlib.pyplot as plt                  # import plotting functions

G   = 6.67408e-11 # gravitational constant
M   = 1           # 1 kg mass
SIG = 10          # density of sheet kg/m2
L   = 10          # side lengths in m

START = .05       # height range
END   = 10
STEP  = .01

FACTOR = G * M * SIG   # so only have to multiply once

################################################################################
# function to compute the gravitational force on the object                    #
# inputs:                                                                      #
#     y, x, z: distance in each direction from the plate's center              #
# output:                                                                      #
#     frc: the force computed                                                  #
################################################################################

def gforce(y,x,z):                     # inner integral variable must come first
    frc = FACTOR * z / ( x**2 + y**2 +z**2 )**1.5
    return frc

# scipy.integrate.dblquad(func, a, b, gfun, hfun, args=(),
#                         epsabs=1.49e-08, epsrel=1.49e-08)
# func - is a function of y, then x, that is func(y,x)
# a,b - is the range of x
# gfun - lower boundary curve in y, and is a function of x
# hfun - upper boundary curve in y, and is a funciton of x
# epsabs - absolute tolerance
# epsrel - relative tolerance
# args - must be a sequence, e.g. a list for dblquad!

z_vals = np.arange(START,END,STEP)         # the discrete points to sample
num_zs = len(z_vals)                       # how many there are
g_vals = np.zeros(num_zs,float)            # array to hold the answers
for index in range(num_zs):                # for the range of z values
    z = z_vals[index]                      # extract the current z

    # first limits are for the outer integral in x
    # second limits are for the inner integral in y which is a function of x
    # note they are just constants -L/2 and L/2
    res, err = dblquad(gforce,-L/2,L/2,    # compute force, discard the error
                       lambda x: -L/2, lambda x: L/2,
                       args=([z]))

    g_vals[index] = res     # add the answer the array

plt.plot(z_vals,g_vals)                   # and plot the result
plt.xlabel("distance from plane (m)")
plt.ylabel("gravitional force (N)")
plt.title('Gravitational Force')
plt.show()
