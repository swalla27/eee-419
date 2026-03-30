# example of generating Gaussian random numbers
# author: allee updated by sdm

import numpy as np                                    # import packages
import matplotlib.pyplot as plt
from random import random

# constants
N = 100                                               # number of particles
sigma = 1                                             # standard deviation

x = np.zeros(N,float)                                 # hold x values
y = np.zeros(N,float)                                 # hold y values

# generate two Gaussian numbers
for i in range(N):                                    # for each particle...
    # use the formula we derived
    r = np.sqrt(-2 * sigma * sigma * np.log(1.0-random()))
    theta = 2 * np.pi * random()                      # uniform between 0-2pi
    x[i] = r * np.cos(theta)                          # convert back
    y[i] = r * np.sin(theta)

plt.scatter(x,y)                                      # and plot the results
plt.xlabel('x')
plt.ylabel('y',rotation=0)
plt.title('Gaussian Distribution in X and Y')
plt.show()
