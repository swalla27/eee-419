# example of Rutherford gold scattering experiment
# author: allee updated by sdm

import numpy as np                 # import packages
import matplotlib.pyplot as plt
from random import random

# constants
Z = 79                             # atomic number of gold
Q = 1.602e-19                      # charge on electron in C
E = 7.7e6*Q                        # energy of alpha particle in J
EPSILON0 = 8.854e-12               # permittivity of free space MKS
A0 = 5.292e-11                     # Bohr radius
N = 1000000                        # number of particles

sigma = A0/100                     # for Gaussian distribution
two_sig_sq = -2 * sigma * sigma    # precompute

# if less than bcrit, then the particle is scattered
bcrit = Z*Q*Q/(2*np.pi*EPSILON0*E)

# main program - NOTE: don't need both random values from Gaussian - just r!
count = 0                          # number of particles backscattered
for i in range(N):                 # for each particle...

    r = np.sqrt(two_sig_sq * np.log(1.0-random()))  # derived formula

    if r < bcrit:                  # within the radius
        count += 1
        
print(count,' particles were reflected out of ',N)
