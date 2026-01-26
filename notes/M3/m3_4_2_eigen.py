# Example 6.2 as an eigenvalue problem
# author: allee updated by sdm
# calculate eigenvector and eigenvalues

import numpy as np                        # import math and arrays
import matplotlib.pyplot as plt           # import plotting
from numpy.linalg import eigvalsh, eigh   # import eigenvalue function

# constants
N = 26                                    # number of masses
m = 1.0                                   # mass of each mass
k = 6.0                                   # spring constant

# set up the coefficient array
coeff = np.eye(N,N)*2 - np.eye(N,N,k=1) - np.eye(N,N,k=-1)
np.set_printoptions(linewidth=120)        # make it so rows on one line!
print(coeff) 

# find eigenvalues
# eig_val = eigvalsh(coeff)

# find both eigenvalues and eigenvector
eig_val,eig_vec = eigh(coeff)
print(eig_val)
print(eig_vec)

# plot the energy vs the modes
plt.plot(np.sqrt(eig_val))                # sqrt is proportial to energy
plt.xlabel('mode')
plt.ylabel('\u221D energy')               # \u#### prints the unicode char
plt.title('\u221D Energy vs Mode')
plt.show()

# plot the first three frequencies
plt.plot(abs(eig_vec[0]))
plt.plot(abs(eig_vec[1]))
plt.plot(abs(eig_vec[2]))
plt.xlabel('mass number - 1')
plt.ylabel('position')
plt.title('First Three Frequencies - Fundamental Modes')
plt.show()
